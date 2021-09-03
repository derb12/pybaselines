# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on August 4, 2021
@author: Donald Erb

"""

from math import ceil
from functools import partial

import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from ._algorithm_setup import _setup_splines
from ._compat import jit
from .utils import _MIN_FLOAT, _quantile_loss, gaussian, relative_difference


@jit(nopython=True, cache=True)
def _assign_weights(bin_mapping, posterior_prob, residual):
    """
    Creates weights based on residual values within a posterior probabilty.

    Parameters
    ----------
    bin_mapping : numpy.ndarray, shape (N,)
        An array of integers that maps each item in `residual` to the corresponding
        bin index in `posterior_prob`.
    posterior_prob : numpy.ndarray, shape (M,)
        The array of the posterior probability that each value belongs to the
        gaussian distribution of the noise.
    residual : numpy.ndarray, shape (N,)
        The array of residuals.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weighting based on the residuals and their position in the posterior
        probability.

    Notes
    -----
    The code is not given by the reference; however, the reference describes the posterior
    probability and helps to understand how this weighting scheme is derived.

    References
    ----------
    de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
    Intelligent Laboratory Systems. 2012, 117, 56-60.

    """
    num_data = residual.shape[0]
    weights = np.empty(num_data)
    # TODO this seems like it would work in parallel, but it instead slows down
    for i in range(num_data):
        weights[i] = posterior_prob[bin_mapping[i]]

    return weights


@jit(nopython=True, cache=True)
def __mapped_histogram(data, num_bins, histogram):
    """
    Creates a normalized histogram of the data and a mapping of the indices.

    Parameters
    ----------
    data : numpy.ndarray, shape (N,)
        The data to be made into a histogram.
    num_bins : int
        The number of bins for the histogram.
    histogram : numpy.ndarray
        An array of zeros that will be modified inplace into the histogram.

    Returns
    -------
    bins : numpy.ndarray, shape (`num_bins` + 1)
        The bin edges for the histogram. Follows numpy's implementation such that
        each bin is inclusive on the left edge and exclusive on the right edge, except
        for the last bin which is inclusive on both edges.
    bin_mapping : numpy.ndarray, shape (N,)
        An array of integers that maps each item in `data` to its index within `histogram`.

    Notes
    -----
    `histogram` is modified inplace and converted to a probability density function
    (total area = 1) after the counting.

    """
    num_data = data.shape[0]
    bins = np.linspace(data.min(), data.max(), num_bins + 1)
    bin_mapping = np.empty(num_data, dtype=np.intp)
    bin_frequency = num_bins / (bins[-1] - bins[0])
    bin_0 = bins[0]
    last_index = num_bins - 1
    # TODO this seems like it would work in parallel, but it instead slows down
    for i in range(num_data):
        index = int((data[i] - bin_0) * bin_frequency)
        if index == num_bins:
            histogram[last_index] += 1
            bin_mapping[i] = last_index
        else:
            histogram[index] += 1
            bin_mapping[i] = index

    # normalize histogram such that area=1 so that it is a probability density function
    histogram /= (num_data * (bins[1] - bins[0]))

    return bins, bin_mapping


def _mapped_histogram(data, num_bins):
    """
    Creates a histogram of the data and a mapping of the indices.

    Parameters
    ----------
    data : numpy.ndarray, shape (N,)
        The data to be made into a histogram.
    num_bins : int
        The number of bins for the histogram.

    Returns
    -------
    histogram : numpy.ndarray, shape (`num_bins`)
        The histogram of the data, normalized so that its area is 1.
    bins : numpy.ndarray, shape (`num_bins` + 1)
        The bin edges for the histogram. Follows numpy's implementation such that
        each bin is inclusive on the left edge and exclusive on the right edge, except
        for the last bin which is inclusive on both edges.
    bin_mapping : numpy.ndarray, shape (N,)
        An array of integers that maps each item in `data` to its index within `histogram`.

    """
    # create zeros array outside of numba function since numba's implementation
    # of np.zeros is much slower than numpy's (https://github.com/numba/numba/issues/7259);
    # TODO once that numba issue is fixed, merge this with __mapped_histogram
    histogram = np.zeros(num_bins)
    bins, bin_mapping = __mapped_histogram(data, num_bins, histogram)

    return histogram, bins, bin_mapping


def _mixture_pdf(x, n, sigma, n_2=0, pos_uniform=None, neg_uniform=None):
    """
    The probability density function of a Gaussian and one or two uniform distributions.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the distribution.
    n : float
        The fraction of the distribution belonging to the Gaussian.
    sigma : float
        The standard deviation of the Gaussian distribution.
    n_2 : float, optional
        If `neg_uniform` or `pos_uniform` is None, then `n_2` is just an unused input.
        Otherwise, it is the fraction of the distribution belonging to the positive
        uniform distribution. Default is 0.
    pos_uniform : numpy.ndarray, shape (N,), optional
        The array of the positive uniform distributtion. Default is None.
    neg_uniform : numpy.ndarray, shape (N,), optional
        The array of the negative uniform distribution. Default is None.

    Returns
    -------
    numpy.ndarray
        The total probability density function for the mixture model.

    References
    ----------
    de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
    Intelligent Laboratory Systems. 2012, 117, 56-60.

    """
    # no error handling for if both pos_uniform and neg_uniform are None since this
    # is an internal function
    if neg_uniform is None:
        n1 = n
        n2 = 1 - n
        n3 = 0
        neg_uniform = 0
    elif pos_uniform is None:  # never actually used, but nice to have for the future
        n1 = n
        n2 = 0
        n3 = 1 - n
        pos_uniform = 0
    else:
        n1 = n
        n2 = n_2
        n3 = 1 - n - n_2
    # the gaussian should be area-normalized, so set height accordingly
    height = 1 / (max(sigma, _MIN_FLOAT) * np.sqrt(2 * np.pi))

    return n1 * gaussian(x, height, 0, sigma) + n2 * pos_uniform + n3 * neg_uniform


def mixture_model(data, lam=1e5, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                  max_iter=50, tol=1e-3, weights=None, symmetric=False, num_bins=None):
    """
    Considers the data as a mixture model composed of noise and peaks.

    Weights are iteratively assigned by calculating the probability each value in
    the residual belongs to a normal distribution representing the noise.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Used to set the initial weights before performing
        expectation-maximization. Default is 1e-2.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 3
        (third order differential matrix). Typical values are 2 or 3.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1, and then
        two iterations of reweighted least-squares are performed to provide starting
        weights for the expectation-maximization of the mixture model.
    symmetric : bool, optional
        If False (default), the total mixture model will be composed of one normal
        distribution for the noise and one uniform distribution for positive non-noise
        residuals. If True, an additional uniform distribution will be added to the
        mixture model for negative non-noise residuals. Only need to set `symmetric`
        to True when peaks are both positive and negative.
    num_bins : int, optional
        The number of bins to use when transforming the residuals into a probability
        density distribution. Default is None, which uses ``ceil(sqrt(N))``.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    References
    ----------
    de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
    Intelligent Laboratory Systems. 2012, 117, 56-60.

    """
    if not 0 < p < 1:
        raise ValueError('p must be between 0 and 1')
    # TODO maybe provide a way to find the optimal lam value for the spline,
    # using AIC, AICc, or generalized cross-validation
    # TODO figure out how to do the B.T * W * B and B.T * (w * y) operations using the banded
    # representations since that is the only part preventing using fully banded representations

    y, _, spl_basis, weight_array, penalty_matrix = _setup_splines(
        data, None, weights, spline_degree, num_knots, True, diff_order, lam
    )
    weight_matrix = diags(weight_array)
    if weights is None:
        # perform 2 iterations: first is a least-squares fit and second is initial
        # reweighted fit; 2 fits are needed to get weights to have a decent starting
        # distribution for the expectation-maximization
        for _ in range(2):
            coef = spsolve(
                spl_basis.T * weight_matrix * spl_basis + penalty_matrix,
                spl_basis.T * (weight_array * y),
                permc_spec='NATURAL'
            )
            baseline = spl_basis * coef
            mask = y > baseline
            weight_array = mask * p + (~mask) * (1 - p)
            weight_matrix.setdiag(weight_array)

    # now perform the expectation-maximization
    # TODO not sure if there is a better way to do this than transforming
    # the residual into a histogram, fitting the histogram, and then assigning
    # weights based on the bins; actual expectation-maximization uses log(probability)
    # directly estimates sigma from that, and then calculates the percentages, maybe
    # that would be faster/more stable?
    if num_bins is None:
        num_bins = ceil(np.sqrt(y.shape[0]))

    # uniform probability density distribution for positive residuals, constant
    # from 0 to max(residual), and 0 for residuals < 0
    pos_uniform_pdf = np.empty(num_bins)
    tol_history = np.empty(max_iter + 1)
    residual = y - baseline
    if symmetric:
        # the 0.2 * std(residual) is an "okay" estimate that at least
        # scales with the range of y
        fit_params = (0.5, 0.2 * np.std(residual), 0.25)
        bounds = ((0, 0, 0), (1, np.inf, 1))
        # create a second uniform pdf for negative residual values
        neg_uniform_pdf = np.empty(num_bins)
    else:
        fit_params = (0.5, 0.2 * np.std(residual))
        bounds = ((0, 0), (1, np.inf))
        neg_uniform_pdf = None
    for i in range(max_iter + 1):
        residual_hist, bin_edges, bin_mapping = _mapped_histogram(residual, num_bins)
        # average bin edges to get better x-values for fitting
        bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        pos_uniform_mask = bins < 0
        pos_uniform_pdf[~pos_uniform_mask] = 1 / max(abs(residual.max()), 1e-6)
        pos_uniform_pdf[pos_uniform_mask] = 0
        if symmetric:
            neg_uniform_mask = bins > 0
            neg_uniform_pdf[~neg_uniform_mask] = 1 / max(abs(residual.min()), 1e-6)
            neg_uniform_pdf[neg_uniform_mask] = 0

        # method is dogbox since trf gives a RuntimeWarning due to nan somewhere
        # occuring in the bounds
        fit_params = curve_fit(
            partial(_mixture_pdf, pos_uniform=pos_uniform_pdf, neg_uniform=neg_uniform_pdf),
            bins, residual_hist, p0=fit_params, bounds=bounds, check_finite=False,
            method='dogbox'
        )[0]
        if symmetric:
            uniform_pdf = (
                fit_params[2] * pos_uniform_pdf
                + (1 - fit_params[0] - fit_params[2]) * neg_uniform_pdf
            )
        else:
            uniform_pdf = (1 - fit_params[0]) * pos_uniform_pdf
        gaus_pdf = fit_params[0] * gaussian(
            bins, 1 / (fit_params[1] * np.sqrt(2 * np.pi)), 0, fit_params[1]
        )
        # no need to clip between 0 and 1 if dividing by _MIN_FLOAT since that
        # means the numerator is also 0
        posterior_prob = gaus_pdf / np.maximum(gaus_pdf + uniform_pdf, _MIN_FLOAT)
        new_weights = _assign_weights(bin_mapping, posterior_prob, residual)

        calc_difference = relative_difference(weight_array, new_weights)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break

        weight_array = new_weights
        weight_matrix.setdiag(weight_array)
        coef = spsolve(
            spl_basis.T * weight_matrix * spl_basis + penalty_matrix,
            spl_basis.T * (weight_array * y),
            permc_spec='NATURAL'
        )
        baseline = spl_basis * coef
        residual = y - baseline

    # TODO return spline coefficients? would be useless without the basis matrix or
    # the knot locations; probably don't want to return basis since it's not useful;
    # best thing would be return the inner knots and coefficients so that an equivalent
    # spline could be recreated with scipy, and let scipy handle extrapolation, etc.
    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

    return baseline, params


def irsqr(data, lam=100, quantile=0.05, num_knots=100, spline_degree=3, diff_order=3,
          max_iter=100, tol=1e-6, weights=None, eps=None):
    """
    Iterative Reweighted Spline Quantile Regression (IRSQR).

    Fits the baseline using quantile regression with penalized splines.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    quantile : float, optional
        The quantile at which to fit the baseline. Default is 0.05.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 3
        (third order differential matrix). Typical values are 3, 2, or 1.
    max_iter : int, optional
        The max number of fit iterations. Default is 100.
    tol : float, optional
        The exit criteria. Default is 1e-6.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.
    eps : float, optional
        A small value added to the square of the residual to prevent dividing by 0.
        Default is None, which uses the square of the maximum-absolute-value of the
        fit each iteration multiplied by 1e-6.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Raises
    ------
    ValueError
        Raised if quantile is not between 0 and 1.

    References
    ----------
    Han, Q., et al. Iterative Reweighted Quantile Regression Using Augmented Lagrangian
    Optimization for Baseline Correction. 2018 5th International Conference on Information
    Science and Control Engineering (ICISCE), 2018, 280-284.

    """
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1')

    y, _, spl_basis, weight_array, penalty_matrix = _setup_splines(
        data, None, weights, spline_degree, num_knots, True, diff_order, lam
    )
    weight_matrix = diags(weight_array)
    old_coef = np.zeros(spl_basis.shape[1])
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        coef = spsolve(
            spl_basis.T * weight_matrix * spl_basis + penalty_matrix,
            spl_basis.T * (weight_array * y),
            permc_spec='NATURAL'
        )
        baseline = spl_basis * coef
        calc_difference = relative_difference(old_coef, coef)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        old_coef = coef
        weight_array = _quantile_loss(y, baseline, quantile, eps)
        weight_matrix.setdiag(weight_array)

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

    return baseline, params
