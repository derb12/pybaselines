# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on August 4, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from ._algorithm_setup import _setup_splines
from .utils import _quantile_loss, relative_difference


def mixture_model(data, lam=100, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                  max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using a mixture of penalized splines and asymmetric least squares.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Default is 1e-2.
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
        will be an array with size equal to N and all values set to 1.

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
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        coef = spsolve(
            spl_basis.T * weight_matrix * spl_basis + penalty_matrix,
            spl_basis.T * (weight_array * y),
            permc_spec='NATURAL'
        )
        baseline = spl_basis.dot(coef)
        mask = y > baseline
        new_weights = mask * p + (~mask) * (1 - p)
        calc_difference = relative_difference(weight_array, new_weights)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

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
        baseline = spl_basis.dot(coef)
        calc_difference = relative_difference(old_coef, coef)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        old_coef = coef
        weight_array = _quantile_loss(y, baseline, quantile, eps)
        weight_matrix.setdiag(weight_array)

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

    return baseline, params
