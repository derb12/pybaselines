# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on April 25, 2023
@author: Donald Erb

"""

from functools import partial
from math import ceil
import warnings

import numpy as np
from scipy.optimize import curve_fit

from .. import _weighting
from ..utils import ParameterWarning, gaussian, relative_difference, _MIN_FLOAT
from ._algorithm_setup import _Algorithm2D
from ._whittaker_utils import PenalizedSystem2D
from .._compat import _HAS_NUMBA, jit


class _Spline(_Algorithm2D):
    """A base class for all spline algorithms."""

    @_Algorithm2D._register(sort_keys=('weights',))
    def mixture_model(self, data, lam=1e3, p=1e-2, num_knots=25, spline_degree=3, diff_order=3,
                      max_iter=50, tol=1e-3, weights=None, symmetric=False, num_bins=None):
        """
        Considers the data as a mixture model composed of noise and peaks.

        Weights are iteratively assigned by calculating the probability each value in
        the residual belongs to a normal distribution representing the noise.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Used to set the initial weights before performing
            expectation-maximization. Default is 1e-2.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines along the rows and columns, respectively. If a
            single value is given, both will use the same value. Default is 25.
        spline_degree : int or Sequence[int, int], optional
            The degree of the splines along the rows and columns, respectively. If a single
            value is given, both will use the same value. Default is 3, which is a cubic spline.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 3 (third order differential matrix). Typical values are 2 or 3.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1, and then
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
            density distribution. Default is None, which uses ``ceil(sqrt(M * N))``.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
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
        Intelligent Laboratory Systems, 2012, 117, 56-60.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        # scale y between -1 and 1 so that the residual fit is more numerically stable
        y_domain = np.polynomial.polyutils.getdomain(y.ravel())
        y = np.polynomial.polyutils.mapdomain(y, y_domain, np.array([-1., 1.]))

        if weights is not None:
            baseline = self.pspline.solve(y, weight_array)
        else:
            # perform 2 iterations: first is a least-squares fit and second is initial
            # reweighted fit; 2 fits are needed to get weights to have a decent starting
            # distribution for the expectation-maximization
            if symmetric and not 0.2 < p < 0.8:
                # p values far away from 0.5 with symmetric=True give bad initial weights
                # for the expectation maximization
                warnings.warn(
                    'should use a p value closer to 0.5 when symmetric is True',
                    ParameterWarning, stacklevel=2
                )
            for _ in range(2):
                baseline = self.pspline.solve(y, weight_array)
                weight_array = _weighting._asls(y, baseline, p)

        # now perform the expectation-maximization
        # TODO not sure if there is a better way to do this than transforming
        # the residual into a histogram, fitting the histogram, and then assigning
        # weights based on the bins; actual expectation-maximization uses log(probability)
        # directly estimates sigma from that, and then calculates the percentages, maybe
        # that would be faster/more stable?
        if num_bins is None:
            num_bins = ceil(np.sqrt(self._len[0] * self._len[1]))

        # uniform probability density distribution for positive residuals, constant
        # from 0 to max(residual), and 0 for residuals < 0
        pos_uniform_pdf = np.empty(num_bins)
        tol_history = np.empty(max_iter + 1)
        residual = y - baseline

        # the 0.2 * std(residual) is an "okay" starting sigma estimate
        fit_params = [0.5, np.log10(0.2 * np.std(residual))]
        bounds = [[0, -np.inf], [1, np.inf]]
        if symmetric:
            fit_params.append(0.25)
            bounds[0].append(0)
            bounds[1].append(1)
            # create a second uniform pdf for negative residual values
            neg_uniform_pdf = np.empty(num_bins)
        else:
            neg_uniform_pdf = None

        # convert bounds to numpy array since curve_fit will use np.asarray each iteration
        bounds = np.array(bounds)
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

            fit_func = partial(
                _mixture_pdf, pos_uniform=pos_uniform_pdf, neg_uniform=neg_uniform_pdf
            )
            # use dogbox method since trf gives RuntimeWarnings from nans appearing
            # somehow during optimization; trf is also prone to failure when symmetric=True
            fit_params = curve_fit(
                fit_func, bins, residual_hist, p0=fit_params, bounds=bounds,
                check_finite=False, method='dogbox'
            )[0]
            sigma = 10**fit_params[1]
            gaus_pdf = fit_params[0] * gaussian(bins, 1 / (sigma * np.sqrt(2 * np.pi)), 0, sigma)
            posterior_prob = gaus_pdf / np.maximum(fit_func(bins, *fit_params), _MIN_FLOAT)
            # need to clip since a bad initial start can erroneously set the sum of the fractions
            # of each distribution to > 1
            np.clip(posterior_prob, 0, 1, out=posterior_prob)
            new_weights = posterior_prob[bin_mapping].reshape(self._len)

            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

            weight_array = new_weights
            baseline = self.pspline.solve(y, weight_array)
            residual = y - baseline

        # TODO could potentially return a BSpline object from scipy.interpolate
        # using knots, spline degree, and coef, but would need to allow user to
        # input the x-values for it to be useful
        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        baseline = np.polynomial.polyutils.mapdomain(baseline, np.array([-1., 1.]), y_domain)

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def irsqr(self, data, lam=1e3, quantile=0.05, num_knots=25, spline_degree=3,
              diff_order=3, max_iter=100, tol=1e-6, weights=None, eps=None):
        """
        Iterative Reweighted Spline Quantile Regression (IRSQR).

        Fits the baseline using quantile regression with penalized splines.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 0.05.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines along the rows and columns, respectively. If a
            single value is given, both will use the same value. Default is 25.
        spline_degree : int or Sequence[int, int], optional
            The degree of the splines along the rows and columns, respectively. If a single
            value is given, both will use the same value. Default is 3, which is a cubic spline.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 3 (third order differential matrix). Typical values are 2 or 3.
        max_iter : int, optional
            The max number of fit iterations. Default is 100.
        tol : float, optional
            The exit criteria. Default is 1e-6.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        eps : float, optional
            A small value added to the square of the residual to prevent dividing by 0.
            Default is None, which uses the square of the maximum-absolute-value of the
            fit each iteration multiplied by 1e-6.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
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

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        old_coef = np.zeros(self.pspline._num_bases[0] * self.pspline._num_bases[1])
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve(y, weight_array)
            calc_difference = relative_difference(old_coef, self.pspline.coef)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            old_coef = self.pspline.coef
            weight_array = _weighting._quantile(y, baseline, quantile, eps)

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_asls(self, data, lam=1e3, p=1e-2, num_knots=25, spline_degree=3, diff_order=2,
                     max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the asymmetric least squares (AsLS) algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 1e-2.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines along the rows and columns, respectively. If a
            single value is given, both will use the same value. Default is 25.
        spline_degree : int or Sequence[int, int], optional
            The degree of the splines along the rows and columns, respectively. If a single
            value is given, both will use the same value. Default is 3, which is a cubic spline.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 1 or 2.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

        See Also
        --------
        Baseline2D.asls

        References
        ----------
        Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

        Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
        Leiden University Medical Centre Report, 2005, 1(1).

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve(y, weight_array)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_iasls(self, data, lam=1e3, p=1e-2, lam_1=1e-4, num_knots=25,
                      spline_degree=3, max_iter=50, tol=1e-3, weights=None, diff_order=2):
        """
        A penalized spline version of the IAsLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e1.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 1e-2.
        lam_1 : float, optional
            The smoothing parameter for the first derivative of the residual. Default is 1e-4.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 1. Default is 2
            (second order differential matrix). Typical values are 2 or 3.

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
            Raised if `p` is not between 0 and 1 or if `diff_order` is less than 2.

        See Also
        --------
        Baseline2D.iasls

        References
        ----------
        He, S., et al. Baseline correction for raman spectra using an improved
        asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        elif np.less(diff_order, 2).any():
            raise ValueError('diff_order must be 2 or greater')

        if weights is None:
            _, _, pseudo_inverse = self._setup_polynomial(
                data, weights=None, poly_order=2, calc_vander=True, calc_pinv=True
            )
            baseline = self.vandermonde @ (pseudo_inverse @ data.ravel())
            weights = _weighting._asls(data, baseline.reshape(self._len), p)

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        # B.T @ P_1 @ B and B.T @ P_1 @ y
        penalized_system_1 = PenalizedSystem2D(self._len, lam_1, diff_order=1)
        p1_partial_penalty = self.pspline.basis.T @ penalized_system_1.penalty

        partial_rhs = p1_partial_penalty @ y.ravel()
        self.pspline.add_penalty(p1_partial_penalty @ self.pspline.basis)

        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve(y, weight_array**2, rhs_extra=partial_rhs)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_airpls(self, data, lam=1e3, num_knots=25, spline_degree=3,
                       diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the airPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 25.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
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

        See Also
        --------
        Baseline2D.airpls

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam, copy_weights=True
        )

        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            try:
                output = self.pspline.solve(y, weight_array)
            except np.linalg.LinAlgError:
                warnings.warn(
                    ('error occurred during fitting, indicating that "tol"'
                     ' is too low, "max_iter" is too high, or "lam" is too high'),
                    ParameterWarning, stacklevel=2
                )
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            else:
                baseline = output

            residual = y - baseline
            neg_mask = residual < 0
            neg_residual = residual[neg_mask]
            if len(neg_residual) < 2:
                # exit if there are < 2 negative residuals since all points or all but one
                # point would get a weight of 0, which fails the solver
                warnings.warn(
                    ('almost all baseline points are below the data, indicating that "tol"'
                     ' is too low and/or "max_iter" is too high'), ParameterWarning,
                    stacklevel=2
                )
                i -= 1  # reduce i so that output tol_history indexing is correct
                break

            residual_l1_norm = abs(neg_residual.sum())
            calc_difference = residual_l1_norm / y_l1_norm
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            # only use negative residual in exp to avoid exponential overflow warnings
            # and accidently creating a weight of nan (inf * 0 = nan)
            weight_array[neg_mask] = np.exp(i * neg_residual / residual_l1_norm)
            weight_array[~neg_mask] = 0

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_arpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                      max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the arPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 25.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
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

        See Also
        --------
        Baseline2D.arpls

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve(y, weight_array)
            new_weights = _weighting._arpls(y, baseline)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_iarpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                       max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the IarPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 25.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 with N points.

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

        See Also
        --------
        Baseline2D.iarpls

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.pspline.solve(y, weight_array)
            new_weights = _weighting._iarpls(y, baseline, i)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if not np.isfinite(calc_difference):
                # catches nan, inf and -inf due to exp(i) being too high or if there
                # are too few negative residuals; no way to catch both conditions before
                # new_weights calculation since it is hard to estimate if
                # (exp(i) / std) * residual will overflow; check calc_difference rather
                # than checking new_weights since non-finite values rarely occur and
                # checking a scalar is faster; cannot use np.errstate since it is not 100% reliable
                warnings.warn(
                    ('nan and/or +/- inf occurred in weighting calculation, likely meaning '
                     '"tol" is too low and/or "max_iter" is too high'), ParameterWarning,
                    stacklevel=2
                )
                break
            elif calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_psalsa(self, data, lam=1e3, p=0.5, k=None, num_knots=25, spline_degree=3,
                       diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the psalsa algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~Baseline2D.asls`.
        num_knots : int, optional
            The number of knots for the spline. Default is 25.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
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
            Raised if `p` is not between 0 and 1.

        See Also
        --------
        Baseline2D.psalsa

        References
        ----------
        Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
        for analytical instruments. 2014 IEEE 11th International Multi-Conference on
        Systems, Signals, and Devices, 2014, 1-5.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        if k is None:
            k = np.std(y) / 10
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve(y, weight_array)
            new_weights = _weighting._psalsa(y, baseline, p, k, self._len)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params


@jit(nopython=True, cache=True)
def _numba_mapped_histogram(data, num_bins, histogram):
    """
    Creates a normalized histogram of the data and a mapping of the indices, using one pass.

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

    Notes
    -----
    If numba is installed, the histogram and bin mapping can both be created in
    one pass, which is faster.

    """
    if _HAS_NUMBA:
        # create zeros array outside of numba function since numba's implementation
        # of np.zeros is much slower than numpy's (https://github.com/numba/numba/issues/7259)
        histogram = np.zeros(num_bins)
        bins, bin_mapping = _numba_mapped_histogram(data.ravel(), num_bins, histogram)
    else:
        histogram, bins = np.histogram(data, num_bins, density=True)
        # leave out last bin edge to account for extra index; leave out first
        # bin edge since np.searchsorted finds indices where bin[i-1] <= val < bin[i]
        # while the desired indices are bin[i] <= val < bin[i + 1]
        bin_mapping = np.searchsorted(bins[1:-1], data, 'right')

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
        Log10 of the standard deviation of the Gaussian distribution.
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

    Notes
    -----
    Defining `sigma` as ``log10(actual sigma)`` allows not bounding `sigma` during
    optimization and allows it to more easily fit different scales.

    References
    ----------
    de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
    Intelligent Laboratory Systems, 2012, 117, 56-60.

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

    actual_sigma = 10**sigma
    # the gaussian should be area-normalized, so set height accordingly
    height = 1 / max(actual_sigma * np.sqrt(2 * np.pi), _MIN_FLOAT)

    return n1 * gaussian(x, height, 0, actual_sigma) + n2 * pos_uniform + n3 * neg_uniform
