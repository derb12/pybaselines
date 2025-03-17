# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on April 25, 2023
@author: Donald Erb

"""

import warnings

import numpy as np

from .. import _weighting
from .._validation import _check_scalar_variable
from ..utils import ParameterWarning, gaussian, relative_difference, _MIN_FLOAT
from ._algorithm_setup import _Algorithm2D
from ._whittaker_utils import PenalizedSystem2D


class _Spline(_Algorithm2D):
    """A base class for all spline algorithms."""

    @_Algorithm2D._register(sort_keys=('weights',))
    def mixture_model(self, data, lam=1e3, p=1e-2, num_knots=25, spline_degree=3, diff_order=3,
                      max_iter=50, tol=1e-3, weights=None, symmetric=False):
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
            will be given `1 - p` weight. Used to set the initial weights before performing
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

        Ghojogh, B., et al. Fitting A Mixture Distribution to Data: Tutorial. arXiv
        preprint arXiv:1901.06708, 2019.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        # scale y between -1 and 1 so that the residual fit is more numerically stable
        # TODO is this still necessary now that expectation-maximization is used? -> still
        # helps to prevent overflows when using gaussian
        y_domain = np.polynomial.polyutils.getdomain(y.ravel())
        y = np.polynomial.polyutils.mapdomain(y, y_domain, np.array([-1., 1.]))

        if weights is not None:
            baseline = pspline.solve(y, weight_array)
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
                baseline = pspline.solve(y, weight_array)
                weight_array = _weighting._asls(y, baseline, p)

        residual = y - baseline
        # the 0.2 * std(residual) is an "okay" starting sigma estimate
        sigma = 0.2 * np.std(residual)
        fraction_noise = 0.5
        if symmetric:
            fraction_positive = 0.25
        else:
            fraction_positive = 1 - fraction_noise
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            # expectation part of expectation-maximization -> calc pdfs and
            # posterior probabilities
            positive_pdf = np.where(
                residual >= 0, fraction_positive / max(abs(residual.max()), 1e-6), 0
            )
            noise_pdf = (
                fraction_noise * gaussian(residual, 1 / (sigma * np.sqrt(2 * np.pi)), 0, sigma)
            )
            total_pdf = noise_pdf + positive_pdf
            if symmetric:
                negative_pdf = np.where(
                    residual < 0,
                    (1 - fraction_noise - fraction_positive) / max(abs(residual.min()), 1e-6),
                    0
                )
                total_pdf += negative_pdf
            posterior_prob_noise = noise_pdf / np.maximum(total_pdf, _MIN_FLOAT)

            calc_difference = relative_difference(weight_array, posterior_prob_noise)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

            # maximization part of expectation-maximization -> update sigma and
            # fractions of each pdf
            noise_sum = posterior_prob_noise.sum()
            sigma = np.sqrt((posterior_prob_noise * residual**2).sum() / noise_sum)
            if not symmetric:
                fraction_noise = posterior_prob_noise.mean()
                fraction_positive = 1 - fraction_noise
            else:
                posterior_prob_positive = positive_pdf / total_pdf
                posterior_prob_negative = negative_pdf / total_pdf

                positive_sum = posterior_prob_positive.sum()
                negative_sum = posterior_prob_negative.sum()
                total_sum = noise_sum + positive_sum + negative_sum

                fraction_noise = noise_sum / total_sum
                fraction_positive = positive_sum / total_sum

            weight_array = posterior_prob_noise
            baseline = pspline.solve(y, weight_array)
            residual = y - baseline

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1]
        }

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

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        old_coef = np.zeros(np.prod(self._spline_basis._num_bases))
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array)
            calc_difference = relative_difference(old_coef, pspline.coef)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            old_coef = pspline.coef
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
            will be given `1 - p` weight. Default is 1e-2.
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

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array)
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
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        lam_1 : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively, of the first
            derivative of the residual. If a single value is given, both will use the same
            value. Default is 1e-4.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines along the rows and columns, respectively. If a
            single value is given, both will use the same value. Default is 25.
        spline_degree : int or Sequence[int, int], optional
            The degree of the splines along the rows and columns, respectively. If a single
            value is given, both will use the same value. Default is 3, which is a cubic spline.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 1 or 2.

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
            baseline = self._polynomial.vandermonde @ (pseudo_inverse @ data.ravel())
            weights = _weighting._asls(data, baseline.reshape(self._shape), p)

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        # B.T @ P_1 @ B and B.T @ P_1 @ y
        penalized_system_1 = PenalizedSystem2D(self._shape, lam_1, diff_order=1)
        p1_partial_penalty = pspline.basis.basis.T @ penalized_system_1.penalty

        partial_rhs = p1_partial_penalty @ y.ravel()
        pspline.add_penalty(p1_partial_penalty @ pspline.basis.basis)

        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array**2, rhs_extra=partial_rhs)
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
                       diff_order=2, max_iter=50, tol=1e-3, weights=None, normalize_weights=True):
        """
        A penalized spline version of the airPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
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
        normalize_weights : bool, optional
            If True (default), will normalize the computed weights between 0 and 1 to improve
            the numerical stabilty. Set to False to use the original implementation, which
            sets weights for all negative residuals to be greater than 1.

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
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = pspline.solve(y, weight_array)
            new_weights, residual_l1_norm, exit_early = _weighting._airpls(
                y, baseline, i, normalize_weights
            )
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = residual_l1_norm / y_l1_norm
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_arpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                      max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the arPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
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
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array)
            new_weights, exit_early = _weighting._arpls(y, baseline)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
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
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
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
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = pspline.solve(y, weight_array)
            new_weights, exit_early = _weighting._iarpls(y, baseline, i)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
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
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~.Baseline2D.asls`.
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
            Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
            than 0.

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

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        if k is None:
            k = np.std(y) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array)
            new_weights = _weighting._psalsa(y, baseline, p, k, self._shape)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_brpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                      max_iter=50, tol=1e-3, max_iter_2=50, tol_2=1e-3, weights=None):
        """
        A penalized spline version of the brPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
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
        max_iter_2 : float, optional
            The number of iterations for updating the proportion of data occupied by peaks.
            Default is 50.
        tol_2 : float, optional
            The exit criteria for the difference between the calculated proportion of data
            occupied by peaks. Default is 1e-3.
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
            * 'tol_history': numpy.ndarray, shape (J, K)
                An array containing the calculated tolerance values for each iteration of
                both threshold values and fit values. Index 0 are the tolerence values for
                the difference in the peak proportion, and indices >= 1 are the tolerance values
                for each fit. All values that were not used in fitting have values of 0. Shape J
                is 2 plus the number of iterations for the threshold to converge (related to
                `max_iter_2`, `tol_2`), and shape K is the maximum of the number of
                iterations for the threshold and the maximum number of iterations for all of
                the fits of the various threshold values (related to `max_iter` and `tol`).

        See Also
        --------
        Baseline2D.brpls

        References
        ----------
        Wang, Q., et al. Spectral baseline estimation using penalized least squares
        with weights derived from the Bayesian method. Nuclear Science and Techniques,
        2022, 140, 250-257.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        beta = 0.5
        j_max = 0
        baseline = y
        baseline_weights = weight_array
        tol_history = np.zeros((max_iter_2 + 2, max(max_iter, max_iter_2) + 1))
        # implementation note: weight_array must always be updated since otherwise when
        # reentering the inner loop, new_baseline and baseline would be the same; instead,
        # use baseline_weights to track which weights produced the output baseline
        for i in range(max_iter_2 + 1):
            for j in range(max_iter + 1):
                new_baseline = pspline.solve(y, weight_array)
                new_weights, exit_early = _weighting._brpls(y, new_baseline, beta)
                if exit_early:
                    j -= 1  # reduce j so that output tol_history indexing is correct
                    tol_2 = np.inf  # ensure it exits outer loop
                    break
                # Paper used norm(old - new) / norm(new) rather than old in the denominator,
                # but I use old in the denominator instead to be consistant with all other
                # algorithms; does not make a major difference
                calc_difference = relative_difference(baseline, new_baseline)
                tol_history[i + 1, j] = calc_difference
                if calc_difference < tol:
                    if i == 0 and j == 0:  # for cases where tol == inf
                        baseline = new_baseline
                    break
                baseline_weights = weight_array
                weight_array = new_weights
                baseline = new_baseline
            j_max = max(j, j_max)

            weight_array = new_weights
            weight_mean = weight_array.mean()
            calc_difference_2 = abs(beta + weight_mean - 1)
            tol_history[0, i] = calc_difference_2
            if calc_difference_2 < tol_2:
                break
            beta = 1 - weight_mean

        params = {
            'weights': baseline_weights, 'tol_history': tol_history[:i + 2, :max(i, j_max) + 1]
        }

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def pspline_lsrpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                       max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the LSRPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
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

        See Also
        --------
        Baseline2D.lsrpls

        References
        ----------
        Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
        Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = pspline.solve(y, weight_array)
            new_weights, exit_early = _weighting._lsrpls(y, baseline, i)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params
