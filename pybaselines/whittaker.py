# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on Sept. 13, 2019
@author: Donald Erb

"""

import numpy as np

from . import _weighting
from ._algorithm_setup import _Algorithm, _class_wrapper
from ._banded_utils import _shift_rows, diff_penalty_diagonals
from ._validation import _check_lam, _check_optional_array, _check_scalar_variable
from .utils import _mollifier_kernel, pad_edges, padded_convolve, relative_difference


class _Whittaker(_Algorithm):
    """A base class for all Whittaker-smoothing-based algorithms."""

    @_Algorithm._register(sort_keys=('weights',))
    def asls(self, data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        Fits the baseline using asymmetric least squares (AsLS) fitting.

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
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
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

        References
        ----------
        Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

        Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
        Leiden University Medical Centre Report, 2005, [unpublished].

        Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def iasls(self, data, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3,
              weights=None, diff_order=2):
        """
        Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm.

        The algorithm consideres both the first and second derivatives of the residual.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with `N` data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        lam_1 : float, optional
            The smoothing parameter for the first derivative of the residual. Default is 1e-4.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be set by fitting the data with a second order polynomial.
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

        References
        ----------
        He, S., et al. Baseline correction for raman spectra using an improved
        asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        elif diff_order < 2:
            raise ValueError('diff_order must be 2 or greater')

        if weights is None:
            _, _, pseudo_inverse = self._setup_polynomial(
                data, weights=None, poly_order=2, calc_vander=True, calc_pinv=True
            )
            baseline = self._polynomial.vandermonde @ (pseudo_inverse @ data)
            weights = _weighting._asls(data, baseline, p)

        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        lambda_1 = _check_lam(lam_1)
        diff_1_diags = diff_penalty_diagonals(self._size, 1, whittaker_system.lower, 1)
        if whittaker_system.using_pentapy:
            diff_1_diags = diff_1_diags[::-1]
        whittaker_system.add_penalty(lambda_1 * diff_1_diags)

        # fast calculation of lam_1 * (D_1.T @ D_1) @ y
        d1_y = y.copy()
        d1_y[0] = y[0] - y[1]
        d1_y[-1] = y[-1] - y[-2]
        d1_y[1:-1] = 2 * y[1:-1] - y[:-2] - y[2:]
        d1_y = lambda_1 * d1_y
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            weight_squared = weight_array**2
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_squared), weight_squared * y + d1_y,
                overwrite_b=True
            )
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def airpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None,
               normalize_weights=True):
        """
        Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
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
        normalize_weights : bool, optional
            If True (default), will normalize the computed weights between 0 and 1 to improve
            the numerical stabilty. Set to False to use the original implementation, which
            sets weights for all negative residuals to be greater than 1.

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

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True, check_output=True
            )
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

    @_Algorithm._register(sort_keys=('weights',))
    def arpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        Asymmetrically reweighted penalized least squares smoothing (arPLS).

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
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

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
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

    @_Algorithm._register(sort_keys=('weights',))
    def drpls(self, data, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None, diff_order=2):
        """
        Doubly reweighted penalized least squares (drPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
        eta : float
            A term for controlling the value of lam; should be between 0 and 1.
            Low values will produce smoother baselines, while higher values will
            more aggressively fit peaks. Default is 0.5.
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
            Raised if `eta` is not between 0 and 1 or if `diff_order` is less than 2.

        References
        ----------
        Xu, D. et al. Baseline correction method based on doubly reweighted
        penalized least squares, Applied Optics, 2019, 58, 3913-3920.

        """
        if not 0 <= eta <= 1:
            raise ValueError('eta must be between 0 and 1')
        elif diff_order < 2:
            raise ValueError('diff_order must be 2 or greater')

        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, allow_lower=False, reverse_diags=False
        )
        # W + P_1 + (I - eta * W) @ P_n -> P_1 + P_n + W @ (I - eta * P_n)
        diff_n_diagonals = -eta * whittaker_system.penalty[::-1]
        diff_n_diagonals[whittaker_system.main_diagonal_index] += 1

        diff_1_diagonals = diff_penalty_diagonals(self._size, 1, False, padding=diff_order - 1)
        whittaker_system.add_penalty(diff_1_diagonals)
        if whittaker_system.using_pentapy:
            whittaker_system.reverse_penalty()

        tol_history = np.empty(max_iter + 1)
        lower_upper_bands = (diff_order, diff_order)
        for i in range(1, max_iter + 2):
            penalty_with_weights = diff_n_diagonals * weight_array
            if not whittaker_system.using_pentapy:
                penalty_with_weights = _shift_rows(penalty_with_weights, diff_order, diff_order)
            baseline = whittaker_system.solve(
                whittaker_system.penalty + penalty_with_weights, weight_array * y,
                overwrite_ab=True, overwrite_b=True, l_and_u=lower_upper_bands
            )
            new_weights, exit_early = _weighting._drpls(y, baseline, i)
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

    @_Algorithm._register(sort_keys=('weights',))
    def iarpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
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

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
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

    @_Algorithm._register(sort_keys=('weights', 'alpha'))
    def aspls(self, data, lam=1e5, diff_order=2, max_iter=100, tol=1e-3,
              weights=None, alpha=None, asymmetric_coef=0.5):
        """
        Adaptive smoothness penalized least squares smoothing (asPLS).

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 100.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.
        alpha : array-like, shape (N,), optional
            An array of values that control the local value of `lam` to better
            fit peak and non-peak regions. If None (default), then the initial values
            will be an array with size equal to N and all values set to 1.
        asymmetric_coef : float
            The asymmetric coefficient for the weighting. Higher values leads to a steeper
            weighting curve (ie. more step-like). Default is 0.5.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'alpha': numpy.ndarray, shape (N,)
                The array of alpha values used for fitting the data in the final iteration.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        Raises
        ------
        ValueError
            Raised if `alpha` and `data` do not have the same shape. Also raised if
            `asymmetric_coef` is not greater than 0.

        Notes
        -----
        The default asymmetric coefficient (`k` in the asPLS paper) is 0.5 instead
        of the 2 listed in the asPLS paper. pybaselines uses the factor of 0.5 since it
        matches the results in Table 2 and Figure 5 of the asPLS paper closer than the
        factor of 2 and fits noisy data much better.

        References
        ----------
        Zhang, F., et al. Baseline correction for infrared spectra using
        adaptive smoothness parameter penalized least squares method.
        Spectroscopy Letters, 2020, 53(3), 222-233.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, allow_lower=False, reverse_diags=True
        )
        alpha_array = _check_optional_array(
            self._size, alpha, check_finite=self._check_finite, name='alpha'
        )
        if self._sort_order is not None and alpha is not None:
            alpha_array = alpha_array[self._sort_order]
        asymmetric_coef = _check_scalar_variable(asymmetric_coef, variable_name='asymmetric_coef')

        main_diag_idx = whittaker_system.main_diagonal_index
        lower_upper_bands = (diff_order, diff_order)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            lhs = whittaker_system.penalty * alpha_array
            lhs[main_diag_idx] = lhs[main_diag_idx] + weight_array
            if not whittaker_system.using_pentapy:
                lhs = _shift_rows(lhs, diff_order, diff_order)
            baseline = whittaker_system.solve(
                lhs, weight_array * y, overwrite_ab=True, overwrite_b=True,
                l_and_u=lower_upper_bands
            )
            new_weights, residual, exit_early = _weighting._aspls(y, baseline, asymmetric_coef)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights
            abs_d = np.abs(residual)
            alpha_array = abs_d / abs_d.max()

        params = {
            'weights': weight_array, 'alpha': alpha_array, 'tol_history': tol_history[:i + 1]
        }

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def psalsa(self, data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3,
               weights=None):
        """
        Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

        Similar to the asymmetric least squares (AsLS) algorithm, but applies an
        exponential decay weighting to values greater than the baseline to allow
        using a higher `p` value to better fit noisy data.

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
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~.Baseline.asls`.
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
            Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
            than 0.

        Notes
        -----
        The exit criteria for the original algorithm was to check whether the signs
        of the residuals do not change between two iterations, but the comparison of
        the l2 norms of the weight arrays between iterations is used instead to be
        more comparable to other Whittaker-smoothing-based algorithms.

        References
        ----------
        Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
        for analytical instruments. 2014 IEEE 11th International Multi-Conference on
        Systems, Signals, and Devices, 2014, 1-5.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        if k is None:
            k = np.std(y) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
            new_weights = _weighting._psalsa(y, baseline, p, k, self._shape)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def derpsalsa(self, data, lam=1e6, p=0.01, k=None, diff_order=2, max_iter=50, tol=1e-3,
                  weights=None, smooth_half_window=None, num_smooths=16, pad_kwargs=None,
                  **kwargs):
        """
        Derivative Peak-Screening Asymmetric Least Squares Algorithm (derpsalsa).

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
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~.Baseline.asls`.
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
        smooth_half_window : int, optional
            The half-window to use for smoothing the data before computing the first
            and second derivatives. Default is None, which will use ``len(data) / 200``.
        num_smooths : int, optional
            The number of times to smooth the data before computing the first
            and second derivatives. Default is 16.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `pad_kwargs`.

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
            Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
            than 0.

        References
        ----------
        Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
        automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
        51(10), 2061-2065.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        if k is None:
            k = np.std(y) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')
        if smooth_half_window is None:
            smooth_half_window = self._size // 200
        # could pad the data every iteration, but it is ~2-3 times slower and only affects
        # the edges, so it's not worth it
        self._deprecate_pad_kwargs(**kwargs)
        pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
        y_smooth = pad_edges(y, smooth_half_window, **pad_kwargs, **kwargs)
        if smooth_half_window > 0:
            smooth_kernel = _mollifier_kernel(smooth_half_window)
            for _ in range(num_smooths):
                y_smooth = padded_convolve(y_smooth, smooth_kernel)
        y_smooth = y_smooth[smooth_half_window:self._size + smooth_half_window]

        diff_y_1 = np.gradient(y_smooth)
        diff_y_2 = np.gradient(diff_y_1)
        # x.dot(x) is same as (x**2).sum() but faster
        rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / self._size)
        rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / self._size)

        diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
        diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
        partial_weights = diff_1_weights * diff_2_weights

        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
            new_weights = _weighting._derpsalsa(y, baseline, p, k, self._shape, partial_weights)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def brpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, max_iter_2=50,
              tol_2=1e-3, weights=None):
        """
        Bayesian Reweighted Penalized Least Squares (BrPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
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
            * 'tol_history': numpy.ndarray, shape (J, K)
                An array containing the calculated tolerance values for each iteration of
                both threshold values and fit values. Index 0 are the tolerence values for
                the difference in the peak proportion, and indices >= 1 are the tolerance values
                for each fit. All values that were not used in fitting have values of 0. Shape J
                is 2 plus the number of iterations for the threshold to converge (related to
                `max_iter_2`, `tol_2`), and shape K is the maximum of the number of
                iterations for the threshold and the maximum number of iterations for all of
                the fits of the various threshold values (related to `max_iter` and `tol`).

        References
        ----------
        Wang, Q., et al. Spectral baseline estimation using penalized least squares
        with weights derived from the Bayesian method. Nuclear Science and Techniques,
        2022, 140, 250-257.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
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
                new_baseline = whittaker_system.solve(
                    whittaker_system.add_diagonal(weight_array), weight_array * y,
                    overwrite_b=True
                )
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

    @_Algorithm._register(sort_keys=('weights',))
    def lsrpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        Locally Symmetric Reweighted Penalized Least Squares (LSRPLS).

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e5.
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

        References
        ----------
        Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
        Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(
                whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
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


_whittaker_wrapper = _class_wrapper(_Whittaker)


@_whittaker_wrapper
def asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None, x_data=None):
    """
    Fits the baseline using asymmetric least squares (AsLS) fitting.

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
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight. Default is 1e-2.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

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

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, [unpublished].

    Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

    """


@_whittaker_wrapper
def iasls(data, x_data=None, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3,
          weights=None, diff_order=2):
    """
    Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm.

    The algorithm consideres both the first and second derivatives of the residual.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with `N` data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight. Default is 1e-2.
    lam_1 : float, optional
        The smoothing parameter for the first derivative of the residual. Default is 1e-4.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be set by fitting the data with a second order polynomial.
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

    References
    ----------
    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    """


@_whittaker_wrapper
def airpls(data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None, x_data=None,
           normalize_weights=True):
    """
    Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    normalize_weights : bool, optional
        If True (default), will normalize the computed weights between 0 and 1 to improve
        the numerical stabilty. Set to False to use the original implementation, which
        sets weights for all negative residuals to be greater than 1.

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

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    """


@_whittaker_wrapper
def arpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None, x_data=None):
    """
    Asymmetrically reweighted penalized least squares smoothing (arPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

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

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """


@_whittaker_wrapper
def drpls(data, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None, diff_order=2, x_data=None):
    """
    Doubly reweighted penalized least squares (drPLS) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    eta : float
        A term for controlling the value of lam; should be between 0 and 1.
        Low values will produce smoother baselines, while higher values will
        more aggressively fit peaks. Default is 0.5.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

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
        Raised if `eta` is not between 0 and 1 or if `diff_order` is less than 2.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """


@_whittaker_wrapper
def iarpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None, x_data=None):
    """
    Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

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

    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    """


@_whittaker_wrapper
def aspls(data, lam=1e5, diff_order=2, max_iter=100, tol=1e-3, weights=None,
          alpha=None, x_data=None, asymmetric_coef=0.5):
    """
    Adaptive smoothness penalized least squares smoothing (asPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    max_iter : int, optional
        The max number of fit iterations. Default is 100.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.
    alpha : array-like, shape (N,), optional
        An array of values that control the local value of `lam` to better
        fit peak and non-peak regions. If None (default), then the initial values
        will be an array with size equal to N and all values set to 1.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    asymmetric_coef : float
        The asymmetric coefficient for the weighting. Higher values leads to a steeper
        weighting curve (ie. more step-like). Default is 0.5.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'alpha': numpy.ndarray, shape (N,)
            The array of alpha values used for fitting the data in the final iteration.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Raises
    ------
    ValueError
        Raised if `alpha` and `data` do not have the same shape. Also raised if `asymmetric_coef`
        is not greater than 0.

    Notes
    -----
    The default asymmetric coefficient (`k` in the asPLS paper) is 0.5 instead
    of the 2 listed in the asPLS paper. pybaselines uses the factor of 0.5 since it
    matches the results in Table 2 and Figure 5 of the asPLS paper closer than the
    factor of 2 and fits noisy data much better.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using
    adaptive smoothness parameter penalized least squares method.
    Spectroscopy Letters, 2020, 53(3), 222-233.

    """


@_whittaker_wrapper
def psalsa(data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3,
           weights=None, x_data=None):
    """
    Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

    Similar to the asymmetric least squares (AsLS) algorithm, but applies an
    exponential decay weighting to values greater than the baseline to allow
    using a higher `p` value to better fit noisy data.

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
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight. Default is 0.5.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :meth:`~.Baseline.asls`.
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
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

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
        Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
        than 0.

    Notes
    -----
    The exit criteria for the original algorithm was to check whether the signs
    of the residuals do not change between two iterations, but the comparison of
    the l2 norms of the weight arrays between iterations is used instead to be
    more comparable to other Whittaker-smoothing-based algorithms.

    References
    ----------
    Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
    for analytical instruments. 2014 IEEE 11th International Multi-Conference on
    Systems, Signals, and Devices, 2014, 1-5.

    """


@_whittaker_wrapper
def derpsalsa(data, lam=1e6, p=0.01, k=None, diff_order=2, max_iter=50, tol=1e-3, weights=None,
              smooth_half_window=None, num_smooths=16, x_data=None, pad_kwargs=None, **kwargs):
    """
    Derivative Peak-Screening Asymmetric Least Squares Algorithm (derpsalsa).

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
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight. Default is 1e-2.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :meth:`~.Baseline.asls`.
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
    smooth_half_window : int, optional
        The half-window to use for smoothing the data before computing the first
        and second derivatives. Default is None, which will use ``len(data) / 200``.
    num_smooths : int, optional
        The number of times to smooth the data before computing the first
        and second derivatives. Default is 16.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from smoothing. Default is None.
    **kwargs

        .. deprecated:: 1.2.0
            Passing additional keyword arguments is deprecated and will be removed in version
            1.4.0. Pass keyword arguments using `pad_kwargs`.

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
        Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
        than 0.

    References
    ----------
    Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
    automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
    51(10), 2061-2065.

    """


@_whittaker_wrapper
def brpls(data, x_data=None, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, max_iter_2=50,
          tol_2=1e-3, weights=None):
    """
    Bayesian Reweighted Penalized Least Squares (BrPLS) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
        * 'tol_history': numpy.ndarray, shape (J, K)
            An array containing the calculated tolerance values for each iteration of
            both threshold values and fit values. Index 0 are the tolerence values for
            the difference in the peak proportion, and indices >= 1 are the tolerance values
            for each fit. All values that were not used in fitting have values of 0. Shape J
            is 2 plus the number of iterations for the threshold to converge (related to
            `max_iter_2`, `tol_2`), and shape K is the maximum of the number of
            iterations for the threshold and the maximum number of iterations for all of
            the fits of the various threshold values (related to `max_iter` and `tol`).

    References
    ----------
    Wang, Q., et al. Spectral baseline estimation using penalized least squares
    with weights derived from the Bayesian method. Nuclear Science and Techniques,
    2022, 140, 250-257.

    """


@_whittaker_wrapper
def lsrpls(data, x_data=None, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Locally Symmetric Reweighted Penalized Least Squares (LSRPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
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

    References
    ----------
    Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
    Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.

    """
