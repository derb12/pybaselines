# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on April 30, 2023
@author: Donald Erb

"""

import numpy as np

from .. import _weighting
from .._compat import diags
from .._validation import _check_optional_array, _check_scalar_variable
from ..utils import _MIN_FLOAT, relative_difference
from ._algorithm_setup import _Algorithm2D
from ._whittaker_utils import PenalizedSystem2D


class _Whittaker(_Algorithm2D):
    """A base class for all Whittaker-smoothing-based algorithms."""

    @_Algorithm2D._register(sort_keys=('weights',))
    def asls(self, data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None,
             num_eigens=(10, 10), return_dof=False):
        """
        Fits the baseline using asymmetric least squares (AsLS) fitting.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

        References
        ----------
        Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

        Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
        Leiden University Medical Centre Report, 2005, 1(1).

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(y, weight_array)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
    def iasls(self, data, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3,
              weights=None, diff_order=2):
        """
        Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm.

        The algorithm consideres both the first and second derivatives of the residual.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        lam_1 : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively, of the first
            derivative of the residual. Default is 1e-4.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be set by fitting the data with a second order polynomial.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 1.
            Default is 2 (second order differential matrix). Typical values are 2 or 3.

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

        References
        ----------
        He, S., et al. Baseline correction for raman spectra using an improved
        asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

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

        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._shape, lam_1, diff_order=1)

        # (W.T @ W + P_1) @ y -> P_1 @ y + W.T @ W @ y
        whittaker_system.add_penalty(penalized_system_1.penalty)
        p1_y = penalized_system_1.penalty @ y
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(y, weight_array**2, rhs_extra=p1_y)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def airpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None,
               num_eigens=(10, 10), return_dof=False, normalize_weights=False):
        """
        Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e6.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.
        normalize_weights : bool, optional
            If True, will normalize the computed weights between 0 and 1 to potentially
            improve the numerical stabilty. Set to False (default) to use the original
            implementation, which sets weights for all negative residuals to be greater than 1.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens

        )
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(y, weight_array)
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

        params = {'tol_history': tol_history[:i]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def arpls(self, data, lam=1e3, diff_order=2, max_iter=50, tol=1e-3, weights=None,
              num_eigens=(10, 10), return_dof=False):
        """
        Asymmetrically reweighted penalized least squares smoothing (arPLS).

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._arpls(y, baseline)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
    def drpls(self, data, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None, diff_order=2):
        """
        Doubly reweighted penalized least squares (drPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e5.
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
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 1.
            Default is 2 (second order differential matrix). Typical values are 2 or 3.

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
            Raised if `eta` is not between 0 and 1 or if `diff_order` is less than 2.

        References
        ----------
        Xu, D. et al. Baseline correction method based on doubly reweighted
        penalized least squares, Applied Optics, 2019, 58, 3913-3920.

        """
        if not 0 <= eta <= 1:
            raise ValueError('eta must be between 0 and 1')
        elif np.less(diff_order, 2).any():
            raise ValueError('diff_order must be 2 or greater')

        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._shape, 1, diff_order=1)
        # W + P_1 + (I - eta * W) @ P_n -> P_1 + P_n + W @ (I - eta * P_n)
        partial_penalty = whittaker_system.penalty + penalized_system_1.penalty
        partial_penalty_2 = -eta * whittaker_system.penalty
        partial_penalty_2.setdiag(partial_penalty_2.diagonal() + 1)
        weight_matrix = diags(weight_array, format='csr')
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.direct_solve(
                partial_penalty + weight_matrix @ partial_penalty_2, weight_array * y
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
            weight_matrix.setdiag(weight_array)

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def iarpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None,
               num_eigens=(10, 10), return_dof=False):
        """
        Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e5.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._iarpls(y, baseline, i)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights', 'alpha'), reshape_keys=('weights', 'alpha'), reshape_baseline=True
    )
    def aspls(self, data, lam=1e5, diff_order=2, max_iter=100, tol=1e-3,
              weights=None, alpha=None, asymmetric_coef=2., alternate_weighting=True):
        """
        Adaptive smoothness penalized least squares smoothing (asPLS).

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e5.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        alpha : array-like, shape (M, N), optional
            An array of values that control the local value of `lam` to better
            fit peak and non-peak regions. If None (default), then the initial values
            will be an array with shape equal to (M, N) and all values set to 1.
        asymmetric_coef : float, optional
            The asymmetric coefficient for the weighting. Higher values leads to a steeper
            weighting curve (ie. more step-like). Default is 2, as used in the asPLS paper [1]_.
            Note that a value of 4 results in the weighting scheme used in the NasPLS
            (Non-sensitive-areas adaptive smoothness penalized least squares smoothing) algorithm
            [2]_.
        alternate_weighting : bool, optional
            If True (default), subtracts the mean of the negative residuals within the weighting
            equation. If False, uses the weighting equation as stated within the asPLS paper [1]_.
            See the Notes section below for more details.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'alpha': numpy.ndarray, shape (M, N)
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
        The weighting scheme as written in the asPLS paper [1]_ does not reproduce the paper's
        results for noisy data. By subtracting the mean of negative residuals (``data-baseline``)
        within the weighting scheme, as used by other algorithms such as ``arPLS`` and ``drPLS``,
        the asPLS paper's results can be correctly replicated (see
        https://github.com/derb12/pybaselines/issues/40 for more details). Given this discrepancy,
        the default for ``aspls`` is to also subtract the negative residuals within the weighting.
        To use the weighting scheme as it is written in the asPLS paper, simply set
        ``alternate_weighting`` to False.

        References
        ----------
        .. [1] Zhang, F., et al. Baseline correction for infrared spectra using
            adaptive smoothness parameter penalized least squares method.
            Spectroscopy Letters, 2020, 53(3), 222-233.
        .. [2] Zhang, F., et al. An automatic baseline correction method based on reweighted
            penalized least squares method for non-sensitive areas. Vibrational Spectroscopy,
            2025, 138, 103806.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(data, lam, diff_order, weights)
        alpha_array = _check_optional_array(
            self._shape, alpha, check_finite=self._check_finite, name='alpha',
            ensure_1d=False, axis=slice(None)
        )
        if self._sort_order is not None and alpha is not None:
            alpha_array = alpha_array[self._sort_order]
        asymmetric_coef = _check_scalar_variable(asymmetric_coef, variable_name='asymmetric_coef')

        # use a sparse matrix to maintain sparsity after multiplication; implementation note:
        # could skip making an alpha matrix and just use alpha_array[:, None] * penalty once
        # the scipy sparse_arrays become standard -> will have to check if timing is affected
        alpha_matrix = diags(alpha_array.ravel(), format='csr')
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            penalty = alpha_matrix @ whittaker_system.penalty
            baseline = whittaker_system.solve(y, weight_array, penalty=penalty)
            new_weights, residual, exit_early = _weighting._aspls(
                y, baseline, asymmetric_coef, alternate_weighting
            )
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights
            # add _MIN_FLOAT so that no values are 0; otherwise, the sparsity of alpha @ penalty
            # can change, which is inefficient
            abs_d = np.abs(residual) + _MIN_FLOAT
            alpha_array = abs_d / abs_d.max()
            alpha_matrix.setdiag(alpha_array)

        params = {
            'weights': weight_array, 'alpha': alpha_array, 'tol_history': tol_history[:i + 1]
        }

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def psalsa(self, data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3,
               weights=None, num_eigens=(10, 10), return_dof=False):
        """
        Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

        Similar to the asymmetric least squares (AsLS) algorithm, but applies an
        exponential decay weighting to values greater than the baseline to allow
        using a higher `p` value to better fit noisy data.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e5.
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
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

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

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        if k is None:
            k = np.std(y) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')

        shape = self._shape if whittaker_system._using_svd else self._size
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = whittaker_system.solve(y, weight_array)
            new_weights = _weighting._psalsa(y, baseline, p, k, shape)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def brpls(self, data, lam=1e3, diff_order=2, max_iter=50, tol=1e-3, max_iter_2=50,
              tol_2=1e-3, weights=None, num_eigens=(10, 10), return_dof=False):
        """
        Bayesian Reweighted Penalized Least Squares (BrPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
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
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Wang, Q., et al. Spectral baseline estimation using penalized least squares
        with weights derived from the Bayesian method. Nuclear Science and Techniques,
        2022, 140, 250-257.

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
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
                new_baseline = whittaker_system.solve(y, weight_array)
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

        params = {'tol_history': tol_history[:i + 2, :max(i, j_max) + 1]}
        if whittaker_system._using_svd:
            params['weights'] = baseline_weights
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(baseline_weights)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = baseline_weights.reshape(self._shape)

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def lsrpls(self, data, lam=1e3, diff_order=2, max_iter=50, tol=1e-3, weights=None,
              num_eigens=(10, 10), return_dof=False, alternate_weighting=False):
        """
        Locally Symmetric Reweighted Penalized Least Squares (LSRPLS).

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter for the rows and columns, respectively. If a single
            value is given, both will use the same value. Larger values will create smoother
            baselines. Default is 1e3.
        diff_order : int or Sequence[int, int], optional
            The order of the differential matrix for the rows and columns, respectively. If
            a single value is given, both will use the same value. Must be greater than 0.
            Default is 2 (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with shape equal to (M, N) and all values set to 1.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10).
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time.
        alternate_weighting : bool, optional
            If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
            the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
            a prefactor term of ``exp(t)``. See the Notes section below for more details.

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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        Notes
        -----
        In the LSRPLS paper [1]_, the weighting equation is written with a prefactor term
        of ``10^t``, where ``t`` is the iteration number, but the plotted weighting curve in
        Figure 1 of the paper shows a prefactor term of ``exp(t)`` instead. Since it is ambiguous
        which prefactor term is actually used for the algorithm, both are permitted by setting
        `alternate_weighting` to True to use ``10^t`` and False to use ``exp(t)``. In practice,
        the prefactor determines how quickly the weighting curve converts from a sigmoidal curve
        to a step curve, and does not heavily influence the result.

        If ``alternate_weighting`` is False, the weighting is the same as the drPLS algorithm [2]_.

        References
        ----------
        .. [1] Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
            Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.
        .. [2] Xu, D. et al. Baseline correction method based on doubly reweighted
            penalized least squares, Applied Optics, 2019, 58, 3913-3920.
        .. [3] Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
            (Preprint), 2023.

        """
        y, weight_array, whittaker_system = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = whittaker_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._lsrpls(y, baseline, i, alternate_weighting)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i]}
        if whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._shape)
            params['weights'] = weight_array.reshape(self._shape)

        return baseline, params
