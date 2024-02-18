# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on April 30, 2023
@author: Donald Erb

"""

import warnings

import numpy as np

from .. import _weighting
from .._compat import diags
from .._validation import _check_optional_array
from ..utils import _MIN_FLOAT, ParameterWarning, relative_difference
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
            will be given `p - 1` weight. Default is 1e-2.
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
            using the full analytical solution, which is typically much slower.
            Default is (10, 10).
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(y, weight_array)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if self.whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = self.whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._len)
            params['weights'] = weight_array.reshape(self._len)

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
            will be given `p - 1` weight. Default is 1e-2.
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
            baseline = self.vandermonde @ (pseudo_inverse @ data.ravel())
            weights = _weighting._asls(data, baseline.reshape(self._len), p)

        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._len, lam_1, diff_order=1)

        # (W.T @ W + P_1) @ y -> P_1 @ y + W.T @ W @ y
        self.whittaker_system.add_penalty(penalized_system_1.penalty)
        p1_y = penalized_system_1.penalty @ y
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(y, weight_array**2, rhs_extra=p1_y)
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
               num_eigens=(10, 10), return_dof=False):
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
            using the full analytical solution, which is typically much slower.
            Default is (10, 10).
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
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, copy_weights=True, num_eigens=num_eigens

        )
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        # Have to have extensive error handling since the weights can all become
        # very small due to the exp(i) term if too many iterations are performed;
        # checking the negative residual length usually prevents any errors, but
        # sometimes not so have to also catch any errors from the solvers
        for i in range(1, max_iter + 2):
            try:
                output = self.whittaker_system.solve(y, weight_array)
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
            if neg_residual.size < 2:
                # exit if there are < 2 negative residuals since all points or all but one
                # point would get a weight of 0, which fails the solver
                warnings.warn(
                    ('almost all baseline points are below the data, indicating that "tol"'
                     ' is too low and/or "max_iter" is too high'), ParameterWarning, stacklevel=2
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

        params = {'tol_history': tol_history[:i]}
        if self.whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = self.whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._len)
            params['weights'] = weight_array.reshape(self._len)

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
            using the full analytical solution, which is typically much slower.
            Default is (10, 10).
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(y, weight_array)
            new_weights = _weighting._arpls(y, baseline)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if self.whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = self.whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._len)
            params['weights'] = weight_array.reshape(self._len)

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

        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._len, 1, diff_order=1)
        # W + P_1 + (I - eta * W) @ P_n -> P_1 + P_n + W @ (I - eta * P_n)
        partial_penalty = self.whittaker_system.penalty + penalized_system_1.penalty
        partial_penalty_2 = -eta * self.whittaker_system.penalty
        partial_penalty_2.setdiag(partial_penalty_2.diagonal() + 1)
        weight_matrix = diags(weight_array, format='csr')
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.whittaker_system.direct_solve(
                partial_penalty + weight_matrix @ partial_penalty_2, weight_array * y
            )
            new_weights = _weighting._drpls(y, baseline, i)
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
            using the full analytical solution, which is typically much slower.
            Default is (10, 10).
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.whittaker_system.solve(y, weight_array)
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

        params = {'tol_history': tol_history[:i]}
        if self.whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = self.whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._len)
            params['weights'] = weight_array.reshape(self._len)

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights', 'alpha'), reshape_keys=('weights', 'alpha'), reshape_baseline=True
    )
    def aspls(self, data, lam=1e5, diff_order=2, max_iter=100, tol=1e-3,
              weights=None, alpha=None):
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

        Notes
        -----
        The weighting uses an asymmetric coefficient (`k` in the asPLS paper) of 0.5 instead
        of the 2 listed in the asPLS paper. pybaselines uses the factor of 0.5 since it
        matches the results in Table 2 and Figure 5 of the asPLS paper closer than the
        factor of 2 and fits noisy data much better.

        References
        ----------
        Zhang, F., et al. Baseline correction for infrared spectra using
        adaptive smoothness parameter penalized least squares method.
        Spectroscopy Letters, 2020, 53(3), 222-233.

        """
        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        alpha_array = _check_optional_array(
            self._len, alpha, check_finite=self._check_finite, name='alpha',
            ensure_1d=False, axis=slice(None)
        )
        if self._sort_order is not None and alpha is not None:
            alpha_array = alpha_array[self._sort_order]

        # use a sparse matrix to maintain sparsity after multiplication; implementation note:
        # could skip making an alpha matrix and just use alpha_array[:, None] * penalty once
        # the scipy sparse_arrays become standard -> will have to check if timing is affected
        alpha_matrix = diags(alpha_array.ravel(), format='csr')
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            penalty = alpha_matrix @ self.whittaker_system.penalty
            baseline = self.whittaker_system.solve(y, weight_array, penalty=penalty)
            new_weights, residual = _weighting._aspls(y, baseline)
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
            will be given `p - 1` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~Baseline2D.asls`.
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
            using the full analytical solution, which is typically much slower.
            Default is (10, 10).
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, num_eigens=num_eigens
        )
        if k is None:
            k = np.std(y) / 10

        shape = self._len if self.whittaker_system._using_svd else np.prod(self._len)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(y, weight_array)
            new_weights = _weighting._psalsa(y, baseline, p, k, shape)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'tol_history': tol_history[:i + 1]}
        if self.whittaker_system._using_svd:
            params['weights'] = weight_array
            if return_dof:
                params['dof'] = self.whittaker_system._calc_dof(weight_array)
        else:
            baseline = baseline.reshape(self._len)
            params['weights'] = weight_array.reshape(self._len)

        return baseline, params
