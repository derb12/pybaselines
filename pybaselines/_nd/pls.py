# -*- coding: utf-8 -*-
"""Penalized Least Squares (PLS) methods for solving baselines.

Generalized methods that cover both Whittaker smoothing and penalized spline (P-Spline) algorithms.

Created on March 27, 2026
@author: Donald Erb

"""

import warnings

import numpy as np

from .. import _weighting
from ..utils import (
    ParameterWarning, _masked_convolve, _mollifier_kernel, pad_edges, padded_convolve,
    relative_difference
)
from .._validation import _check_scalar_variable
from ._algorithm_setup import _handle_io


class _PLSNDMixin:
    """A mixin class for providing penalized least squares methods for 1D and 2D."""

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _asls(self, data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None,
              spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False):
        """
        Fits the baseline using the asymmetric least squares (AsLS) algorithm.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Default is 1e-2.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
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

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            new_weights = _weighting._asls(y - baseline, p=p, mask=self.mask)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _airpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None,
                spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False,
                normalize_weights='deprecated'):
        """
        Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e6.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.
        normalize_weights : bool, optional
            If True, will normalize the computed weights between 0 and 1 to potentially
            improve the numerical stability. Default behavior uses the reference implementation,
            which sets weights for all negative residuals to be greater than 1.

            .. deprecated:: 1.3
                `normalize_weights` is deprecated and will be removed in version 1.5. The
                future behavior will use the reference implementation.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        if normalize_weights != 'deprecated':
            warnings.warn(
                'normalize_weights is deprecated and will be removed in version 1.5.',
                DeprecationWarning, stacklevel=2
            )
        else:
            normalize_weights = False
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(1, max_iter + 2):
            baseline = penalized_system.solve(y, weight_array)
            new_weights, residual_l1_norm, exit_early = _weighting._airpls(
                y - baseline, iteration=i, normalize_weights=normalize_weights, mask=self.mask
            )
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = residual_l1_norm / y_l1_norm
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _arpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None,
               spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False):
        """
        Asymmetrically reweighted penalized least squares smoothing (arPLS).

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e6.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._arpls(y - baseline, mask=self.mask)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _iarpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None,
                spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False):
        """
        Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e5.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(1, max_iter + 2):
            baseline = penalized_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._iarpls(
                y - baseline, iteration=i, mask=self.mask
            )
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _psalsa(self, data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3,
                weights=None, spline_degree=None, num_knots=25, num_eigens=(10, 10),
                return_dof=False):
        """
        Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

        Similar to the asymmetric least squares (AsLS) algorithm, but applies an
        exponential decay weighting to values greater than the baseline to allow
        using a higher `p` value to better fit noisy data.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
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
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
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

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        if k is None:
            k = np.std(y[weight_array > 0]) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')

        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            new_weights = _weighting._psalsa(y - baseline, p=p, k=k, mask=self.mask)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=2)
    def _derpsalsa(self, data, lam=1e6, p=1e-2, k=None, diff_order=2, max_iter=50, tol=1e-3,
                   weights=None, spline_degree=None, num_knots=10, smooth_half_window=None,
                   num_smooths=16, pad_kwargs=None, num_eigens=(10, 10), **kwargs):
        """
        Derivative Peak-Screening Asymmetric Least Squares Algorithm (derpsalsa).

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e5.
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
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        smooth_half_window : int, optional
            The half-window to use for smoothing the data before computing the first
            and second derivatives. Default is None, which will use ``len(data) / 200``.
        num_smooths : int, optional
            The number of times to smooth the data before computing the first
            and second derivatives. Default is 16.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing. Default is None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `pad_kwargs`.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        # NOTE derpsalsa doesn't currently allow 2D
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        if k is None:
            k = np.std(y[weight_array > 0]) / 10
        else:
            k = _check_scalar_variable(k, variable_name='k')
        if smooth_half_window is None:
            smooth_half_window = self._size // 200
        # could pad the data every iteration, but it is ~2-3 times slower and only affects
        # the edges, so it's not worth it
        # TODO why is padding even necessary here??? the smoothed values are only
        # used to setup partial weights from derivatives, so edge effects won't matter much
        self._deprecate_pad_kwargs(**kwargs)
        y_smooth = y
        if smooth_half_window > 0:
            smooth_kernel = _mollifier_kernel(smooth_half_window)
            if self.mask is None:
                pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
                y_smooth = pad_edges(y, smooth_half_window, **pad_kwargs, **kwargs)
                for _ in range(num_smooths):
                    y_smooth = padded_convolve(y_smooth, smooth_kernel)
                y_smooth = y_smooth[smooth_half_window:self._size + smooth_half_window]
            else:
                # no padding applied when masking
                for _ in range(num_smooths):
                    y_smooth = _masked_convolve(
                        y_smooth, smooth_kernel, self.mask, fill_nan=not self._strict_mask
                    )

        diff_y_1 = np.gradient(y_smooth)
        diff_y_2 = np.gradient(diff_y_1)
        # x @ x is same as (x**2).sum() but faster
        rms_diff_1 = np.sqrt((diff_y_1 @ diff_y_1) / self._size)
        rms_diff_2 = np.sqrt((diff_y_2 @ diff_y_2) / self._size)

        diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
        diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
        partial_weights = diff_1_weights * diff_2_weights

        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            new_weights = _weighting._derpsalsa(
                y - baseline, p=p, k=k, partial_weights=partial_weights, mask=self.mask
            )
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _brpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, max_iter_2=50,
               tol_2=1e-3, weights=None, spline_degree=None, num_knots=10, num_eigens=(10, 10),
               return_dof=False):
        """
        Bayesian Reweighted Penalized Least Squares (BrPLS) baseline.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e5.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter_2 : int, optional
            The number of iterations for updating the proportion of data occupied by peaks.
            Default is 50.
        tol_2 : float, optional
            The exit criteria for the difference between the calculated proportion of data
            occupied by peaks. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Wang, Q., et al. Spectral baseline estimation using penalized least squares
        with weights derived from the Bayesian method. Nuclear Science and Techniques,
        2022, 140, 250-257.

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        beta = 0.5
        j_max = 0
        baseline = y
        baseline_weights = weight_array
        tol_history = np.zeros((max_iter_2 + 2, max(max_iter, max_iter_2) + 1))
        success_outer = False
        # implementation note: weight_array must always be updated since otherwise when
        # reentering the inner loop, new_baseline and baseline would be the same; instead,
        # use baseline_weights to track which weights produced the output baseline
        for i in range(max_iter_2 + 1):
            success_inner = False
            for j in range(max_iter + 1):
                new_baseline = penalized_system.solve(y, weight_array)
                new_weights, exit_early = _weighting._brpls(
                    y - new_baseline, beta=beta, mask=self.mask
                )
                if exit_early:
                    j -= 1  # reduce j so that output tol_history indexing is correct
                    tol_2 = np.inf  # ensure it exits outer loop
                    break
                # Paper used norm(old - new) / norm(new) rather than old in the denominator,
                # but I use old in the denominator instead to be consistent with all other
                # algorithms; does not make a major difference
                calc_difference = relative_difference(baseline, new_baseline)
                tol_history[i + 1, j] = calc_difference
                if calc_difference < tol:
                    success_inner = True
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
                success_outer = True
                break
            beta = 1 - weight_mean

        params = {
            'weights': baseline_weights, 'tol_history': tol_history[:i + 2, :max(i, j_max) + 1],
            'result': result_class(penalized_system, baseline_weights),
            'success': success_inner and success_outer
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _lsrpls(self, data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None,
                spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False,
                alternate_weighting=False):
        """
        Locally Symmetric Reweighted Penalized Least Squares (LSRPLS).

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e5.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 2 (second order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        num_eigens : int or Sequence[int, int] or None, optional
            The number of eigenvalues for eigendecomposition of the penalty matrices. Can be a
            single value or a sequence of ints with length equal to the dimensions of `data`.
            Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower. Must be greater
            than `diff_order`. Default is (10, 10). Only used if `data` is two dimensional
            and `spline_degree` is not None.
        return_dof : bool, optional
            If True and `num_eigens` is not None, then the effective degrees of freedom for
            each eigenvector will be calculated and returned in the parameter dictionary.
            Default is False since the calculation takes time. Only used if `data` is
            two dimensional.
        alternate_weighting : bool, optional
            If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
            the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
            a prefactor term of ``exp(t)``. See the Notes section below for more details.

            .. versionadded:: 1.3.0

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
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

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(1, max_iter + 2):
            baseline = penalized_system.solve(y, weight_array)
            new_weights, exit_early = _weighting._lsrpls(
                y - baseline, iteration=i, alternate_weighting=alternate_weighting, mask=self.mask
            )
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i],
            'result': result_class(penalized_system, weight_array), 'success': success
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _mixture_model(self, data, lam=1e5, p=1e-2, num_knots=25, spline_degree=None,
                       diff_order=3, max_iter=50, tol=1e-3, weights=None,
                       symmetric=False):
        """
        Considers the data as a mixture model composed of noise and peaks.

        Weights are iteratively assigned by calculating the probability each value in
        the residual belongs to a normal distribution representing the noise.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e5.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `1 - p` weight. Used to set the initial weights before performing
            expectation-maximization. Default is 1e-2.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 3 (third order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        symmetric : bool, optional
            If False (default), the total mixture model will be composed of one normal
            distribution for the noise and one uniform distribution for positive non-noise
            residuals. If True, an additional uniform distribution will be added to the
            mixture model for negative non-noise residuals. Only need to set `symmetric`
            to True when peaks are both positive and negative.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
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
        de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
        Intelligent Laboratory Systems, 2012, 117, 56-60.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        # NOTE mixture_model doesn't currently allow Whittaker smoothing
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots
        )
        # scale y between -1 and 1 so that the residual fit is more numerically stable
        # TODO is this still necessary now that expectation-maximization is used? -> still
        # helps to prevent overflows when using gaussian
        y_domain = np.polynomial.polyutils.getdomain(y[weight_array > 0].ravel())
        y = np.polynomial.polyutils.mapdomain(y, y_domain, np.array([-1., 1.]))

        if weights is not None:
            baseline = penalized_system.solve(y, weight_array)
        else:
            # perform 2 iterations: first is a least-squares fit and second is initial
            # reweighted fit; 2 fits are needed to get weights to have a decent starting
            # distribution for the expectation-maximization
            if symmetric and not 0.2 < p < 0.8:
                # p values far away from 0.5 with symmetric=True give bad initial weights
                # for the expectation maximization
                warnings.warn(
                    'should use a p value closer to 0.5 when "symmetric" is True',
                    ParameterWarning, stacklevel=2
                )
            for _ in range(2):
                baseline = penalized_system.solve(y, weight_array)
                weight_array = _weighting._asls(y - baseline, p=p, mask=self.mask)

        residual = y - baseline
        # the 0.2 * std(residual) is an "okay" starting sigma estimate
        sigma = 0.2 * np.std(residual[weight_array > 0])
        fraction_noise = 0.5
        if symmetric:
            fraction_positive = 0.25
        else:
            fraction_positive = 1 - fraction_noise
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            posterior_prob_noise, sigma, fraction_noise, fraction_positive = _weighting._em(
                residual, sigma=sigma, fraction_noise=fraction_noise,
                fraction_positive=fraction_positive, symmetric=symmetric, mask=self.mask
            )
            calc_difference = relative_difference(weight_array, posterior_prob_noise)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break

            weight_array = posterior_prob_noise
            baseline = penalized_system.solve(y, weight_array)
            residual = y - baseline

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }

        baseline = np.polynomial.polyutils.mapdomain(baseline, np.array([-1., 1.]), y_domain)

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',), mask_support=1)
    def _irsqr(self, data, lam=1e3, quantile=0.05, num_knots=25, spline_degree=None,
               diff_order=3, max_iter=100, tol=1e-6, weights=None, eps=None):
        """
        Iterative Reweighted Spline Quantile Regression (IRSQR).

        Fits the baseline using quantile regression with penalized splines.

        Parameters
        ----------
        data : array-like, shape (N,) or (M, N)
            The y-values of the measured data. Must not contain missing data (NaN) or Inf.
        lam : float or Sequence[float, float], optional
            The smoothing parameter. Can be a single value or a sequence of floats with length
            equal to the dimensions of `data`. Larger values will create smoother baselines.
            Default is 1e3.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 0.05.
        num_knots : int or Sequence[int, int], optional
            The number of knots for the splines. Can be a single value or a sequence of ints
            with length equal to the dimensions of `data`. Default is 25. Only used if
            `spline_degree` is not None.
        spline_degree : None or int or Sequence[int, int], optional
            The degree of the splines. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Default is None, which will use Whittaker
            smoothing.
        diff_order : int or Sequence[int, int], optional
            The order of the difference matrix. Can be a single value or a sequence of ints with
            length equal to the dimensions of `data`. Must be greater than 0.
            Default is 3 (third order difference matrix).
        max_iter : int, optional
            The max number of fit iterations. Default is 100.
        tol : float, optional
            The exit criteria. Default is 1e-6.
        weights : array-like, shape (N,) or (M, N), optional
            The weighting array. If None (default), then the initial weights
            will be an array with the same shape as `data` with all values set to 1.
        eps : float, optional
            A small value added to the square of the residual to prevent dividing by 0.
            Default is None, which uses the square of the maximum-absolute-value of the
            residual each iteration multiplied by 1e-4.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,) or (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,) or (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': WhittakerResult or WhittakerResult2D or PSplineResult or PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations. The type depends on the dimensions of `data` and if
                `spline_degree` was None.
            * 'success' : bool
                True if the method converged successfully, otherwise False.
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        Raises
        ------
        ValueError
            Raised if `quantile` is not between 0 and 1.

        References
        ----------
        Han, Q., et al. Iterative Reweighted Quantile Regression Using Augmented Lagrangian
        Optimization for Baseline Correction. 2018 5th International Conference on Information
        Science and Control Engineering (ICISCE), 2018, 280-284.

        """
        if not 0 < quantile < 1:
            raise ValueError('quantile must be between 0 and 1')

        # NOTE irsqr doesn't currently allow Whittaker smoothing
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots
        )
        old_coef = np.zeros(penalized_system.tot_bases)
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            calc_difference = relative_difference(old_coef, penalized_system.coef)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            old_coef = penalized_system.coef
            weight_array = _weighting._quantile(
                y - baseline, quantile=quantile, eps=eps, mask=self.mask
            )

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array), 'success': success
        }

        return baseline, params
