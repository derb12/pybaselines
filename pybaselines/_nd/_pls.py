# -*- coding: utf-8 -*-
"""Penalized Least Squares (PLS) methods for solving baselines.

Generalized methods that cover both Whittaker smoothing and penalized spline (P-Spline) algorithms.

Created on March 27, 2026
@author: Donald Erb

"""

import numpy as np

from .. import _weighting
from ..utils import relative_difference
from ._algorithm_setup import _handle_io


class _PLSNDMixin:
    """A mixin class for providing penalized least squares methods for 1D and 2D."""

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',))
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

        Biessy, G. Whittaker-Henderson smoothing revisited: A modern statistical framework for
        practical use. ASTIN Bulletin, 2025, 1-31.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = penalized_system.solve(y, weight_array)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': result_class(penalized_system, weight_array)
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params

    @_handle_io(sort_keys=('weights',), reshape_keys=('weights',))
    def _airpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None,
                spline_degree=None, num_knots=25, num_eigens=(10, 10), return_dof=False,
                normalize_weights=False):
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
            improve the numerical stability. Set to False (default) to use the original
            implementation, which sets weights for all negative residuals to be greater than 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
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
            * 'dof' : numpy.ndarray, shape (`num_eigens[0]`, `num_eigens[1]`)
                Only if `return_dof` is True. The effective degrees of freedom associated
                with each eigenvector. Lower values signify that the eigenvector was
                less important for the fit.

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Biessy, G. Whittaker-Henderson smoothing revisited: A modern statistical framework for
        practical use. ASTIN Bulletin, 2025, 1-31.

        """
        y, weight_array, penalized_system, result_class = self._setup_pls(
            data, lam=lam, diff_order=diff_order, weights=weights, spline_degree=spline_degree,
            num_knots=num_knots, num_eigens=num_eigens
        )
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = penalized_system.solve(y, weight_array)
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

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i],
            'result': result_class(penalized_system, weight_array)
        }
        if return_dof:
            params['dof'] = params['result'].relative_dof()

        return baseline, params
