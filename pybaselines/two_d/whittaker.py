# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on April 30, 2023
@author: Donald Erb

"""

import warnings

import numpy as np

from .. import _weighting
from ._algorithm_setup import _Algorithm2D
from ..utils import (
    ParameterWarning, relative_difference
)


class _Whittaker(_Algorithm2D):
    """A base class for all Whittaker-smoothing-based algorithms."""

    @_Algorithm2D._register(sort_keys=('weights',))
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
            will be given `p - 1` weight. Default is 1e-2.
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
        Leiden University Medical Centre Report, 2005, 1(1).

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        main_diag_idx = self.whittaker_system.main_diagonal_index
        main_diagonal = self.whittaker_system.penalty[main_diag_idx].copy()
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            self.whittaker_system.penalty[main_diag_idx] = main_diagonal + weight_array
            baseline = self.whittaker_system.solve(
                self.whittaker_system.penalty, weight_array * y, overwrite_b=True
            )
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
    def airpls(self, data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

        Parameters
        ----------
        data : array-like
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, copy_weights=True
        )
        y_l1_norm = np.abs(y).sum()
        main_diag_idx = self.whittaker_system.main_diagonal_index
        main_diagonal = self.whittaker_system.penalty[main_diag_idx].copy()
        tol_history = np.empty(max_iter + 1)
        # Have to have extensive error handling since the weights can all become
        # very small due to the exp(i) term if too many iterations are performed;
        # checking the negative residual length usually prevents any errors, but
        # sometimes not so have to also catch any errors from the solvers
        for i in range(1, max_iter + 2):
            self.whittaker_system.penalty[main_diag_idx] = main_diagonal + weight_array
            try:
                output = self.whittaker_system.solve(
                    self.whittaker_system.penalty, weight_array * y, overwrite_b=True,
                    check_output=True
                )
            except np.linalg.LinAlgError:
                warnings.warn(
                    ('error occurred during fitting, indicating that "tol"'
                     ' is too low, "max_iter" is too high, or "lam" is too high'),
                    ParameterWarning
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
                     ' is too low and/or "max_iter" is too high'), ParameterWarning
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

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
    def arpls(self, data, lam=1e3, diff_order=2, max_iter=50, tol=1e-3, weights=None):
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
        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        main_diagonal = self.whittaker_system.penalty.diagonal()
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            self.whittaker_system.penalty.setdiag(main_diagonal + weight_array)
            baseline = self.whittaker_system.solve(
                self.whittaker_system.penalty, weight_array * y
            )
            new_weights = _weighting._arpls(y, baseline)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
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
        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        main_diag_idx = self.whittaker_system.main_diagonal_index
        main_diagonal = self.whittaker_system.penalty[main_diag_idx].copy()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            self.whittaker_system.penalty[main_diag_idx] = main_diagonal + weight_array
            baseline = self.whittaker_system.solve(
                self.whittaker_system.penalty, weight_array * y, overwrite_b=True
            )
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
                     '"tol" is too low and/or "max_iter" is too high'), ParameterWarning
                )
                break
            elif calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm2D._register(sort_keys=('weights',))
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
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`.asls`.
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
        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        if k is None:
            k = np.std(y) / 10
        main_diag_idx = self.whittaker_system.main_diagonal_index
        main_diagonal = self.whittaker_system.penalty[main_diag_idx].copy()
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            self.whittaker_system.penalty[main_diag_idx] = main_diagonal + weight_array
            baseline = self.whittaker_system.solve(
                self.whittaker_system.penalty, weight_array * y, overwrite_b=True
            )
            new_weights = _weighting._psalsa(y, baseline, p, k, self._len)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params
