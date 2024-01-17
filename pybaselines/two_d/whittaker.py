# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on April 30, 2023
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.sparse import diags

from .. import _weighting
from ._algorithm_setup import _Algorithm2D
from ._whittaker_utils import PenalizedSystem2D
from ..utils import (
    ParameterWarning, relative_difference
)
from .._validation import _check_optional_array


class _Whittaker(_Algorithm2D):
    """A base class for all Whittaker-smoothing-based algorithms."""

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, use_banded=True, use_lower=True
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(weight_array), weight_array * y,
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
        data : array-like, shape (N,)
            The y-values of the measured data, with `N` data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 1e-2.
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
        elif (np.asarray(diff_order) < 2).any():
            raise ValueError('diff_order must be 2 or greater')

        if weights is None:
            _, _, pseudo_inverse = self._setup_polynomial(
                data, weights=None, poly_order=2, calc_vander=True, calc_pinv=True
            )
            baseline = self.vandermonde @ (pseudo_inverse @ data.ravel())
            weights = _weighting._asls(data.ravel(), baseline, p).reshape(self._len)

        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._len, lam_1, diff_order=1, use_banded=False)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            penalized_system_1.add_diagonal(weight_array * weight_array)
            baseline = self.whittaker_system.solve(
                self.whittaker_system.penalty + penalized_system_1.penalty,
                penalized_system_1.penalty * y
            )
            assert y.shape == (self._len[0] * self._len[1],)
            assert baseline.shape == (self._len[0] * self._len[1],)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
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
            data, lam, diff_order, weights, copy_weights=True, use_banded=True, use_lower=True

        )
        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        # Have to have extensive error handling since the weights can all become
        # very small due to the exp(i) term if too many iterations are performed;
        # checking the negative residual length usually prevents any errors, but
        # sometimes not so have to also catch any errors from the solvers
        for i in range(1, max_iter + 2):
            try:
                output = self.whittaker_system.solve(
                    self.whittaker_system.add_diagonal(weight_array), weight_array * y,
                    overwrite_b=True
                )
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, use_banded=True, use_lower=True
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
            new_weights = _weighting._arpls(y, baseline)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
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
        elif (np.asarray(diff_order) < 2).any():
            raise ValueError('diff_order must be 2 or greater')

        y, weight_array = self._setup_whittaker(data, lam, diff_order, weights)
        penalized_system_1 = PenalizedSystem2D(self._len, 1, diff_order=1, use_banded=False)
        # W + P_1 + (I - eta * W) @ P_n -> P_1 + P_n + W @ (I - eta * P_n)
        partial_penalty = self.whittaker_system.penalty + penalized_system_1.penalty
        partial_penalty_2 = -eta * self.whittaker_system.penalty
        partial_penalty_2.setdiag(partial_penalty_2.diagonal() + 1)
        weight_matrix = diags(weight_array)
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.whittaker_system.solve(
                partial_penalty + weight_matrix @ partial_penalty_2, weight_array * y,
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

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, use_banded=True, use_lower=True
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
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
                     '"tol" is too low and/or "max_iter" is too high'), ParameterWarning,
                     stacklevel=2
                )
                break
            elif calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

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
        alpha : array-like, shape (N,), optional
            An array of values that control the local value of `lam` to better
            fit peak and non-peak regions. If None (default), then the initial values
            will be an array with size equal to N and all values set to 1.

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

        # use a sparse matrix to maintain sparsity after multiplication
        alpha_matrix = diags(alpha_array.ravel(), format='csr')
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            lhs = alpha_matrix * self.whittaker_system.penalty
            lhs.setdiag(lhs.diagonal() + weight_array)
            baseline = self.whittaker_system.solve(
                lhs, weight_array * y
            )
            new_weights, residual = _weighting._aspls(y, baseline)
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

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_keys=('weights',), reshape_baseline=True
    )
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
        y, weight_array = self._setup_whittaker(
            data, lam, diff_order, weights, use_banded=True, use_lower=True
        )
        if k is None:
            k = np.std(y) / 10
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(weight_array), weight_array * y,
                overwrite_b=True
            )
            new_weights = _weighting._psalsa(y, baseline, p, k, self._len[0] * self._len[1])
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params
