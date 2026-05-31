# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on August 4, 2021
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import grey_opening

from . import _weighting
from ._algorithm_setup import _Algorithm, _class_wrapper
from ._banded_utils import _add_diagonals, _shift_rows, _sparse_to_banded, diff_penalty_matrix
from ._nd.pls import _PLSNDMixin
from ._spline_utils import _basis_midpoints
from ._validation import (
    _check_lam, _check_optional_array, _check_scalar_variable, _check_spline_degree
)
from .results import PSplineResult
from .utils import _sort_array, relative_difference


class _Spline(_Algorithm, _PLSNDMixin):
    """A base class for all spline algorithms."""

    def mixture_model(self, data, lam=1e5, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                      max_iter=50, tol=1e-3, weights=None, symmetric=False):
        """
        Considers the data as a mixture model composed of noise and peaks.

        Weights are iteratively assigned by calculating the probability each value in
        the residual belongs to a normal distribution representing the noise.

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
            will be given `1 - p` weight. Used to set the initial weights before performing
            expectation-maximization. Default is 1e-2.
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
            will be an array with size equal to N and all values set to 1, and then
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

        References
        ----------
        de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
        Intelligent Laboratory Systems, 2012, 117, 56-60.

        Ghojogh, B., et al. Fitting A Mixture Distribution to Data: Tutorial. arXiv
        preprint arXiv:1901.06708, 2019.

        """
        _check_spline_degree(spline_degree)
        return super()._mixture_model(
            data, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            symmetric=symmetric
        )

    def irsqr(self, data, lam=100, quantile=0.05, num_knots=100, spline_degree=3,
              diff_order=3, max_iter=100, tol=1e-6, weights=None, eps=None):
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

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
        _check_spline_degree(spline_degree)
        return super()._irsqr(
            data, lam=lam, quantile=quantile, num_knots=num_knots, spline_degree=spline_degree,
            diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights, eps=eps
        )

    def pspline_asls(self, data, lam=1e3, p=1e-2, num_knots=100, spline_degree=3, diff_order=2,
                     max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the asymmetric least squares (AsLS) algorithm.

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
            will be given `1 - p` weight. Default is 1e-2.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

        See Also
        --------
        Baseline.asls

        References
        ----------
        Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

        Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
        Leiden University Medical Centre Report, 2005, [unpublished].

        Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._asls(
            data, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

    @_Algorithm._handle_io(sort_keys=('weights',))
    def pspline_iasls(self, data, lam=1e1, p=1e-2, lam_1=1e-4, num_knots=100,
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
            than the baseline will be given ``p**2`` weight, and values less than the baseline
            will be given ``(1 - p)**2`` weight. Default is 1e-2.
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

                .. versionchanged:: 1.3.0
                    Prior to version 1.3.0, the returned weights were the non-squared
                    values (ie. ``p`` or ``1 - p``).

            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1 or if `diff_order` is less than 2.

        See Also
        --------
        Baseline.iasls

        Notes
        -----
        Although both ``pspline_iasls`` and :meth:`~.Baseline.pspline_asls` use `p` for defining
        the weights, the appropriate `p` value for ``pspline_iasls`` will be approximately equal
        to the square root of the value used for ``pspline_asls`` when `p` is small since
        ``pspline_iasls`` uses squared weights.

        References
        ----------
        He, S., et al. Baseline correction for raman spectra using an improved
        asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

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
            weights = _weighting._iasls(data, baseline, p)

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        # B.T @ D_1.T @ D_1 @ B and B.T @ D_1.T @ D_1 @ y
        d1_penalty = (
            pspline.basis.basis.T
            @ (_check_lam(lam_1) * diff_penalty_matrix(self._size, 1))
        )
        partial_rhs = d1_penalty @ y
        # now change d1_penalty back to banded array
        d1_penalty = _sparse_to_banded(d1_penalty @ pspline.basis.basis)[0]
        if pspline.lower:
            d1_penalty = d1_penalty[len(d1_penalty) // 2:]
        pspline.add_penalty(d1_penalty)

        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array, rhs_extra=partial_rhs)
            new_weights = _weighting._iasls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': PSplineResult(pspline, weight_array, rhs_extra=d1_penalty)
        }

        return baseline, params

    def pspline_airpls(self, data, lam=1e3, num_knots=100, spline_degree=3,
                       diff_order=2, max_iter=50, tol=1e-3, weights=None, normalize_weights=False):
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
            The number of knots for the spline. Default is 100.
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
        normalize_weights : bool, optional
            If True, will normalize the computed weights between 0 and 1 to potentially
            improve the numerical stability. Set to False (default) to use the original
            implementation, which sets weights for all negative residuals to be greater than 1.

        Returns
        -------
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        See Also
        --------
        Baseline.airpls

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._airpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            normalize_weights=normalize_weights
        )

    def pspline_arpls(self, data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
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
            The number of knots for the spline. Default is 100.
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        See Also
        --------
        Baseline.arpls

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._arpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

    @_Algorithm._handle_io(sort_keys=('weights',))
    def pspline_drpls(self, data, lam=1e3, eta=0.5, num_knots=100, spline_degree=3,
                      diff_order=2, max_iter=50, tol=1e-3, weights=None):
        """
        A penalized spline version of the drPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        eta : float
            A term for controlling the value of lam; should be between 0 and 1.
            Low values will produce smoother baselines, while higher values will
            more aggressively fit peaks. Default is 0.5.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
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
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `eta` is not between 0 and 1 or if `diff_order` is less than 2.

        See Also
        --------
        Baseline.drpls

        References
        ----------
        Xu, D. et al. Baseline correction method based on doubly reweighted
        penalized least squares, Applied Optics, 2019, 58, 3913-3920.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 <= eta <= 1:
            raise ValueError('eta must be between 0 and 1')
        elif diff_order < 2:
            raise ValueError('diff_order must be 2 or greater')

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam,
            allow_lower=False, reverse_diags=False
        )
        # B.T @ W @ B + B.T @ P_1 @ B + (I - eta * W) @ P_n
        # -> B.T @ W @ B + B.T @ P_1 @ B + P_n - eta * W @ P_n
        # reverse P_n for the eta * W @ P_n calculation to match the original diagonal
        # structure of the sparse matrix after multiplication with interpolated weights
        diff_n_diagonals = -eta * pspline.penalty[::-1]
        # save bands from here since adding B.T @ P_1 @ B may change band structure later
        shifted_bands = pspline.num_bands

        # B.T @ P_1 @ B + P_n
        d1_penalty = _sparse_to_banded(
            pspline.basis.basis.T @ diff_penalty_matrix(self._size, 1) @ pspline.basis.basis
        )[0]
        pspline.add_penalty(d1_penalty)

        interp_pts = _basis_midpoints(pspline.basis.knots, pspline.basis.spline_degree)
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            diff_n_w_diagonals = _shift_rows(
                diff_n_diagonals * np.interp(interp_pts, self.x, weight_array),
                shifted_bands, shifted_bands
            )
            penalty = _add_diagonals(pspline.penalty, diff_n_w_diagonals, lower_only=False)
            baseline = pspline.solve(y, weight_array, penalty=penalty)
            new_weights, exit_early = _weighting._drpls(y, baseline, i)
            if exit_early:
                i -= 1  # reduce i so that output tol_history indexing is correct
                break

            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i - 1] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i],
            'result': PSplineResult(pspline, weight_array, penalty=penalty)
        }

        return baseline, params

    def pspline_iarpls(self, data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
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
            The number of knots for the spline. Default is 100.
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        See Also
        --------
        Baseline.iarpls

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._iarpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

    @_Algorithm._handle_io(sort_keys=('weights', 'alpha'))
    def pspline_aspls(self, data, lam=1e4, num_knots=100, spline_degree=3, diff_order=2,
                      max_iter=100, tol=1e-3, weights=None, alpha=None, asymmetric_coef=2.,
                      alternate_weighting=True):
        """
        A penalized spline version of the asPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
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
        asymmetric_coef : float, optional
            The asymmetric coefficient for the weighting. Higher values leads to a steeper
            weighting curve (ie. more step-like). Default is 2, as used in the asPLS paper [1]_.
            Note that a value of 4 results in the weighting scheme used in the NasPLS
            (Non-sensitive-areas adaptive smoothness penalized least squares smoothing) algorithm
            [2]_.

            .. versionadded:: 1.2.0

        alternate_weighting : bool, optional
            If True (default), subtracts the mean of the negative residuals within the weighting
            equation. If False, uses the weighting equation as stated within the asPLS paper [1]_.
            See the Notes section below for more details.

            .. versionadded:: 1.3.0

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
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `alpha` and `data` do not have the same shape. Also raised if
            `asymmetric_coef` is not greater than 0.

        See Also
        --------
        Baseline.aspls

        Notes
        -----
        The weighting scheme as written in the asPLS paper [1]_ does not reproduce the paper's
        results for noisy data. By subtracting the mean of negative residuals (ie. the negative
        values of ``data - baseline``) within the weighting scheme, the asPLS paper's results can
        be correctly replicated (see https://github.com/derb12/pybaselines/issues/40 for more
        details). Given this discrepancy, the default for ``pspline_aspls`` is to also subtract the
        negative residuals within the weighting. To use the weighting scheme as it is written in
        the asPLS paper, set ``alternate_weighting`` to False.

        References
        ----------
        .. [1] Zhang, F., et al. Baseline correction for infrared spectra using
            adaptive smoothness parameter penalized least squares method.
            Spectroscopy Letters, 2020, 53(3), 222-233.
        .. [2] Zhang, F., et al. An automatic baseline correction method based on reweighted
            penalized least squares method for non-sensitive areas. Vibrational Spectroscopy,
            2025, 138, 103806.
        .. [3] Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
            Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam,
            allow_lower=False, reverse_diags=True
        )
        alpha_array = _check_optional_array(
            self._size, alpha, check_finite=self._check_finite, name='alpha'
        )
        if self._sort_order is not None and alpha is not None:
            alpha_array = alpha_array[self._sort_order]
        asymmetric_coef = _check_scalar_variable(asymmetric_coef, variable_name='asymmetric_coef')

        interp_pts = _basis_midpoints(pspline.basis.knots, pspline.basis.spline_degree)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            # convert alpha_array from len(y) to basis.shape[1]
            alpha_penalty = _shift_rows(
                pspline.penalty * np.interp(interp_pts, self.x, alpha_array),
                pspline.num_bands, pspline.num_bands
            )
            baseline = pspline.solve(y, weight_array, penalty=alpha_penalty)
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
            abs_d = np.abs(residual)
            alpha_array = abs_d / abs_d.max()

        params = {
            'weights': weight_array, 'alpha': alpha_array, 'tol_history': tol_history[:i + 1],
            'result': PSplineResult(pspline, weight_array, penalty=alpha_penalty)
        }

        return baseline, params

    def pspline_psalsa(self, data, lam=1e3, p=0.5, k=None, num_knots=100, spline_degree=3,
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
            will be given `1 - p` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~.Baseline.asls`.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
            than 0.

        See Also
        --------
        Baseline.psalsa

        References
        ----------
        Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
        for analytical instruments. 2014 IEEE 11th International Multi-Conference on
        Systems, Signals, and Devices, 2014, 1-5.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._psalsa(
            data, lam=lam, p=p, k=k, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

    def pspline_derpsalsa(self, data, lam=1e2, p=1e-2, k=None, num_knots=100, spline_degree=3,
                          diff_order=2, max_iter=50, tol=1e-3, weights=None,
                          smooth_half_window=None, num_smooths=16, pad_kwargs=None, **kwargs):
        """
        A penalized spline version of the derpsalsa algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e2.
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
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
            than 0.

        See Also
        --------
        Baseline.derpsalsa

        References
        ----------
        Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
        automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
        51(10), 2061-2065.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._derpsalsa(
            data, lam=lam, p=p, k=k, num_knots=num_knots, spline_degree=spline_degree,
            diff_order=diff_order, max_iter=max_iter, tol=tol, weights=weights,
            smooth_half_window=smooth_half_window, num_smooths=num_smooths,
            pad_kwargs=pad_kwargs, **kwargs
        )

    @_Algorithm._handle_io(sort_keys=('weights',))
    def pspline_mpls(self, data, half_window=None, lam=1e3, p=0.0, num_knots=100, spline_degree=3,
                     diff_order=2, tol=None, max_iter=None, weights=None, window_kwargs=None,
                     **kwargs):
        """
        A penalized spline version of the morphological penalized least squares (MPLS) algorithm.

        .. deprecated:: 1.3.0
            ``pspline_mpls`` is deprecated and will be removed in version 1.5.0. Use
            :func:`pybaselines.utils.pspline_smooth` with the output weights of
            :meth:`~Baseline.mpls` instead.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.estimate_window` and `window_kwargs`.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Anchor points
            identified by the procedure in [1]_ are given a weight of `1 - p`, and all
            other points have a weight of `p`. Default is 0.0.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        tol : float, optional, deprecated

            .. deprecated:: 1.2.0
                ``tol`` is deprecated since it was not used within pspline_mpls
                and will be removed in pybaselines version 1.4.0.

        max_iter : int, optional, deprecated

            .. deprecated:: 1.2.0
                ``max_iter`` is deprecated since it was not used within pspline_mpls
                and will be removed in pybaselines version 1.4.0.

        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the weights will be
            calculated following the procedure in [1]_.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.estimate_window` for
            estimating the half window if `half_window` is None. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `window_kwargs`.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'half_window': int
                The half window used for the morphological calculations.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        Raises
        ------
        ValueError
            Raised if p is not between 0 and 1.

        See Also
        --------
        Baseline.mpls

        References
        ----------
        .. [1] Li, Zhong, et al. Morphological weighted penalized least squares for
            background correction. Analyst, 2013, 138, 4483-4492.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        warnings.warn(
            ('"pspline_mpls" is deprecated and will be removed in version 1.5.0. Use '
             'pybaselines.utils.pspline_smooth with the output weights of "mpls" instead.'),
             DeprecationWarning, stacklevel=2
        )
        if not 0 <= p <= 1:
            raise ValueError('p must be between 0 and 1')
        if tol is not None or max_iter is not None:
            warnings.warn(
                ('passing "tol" or "max_iter" to pspline_mpls was deprecated in version 1.2.0 and '
                 'will be removed in version 1.4.0'),
                DeprecationWarning, stacklevel=2
            )

        y, half_wind = self._setup_morphology(data, half_window, window_kwargs, **kwargs)
        if weights is not None:
            w = weights
        else:
            rough_baseline = grey_opening(y, [2 * half_wind + 1])
            diff = np.diff(
                np.concatenate([rough_baseline[:1], rough_baseline, rough_baseline[-1:]])
            )
            # diff == 0 means the point is on a flat segment, and diff != 0 means the
            # adjacent point is not the same flat segment. The union of the two finds
            # the endpoints of each segment, and np.flatnonzero converts the mask to
            # indices; indices will always be even-sized.
            indices = np.flatnonzero(
                ((diff[1:] == 0) | (diff[:-1] == 0)) & ((diff[1:] != 0) | (diff[:-1] != 0))
            )
            w = np.full(self._shape, p)
            # find the index of min(y) in the region between flat regions
            for previous_segment, next_segment in zip(indices[1::2], indices[2::2]):
                index = np.argmin(y[previous_segment:next_segment + 1]) + previous_segment
                w[index] = 1 - p

            # have to invert the weight ordering the matching the original input y ordering
            # since it will be sorted within _setup_spline
            w = _sort_array(w, self._inverted_order)

        _, weight_array, pspline = self._setup_spline(
            y, w, spline_degree, num_knots, True, diff_order, lam
        )
        baseline = pspline.solve(y, weight_array)

        params = {
            'weights': weight_array, 'half_window': half_wind,
            'result': PSplineResult(pspline, weight_array)
        }
        return baseline, params

    def pspline_brpls(self, data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                      max_iter=50, tol=1e-3, max_iter_2=50, tol_2=1e-3, weights=None):
        """
        A penalized spline version of the BrPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
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
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.

        Returns
        -------
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray, shape (J, K)
                An array containing the calculated tolerance values for each iteration of
                both threshold values and fit values. Index 0 are the tolerance values for
                the difference in the peak proportion, and indices >= 1 are the tolerance values
                for each fit. All values that were not used in fitting have values of 0. Shape J
                is 2 plus the number of iterations for the threshold to converge (related to
                `max_iter_2`, `tol_2`), and shape K is the maximum of the number of
                iterations for the threshold and the maximum number of iterations for all of
                the fits of the various threshold values (related to `max_iter` and `tol`).
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        See Also
        --------
        Baseline.brpls

        References
        ----------
        Wang, Q., et al. Spectral baseline estimation using penalized least squares
        with weights derived from the Bayesian method. Nuclear Science and Techniques,
        2022, 140, 250-257.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._brpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            max_iter_2=max_iter_2, tol_2=tol_2, weights=weights,
            spline_degree=spline_degree, num_knots=num_knots
        )

    def pspline_lsrpls(self, data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                       max_iter=50, tol=1e-3, weights=None, alternate_weighting=False):
        """
        A penalized spline version of the LSRPLS algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points. Must not
            contain missing data (NaN) or Inf.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
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
        alternate_weighting : bool, optional
            If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
            the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
            a prefactor term of ``exp(t)``. See the Notes section below for more details.

            .. versionadded:: 1.3.0

        Returns
        -------
        numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult
                An object that can use the results of the fit to perform additional
                calculations.

        See Also
        --------
        Baseline.lsrpls

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
        _check_spline_degree(spline_degree)
        return super()._lsrpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            alternate_weighting=alternate_weighting
        )


_spline_wrapper = _class_wrapper(_Spline)


@_spline_wrapper
def mixture_model(data, lam=1e5, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                  max_iter=50, tol=1e-3, weights=None, symmetric=False, x_data=None):
    """
    Considers the data as a mixture model composed of noise and peaks.

    Weights are iteratively assigned by calculating the probability each value in
    the residual belongs to a normal distribution representing the noise.

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
        will be given `1 - p` weight. Used to set the initial weights before performing
        expectation-maximization. Default is 1e-2.
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
        will be an array with size equal to N and all values set to 1, and then
        two iterations of reweighted least-squares are performed to provide starting
        weights for the expectation-maximization of the mixture model.
    symmetric : bool, optional
        If False (default), the total mixture model will be composed of one normal
        distribution for the noise and one uniform distribution for positive non-noise
        residuals. If True, an additional uniform distribution will be added to the
        mixture model for negative non-noise residuals. Only need to set `symmetric`
        to True when peaks are both positive and negative.
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


@_spline_wrapper
def irsqr(data, lam=100, quantile=0.05, num_knots=100, spline_degree=3, diff_order=3,
          max_iter=100, tol=1e-6, weights=None, eps=None, x_data=None):
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


def corner_cutting(data, x_data=None, max_iter=100):
    """
    Iteratively removes corner points and creates a Bezier spline from the remaining points.

    .. deprecated:: 1.3.0
        ``corner_cutting`` was moved to `pybaselines.classification.
        `pybaselines.spline.corner_cutting` will be removed in version 1.5.0.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    max_iter : int, optional
        The maximum number of iterations to try to remove corner points. Default is
        100. Typically all corner points are removed in 10 to 20 iterations.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    References
    ----------
    Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline
    Construction. Analyst, 2015, 140(23), 7984-7996.

    """
    warnings.warn(
        '"corner_cutting" was moved to pybaselines.classification in version 1.3.0.'
        '"pybaselines.spline.corner_cutting will be removed in version 1.5',
        DeprecationWarning, stacklevel=4
    )
    from .classification import corner_cutting as classification_cc
    return classification_cc(data, x_data=x_data, max_iter=max_iter)


@_spline_wrapper
def pspline_asls(data, lam=1e3, p=1e-2, num_knots=100, spline_degree=3, diff_order=2,
                 max_iter=50, tol=1e-3, weights=None, x_data=None):
    """
    A penalized spline version of the asymmetric least squares (AsLS) algorithm.

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
        will be given `1 - p` weight. Default is 1e-2.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
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

    Raises
    ------
    ValueError
        Raised if `p` is not between 0 and 1.

    See Also
    --------
    pybaselines.whittaker.asls

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, [unpublished].

    Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_iasls(data, x_data=None, lam=1e1, p=1e-2, lam_1=1e-4, num_knots=100,
                  spline_degree=3, max_iter=50, tol=1e-3, weights=None, diff_order=2):
    """
    A penalized spline version of the IAsLS algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e1.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight. Default is 1e-2.
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
    pybaselines.whittaker.iasls

    References
    ----------
    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_airpls(data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                   max_iter=50, tol=1e-3, weights=None, x_data=None, normalize_weights=False):
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
        The number of knots for the spline. Default is 100.
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
    normalize_weights : bool, optional
        If True, will normalize the computed weights between 0 and 1 to potentially
        improve the numerical stability. Set to False (default) to use the original
        implementation, which sets weights for all negative residuals to be greater than 1.

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
    pybaselines.whittaker.airpls

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_arpls(data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                  max_iter=50, tol=1e-3, weights=None, x_data=None):
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
        The number of knots for the spline. Default is 100.
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
    pybaselines.whittaker.arpls

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_drpls(data, lam=1e3, eta=0.5, num_knots=100, spline_degree=3, diff_order=2,
                  max_iter=50, tol=1e-3, weights=None, x_data=None):
    """
    A penalized spline version of the drPLS algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e3.
    eta : float
        A term for controlling the value of lam; should be between 0 and 1.
        Low values will produce smoother baselines, while higher values will
        more aggressively fit peaks. Default is 0.5.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
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

    Raises
    ------
    ValueError
        Raised if `eta` is not between 0 and 1 or if `diff_order` is less than 2.

    See Also
    --------
    pybaselines.whittaker.drpls

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_iarpls(data, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                   max_iter=50, tol=1e-3, weights=None, x_data=None):
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
        The number of knots for the spline. Default is 100.
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
    pybaselines.whittaker.iarpls

    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_aspls(data, lam=1e4, num_knots=100, spline_degree=3, diff_order=2,
                  max_iter=100, tol=1e-3, weights=None, alpha=None, x_data=None,
                  asymmetric_coef=2., alternate_weighting=True):
    """
    A penalized spline version of the asPLS algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e3.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
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
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
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

    See Also
    --------
    pybaselines.whittaker.aspls

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
    .. [3] Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_psalsa(data, lam=1e3, p=0.5, k=None, num_knots=100, spline_degree=3,
                   diff_order=2, max_iter=50, tol=1e-3, weights=None, x_data=None):
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
        will be given `1 - p` weight. Default is 0.5.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :meth:`~.Baseline.asls`.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
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

    Raises
    ------
    ValueError
        Raised if `p` is not between 0 and 1. Also raised if `k` is not greater
        than 0.

    See Also
    --------
    pybaselines.whittaker.psalsa

    References
    ----------
    Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
    for analytical instruments. 2014 IEEE 11th International Multi-Conference on
    Systems, Signals, and Devices, 2014, 1-5.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_derpsalsa(data, lam=1e2, p=1e-2, k=None, num_knots=100, spline_degree=3,
                      diff_order=2, max_iter=50, tol=1e-3, weights=None,
                      smooth_half_window=None, num_smooths=16, x_data=None, pad_kwargs=None,
                      **kwargs):
    """
    A penalized spline version of the derpsalsa algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e2.
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
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
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
    smooth_half_window : int, optional
        The half-window to use for smoothing the data before computing the first
        and second derivatives. Default is None, which will use ``len(data) / 200``.
    num_smooths : int, optional
        The number of times to smooth the data before computing the first
        and second derivatives. Default is 16.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
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

    See Also
    --------
    pybaselines.whittaker.derpsalsa

    References
    ----------
    Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
    automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
    51(10), 2061-2065.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_mpls(data, x_data=None, half_window=None, lam=1e3, p=0.0, num_knots=100,
                 spline_degree=3, diff_order=2, tol=None, max_iter=None, weights=None,
                 window_kwargs=None, **kwargs):
    """
    A penalized spline version of the morphological penalized least squares (MPLS) algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.estimate_window` and `window_kwargs`.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e3.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Anchor points
        identified by the procedure in [1]_ are given a weight of `1 - p`, and all
        other points have a weight of `p`. Default is 0.0.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    tol : float, optional, deprecated

        .. deprecated:: 1.2.0
            ``tol`` is deprecated since it was not used within pspline_mpls
            and will be removed in pybaselines version 1.4.0.

    max_iter : int, optional, deprecated

        .. deprecated:: 1.2.0
            ``max_iter`` is deprecated since it was not used within pspline_mpls
            and will be removed in pybaselines version 1.4.0.

    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be
        calculated following the procedure in [1]_.
    window_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.estimate_window` for
        estimating the half window if `half_window` is None. Default is None.
    **kwargs

        .. deprecated:: 1.2.0
            Passing additional keyword arguments is deprecated and will be removed in version
            1.4.0. Pass keyword arguments using `window_kwargs`.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'half_window': int
            The half window used for the morphological calculations.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    References
    ----------
    .. [1] Li, Zhong, et al. Morphological weighted penalized least squares for
        background correction. Analyst, 2013, 138, 4483-4492.

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """


@_spline_wrapper
def pspline_brpls(data, x_data=None, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                  max_iter=50, tol=1e-3, max_iter_2=50, tol_2=1e-3, weights=None):
    """
    A penalized spline version of the BrPLS algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
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
            both threshold values and fit values. Index 0 are the tolerance values for
            the difference in the peak proportion, and indices >= 1 are the tolerance values
            for each fit. All values that were not used in fitting have values of 0. Shape J
            is 2 plus the number of iterations for the threshold to converge (related to
            `max_iter_2`, `tol_2`), and shape K is the maximum of the number of
            iterations for the threshold and the maximum number of iterations for all of
            the fits of the various threshold values (related to `max_iter` and `tol`).

    See Also
    --------
    pybaselines.whittaker.brpls

    References
    ----------
    Wang, Q., et al. Spectral baseline estimation using penalized least squares
    with weights derived from the Bayesian method. Nuclear Science and Techniques,
    2022, 140, 250-257.

    """


@_spline_wrapper
def pspline_lsrpls(data, x_data=None, lam=1e3, num_knots=100, spline_degree=3, diff_order=2,
                   max_iter=50, tol=1e-3, weights=None, alternate_weighting=False):
    """
    A penalized spline version of the LSRPLS algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e3.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
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
    alternate_weighting : bool, optional
        If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
        the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
        a prefactor term of ``exp(t)``. See the Notes section below for more details.

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
    pybaselines.whittaker.lsrpls

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
