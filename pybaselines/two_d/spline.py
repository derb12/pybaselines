# -*- coding: utf-8 -*-
"""Functions for fitting baselines using splines.

Created on April 25, 2023
@author: Donald Erb

"""


import numpy as np

from .. import _weighting
from .._nd.pls import _PLSNDMixin
from .._validation import _check_spline_degree
from ..results import PSplineResult2D
from ..utils import _masked_matvec, relative_difference
from ._algorithm_setup import _Algorithm2D
from ._whittaker_utils import PenalizedSystem2D


class _Spline(_Algorithm2D, _PLSNDMixin):
    """A base class for all spline algorithms."""

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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

        References
        ----------
        de Rooi, J., et al. Mixture models for baseline estimation. Chemometric and
        Intelligent Laboratory Systems, 2012, 117, 56-60.

        """
        _check_spline_degree(spline_degree)
        return super()._mixture_model(
            data, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            symmetric=symmetric
        )

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
            residual each iteration multiplied by 1e-4.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._asls(
            data, lam=lam, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

    @_Algorithm2D._handle_io(sort_keys=('weights',), mask_support=1)
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
            than the baseline will be given ``p**2`` weight, and values less than the baseline
            will be given ``(1 - p)**2`` weight. Default is 1e-2.
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

                .. versionchanged:: 1.3.0
                    Prior to version 1.3.0, the returned weights were the non-squared
                    values (ie. ``p`` or ``1 - p``).

            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1 or if `diff_order` is less than 2.

        See Also
        --------
        Baseline2D.iasls

        Notes
        -----
        Although both ``pspline_iasls`` and :meth:`~.Baseline2D.pspline_asls` use `p` for defining
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

        if weights is None:
            _, _, pseudo_inverse = self._setup_polynomial(
                data, weights=None, poly_order=2, calc_vander=True, calc_pinv=True
            )
            baseline = self._polynomial.vandermonde @ (pseudo_inverse @ data.ravel())
            weights = _weighting._iasls(data - baseline.reshape(self._shape), p=p, mask=self.mask)

        y, weight_array, pspline = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        # B.T @ P_1 @ B and B.T @ P_1 @ y
        penalized_system_1 = PenalizedSystem2D(self._shape, lam_1, diff_order=1)
        d1_penalty = pspline.basis.basis.T @ penalized_system_1.penalty
        if self.mask is None:
            partial_rhs = d1_penalty @ y.ravel()
        else:
            partial_rhs = _masked_matvec(d1_penalty, y.ravel(), self.mask.ravel())

        d1_penalty = d1_penalty @ pspline.basis.basis
        pspline.add_penalty(d1_penalty)

        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline = pspline.solve(y, weight_array, rhs_extra=partial_rhs)
            new_weights = _weighting._iasls(y - baseline, p=p, mask=self.mask)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            weight_array = new_weights

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1],
            'result': PSplineResult2D(pspline, weight_array, rhs_extra=d1_penalty),
            'success': success
        }

        return baseline, params

    def pspline_airpls(self, data, lam=1e3, num_knots=25, spline_degree=3,
                       diff_order=2, max_iter=50, tol=1e-3, weights=None,
                       normalize_weights='deprecated'):
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
            If True, will normalize the computed weights between 0 and 1 to potentially
            improve the numerical stability. Default behavior uses the reference implementation,
            which sets weights for all negative residuals to be greater than 1.

            .. deprecated:: 1.3
                `normalize_weights` is deprecated and will be removed in version 1.5. The
                future behavior will use the reference implementation.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._airpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            normalize_weights=normalize_weights
        )

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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._arpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._iarpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._psalsa(
            data, lam=lam, p=p, k=k, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots
        )

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
        max_iter_2 : int, optional
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
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
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
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

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
        _check_spline_degree(spline_degree)
        return super()._brpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            max_iter_2=max_iter_2, tol_2=tol_2, weights=weights,
            spline_degree=spline_degree, num_knots=num_knots
        )

    def pspline_lsrpls(self, data, lam=1e3, num_knots=25, spline_degree=3, diff_order=2,
                       max_iter=50, tol=1e-3, weights=None, alternate_weighting=False):
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
        alternate_weighting : bool, optional
            If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
            the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
            a prefactor term of ``exp(t)``. See the Notes section below for more details.

            .. versionadded:: 1.3.0

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'result': PSplineResult2D
                An object that can use the results of the fit to perform additional
                calculations.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

        See Also
        --------
        Baseline2D.lsrpls

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
        .. [3] Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
            Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        _check_spline_degree(spline_degree)
        return super()._lsrpls(
            data, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            weights=weights, spline_degree=spline_degree, num_knots=num_knots,
            alternate_weighting=alternate_weighting
        )
