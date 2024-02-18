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
from ._banded_utils import _add_diagonals, _shift_rows, diff_penalty_diagonals
from ._compat import dia_object, jit, trapezoid
from ._spline_utils import _basis_midpoints
from ._validation import _check_lam, _check_optional_array
from .utils import (
    ParameterWarning, _mollifier_kernel, _sort_array, gaussian, pad_edges, padded_convolve,
    relative_difference
)


class _Spline(_Algorithm):
    """A base class for all spline algorithms."""

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
    def mixture_model(self, data, lam=1e5, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                      max_iter=50, tol=1e-3, weights=None, symmetric=False, num_bins=None):
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
            will be given `p - 1` weight. Used to set the initial weights before performing
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
        num_bins : int, optional, deprecated

            .. deprecated:: 1.1.0
                ``num_bins`` is deprecated since it is no longer necessary for performing
                the expectation-maximization and will be removed in pybaselines version 1.3.0.

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
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        if num_bins is not None:
            warnings.warn(
                '"num_bins" was deprecated in version 1.1.0 and will be removed in version 1.3.0',
                DeprecationWarning, stacklevel=2
            )

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        # scale y between -1 and 1 so that the residual fit is more numerically stable
        # TODO is this still necessary now that expectation-maximization is used? -> still
        # helps to prevent overflows when using gaussian
        y_domain = np.polynomial.polyutils.getdomain(y)
        y = np.polynomial.polyutils.mapdomain(y, y_domain, np.array([-1., 1.]))

        if weights is not None:
            baseline = self.pspline.solve_pspline(y, weight_array)
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
                baseline = self.pspline.solve_pspline(y, weight_array)
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
            positive_pdf = (
                fraction_positive * (residual >= 0) * (1 / max(abs(residual.max()), 1e-6))
            )
            noise_pdf = (
                fraction_noise * gaussian(residual, 1 / (sigma * np.sqrt(2 * np.pi)), 0, sigma)
            )
            total_pdf = noise_pdf + positive_pdf
            if symmetric:
                negative_pdf = (
                    (1 - fraction_noise - fraction_positive)
                    * (residual < 0) * (1 / max(abs(residual.min()), 1e-6))
                )
                total_pdf += negative_pdf
            posterior_prob_noise = noise_pdf / total_pdf

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
            baseline = self.pspline.solve_pspline(y, weight_array)
            residual = y - baseline

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 1]
        }

        baseline = np.polynomial.polyutils.mapdomain(baseline, np.array([-1., 1.]), y_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
        if not 0 < quantile < 1:
            raise ValueError('quantile must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        old_coef = np.zeros(self.pspline._num_bases)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array)
            calc_difference = relative_difference(old_coef, self.pspline.coef)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            old_coef = self.pspline.coef
            weight_array = _weighting._quantile(y, baseline, quantile, eps)

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register
    def corner_cutting(self, data, max_iter=100):
        """
        Iteratively removes corner points and creates a Bezier spline from the remaining points.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        max_iter : int, optional
            The maximum number of iterations to try to remove corner points. Default is
            100. Typically all corner points are removed in 10 to 20 iterations.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            An empty dictionary, just to match the output of all other algorithms.

        References
        ----------
        Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline
        Construction. Analyst, 2015, 140(23), 7984-7996.

        """
        y, mask = self._setup_spline(data, make_basis=False)
        mask = mask.astype(bool, copy=False)

        areas = np.zeros(max_iter)
        kept_points = np.zeros(self._len, int)
        old_area = trapezoid(y, self.x)
        old_sum = self._len
        ym = y
        xm = self.x
        for i in range(max_iter):
            new_mask = mask[mask]
            new_mask[1:-1] = (
                ym[1:-1] < ym[:-2] + (xm[1:-1] - xm[:-2]) * (ym[2:] - ym[:-2]) / (xm[2:] - xm[:-2])
            )
            mask[mask] = new_mask

            new_sum = mask.sum()
            num_corners = old_sum - new_sum
            if num_corners == 0:
                i -= 1  # subtract 1 so that areas is correctly indexed
                break
            old_sum = new_sum

            kept_points[mask] += 1

            xm = self.x[mask]
            ym = y[mask]

            # TODO area calculation does not match reference values; need to recheck
            # and figure out the correct criteria
            # area = (
            #     (xm[1:-1] - xm[:-2]) * (ym[1:-1] + ym[:-2])
            #     + (xm[2:] - xm[1:-1]) * (ym[2:] + ym[1:-1])
            #     - (xm[2:] - xm[:-2]) * (ym[2:] + ym[:-2])
            # ).sum()
            area = trapezoid(ym, xm)
            areas[i] = (old_area - area) / num_corners
            old_area = area

        areas = areas[:i + 1]

        max_area = np.argmax(areas) - 1  # include points before largest area loss
        mask = kept_points >= max_area

        baseline = _quadratic_bezier_spline(self.x, y, np.flatnonzero(mask))

        # TODO maybe return areas and kept_points so that users can decide to use a
        # different iteration to build the spline; need to figure out area calculation
        # first, and then decide what to return; if so, need to make the bezier spline
        # function public and do input validation
        return baseline, {}

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
            will be given `p - 1` weight. Default is 1e-2.
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
        Leiden University Medical Centre Report, 2005, 1(1).

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 1e-2.
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
        Baseline.iasls

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
            baseline = self.vandermonde @ (pseudo_inverse @ data)
            weights = _weighting._asls(data, baseline, p)

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )

        # B.T @ D_1.T @ D_1 @ B and B.T @ D_1.T @ D_1 @ y
        d1_penalty = _check_lam(lam_1) * diff_penalty_diagonals(self._len, 1, lower_only=False)
        d1_penalty = (
            self.pspline.basis.T
            @ dia_object((d1_penalty, np.array([1, 0, -1])), shape=(self._len, self._len)).tocsr()
        )
        partial_rhs = d1_penalty @ y
        # now change d1_penalty back to banded array
        d1_penalty = (d1_penalty @ self.pspline.basis).todia().data[::-1]
        if self.pspline.lower:
            d1_penalty = d1_penalty[len(d1_penalty) // 2:]
        self.pspline.add_penalty(d1_penalty)

        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array**2, rhs_extra=partial_rhs)
            new_weights = _weighting._asls(y, baseline, p)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
    def pspline_airpls(self, data, lam=1e3, num_knots=100, spline_degree=3,
                       diff_order=2, max_iter=50, tol=1e-3, weights=None):
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
        Baseline.airpls

        References
        ----------
        Zhang, Z.M., et al. Baseline correction using adaptive iteratively
        reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam, copy_weights=True
        )

        y_l1_norm = np.abs(y).sum()
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            try:
                output = self.pspline.solve_pspline(y, weight_array)
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

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
        Baseline.arpls

        References
        ----------
        Baek, S.J., et al. Baseline correction using asymmetrically reweighted
        penalized least squares smoothing. Analyst, 2015, 140, 250-257.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array)
            new_weights = _weighting._arpls(y, baseline)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam,
            allow_lower=False, reverse_diags=False
        )
        # B.T @ W @ B + P_1 + (I - eta * W) @ P_n -> B.T @ W @ B + P_1 + P_n - eta * W @ P_n
        # reverse P_n for the eta * W @ P_n calculation to match the original diagonal
        # structure of the sparse matrix
        diff_n_diagonals = -eta * self.pspline.penalty[::-1]
        penalty_bands = self.pspline.num_bands
        self.pspline.add_penalty(diff_penalty_diagonals(self.pspline._num_bases, 1, False))

        interp_pts = _basis_midpoints(self.pspline.knots, self.pspline.spline_degree)
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            diff_n_w_diagonals = _shift_rows(
                diff_n_diagonals * np.interp(interp_pts, self.x, weight_array),
                penalty_bands, penalty_bands
            )
            baseline = self.pspline.solve_pspline(
                y, weight_array,
                penalty=_add_diagonals(self.pspline.penalty, diff_n_w_diagonals, lower_only=False)
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

        params = {'weights': weight_array, 'tol_history': tol_history[:i]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
        Baseline.iarpls

        References
        ----------
        Ye, J., et al. Baseline correction method based on improved asymmetrically
        reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
        59, 10933-10943.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        tol_history = np.empty(max_iter + 1)
        for i in range(1, max_iter + 2):
            baseline = self.pspline.solve_pspline(y, weight_array)
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

    @_Algorithm._register(sort_keys=('weights', 'alpha'), dtype=float, order='C')
    def pspline_aspls(self, data, lam=1e4, num_knots=100, spline_degree=3, diff_order=2,
                      max_iter=100, tol=1e-3, weights=None, alpha=None):
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

        See Also
        --------
        Baseline.aspls

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

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam,
            allow_lower=False, reverse_diags=True
        )
        alpha_array = _check_optional_array(
            self._len, alpha, check_finite=self._check_finite, name='alpha'
        )
        if self._sort_order is not None and alpha is not None:
            alpha_array = alpha_array[self._sort_order]

        interp_pts = _basis_midpoints(self.pspline.knots, self.pspline.spline_degree)
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            # convert alpha_array from len(y) to basis.shape[1]
            alpha_penalty = _shift_rows(
                self.pspline.penalty * np.interp(interp_pts, self.x, alpha_array),
                self.pspline.num_bands, self.pspline.num_bands
            )
            baseline = self.pspline.solve_pspline(y, weight_array, alpha_penalty)
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

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
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
            will be given `p - 1` weight. Default is 0.5.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~Baseline.asls`.
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

        Raises
        ------
        ValueError
            Raised if `p` is not between 0 and 1.

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
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        if k is None:
            k = np.std(y) / 10
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array)
            new_weights = _weighting._psalsa(y, baseline, p, k, self._len)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
    def pspline_derpsalsa(self, data, lam=1e2, p=1e-2, k=None, num_knots=100, spline_degree=3,
                          diff_order=2, max_iter=50, tol=1e-3, weights=None,
                          smooth_half_window=None, num_smooths=16, **pad_kwargs):
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
            will be given `p - 1` weight. Default is 1e-2.
        k : float, optional
            A factor that controls the exponential decay of the weights for baseline
            values greater than the data. Should be approximately the height at which
            a value could be considered a peak. Default is None, which sets `k` to
            one-tenth of the standard deviation of the input data. A large k value
            will produce similar results to :meth:`~Baseline.asls`.
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
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing.

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
        Baseline.derpsalsa

        References
        ----------
        Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
        automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
        51(10), 2061-2065.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 < p < 1:
            raise ValueError('p must be between 0 and 1')
        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam
        )
        if k is None:
            k = np.std(y) / 10

        if smooth_half_window is None:
            smooth_half_window = self._len // 200
        # could pad the data every iteration, but it is ~2-3 times slower and only affects
        # the edges, so it's not worth it
        y_smooth = pad_edges(y, smooth_half_window, **pad_kwargs)
        if smooth_half_window > 0:
            smooth_kernel = _mollifier_kernel(smooth_half_window)
            for _ in range(num_smooths):
                y_smooth = padded_convolve(y_smooth, smooth_kernel)
        y_smooth = y_smooth[smooth_half_window:self._len + smooth_half_window]

        diff_y_1 = np.gradient(y_smooth)
        diff_y_2 = np.gradient(diff_y_1)
        # x.dot(x) is same as (x**2).sum() but faster
        rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / self._len)
        rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / self._len)

        diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
        diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
        partial_weights = diff_1_weights * diff_2_weights
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline = self.pspline.solve_pspline(y, weight_array)
            new_weights = _weighting._derpsalsa(y, baseline, p, k, self._len, partial_weights)
            calc_difference = relative_difference(weight_array, new_weights)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            weight_array = new_weights

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def pspline_mpls(self, data, half_window=None, lam=1e3, p=0.0, num_knots=100, spline_degree=3,
                     diff_order=2, tol=1e-3, max_iter=50, weights=None, **window_kwargs):
        """
        A penalized spline version of the morphological penalized least squares (MPLS) algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e3.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Anchor points
            identified by the procedure in [32]_ are given a weight of `1 - p`, and all
            other points have a weight of `p`. Default is 0.0.
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
            The weighting array. If None (default), then the weights will be
            calculated following the procedure in [32]_.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

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

        See Also
        --------
        Baseline.mpls

        References
        ----------
        .. [32] Li, Zhong, et al. Morphological weighted penalized least squares for
            background correction. Analyst, 2013, 138, 4483-4492.

        Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
        Reviews: Computational Statistics, 2010, 2(6), 637-653.

        """
        if not 0 <= p <= 1:
            raise ValueError('p must be between 0 and 1')

        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
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
            w = np.full(y.shape[0], p)
            # find the index of min(y) in the region between flat regions
            for previous_segment, next_segment in zip(indices[1::2], indices[2::2]):
                index = np.argmin(y[previous_segment:next_segment + 1]) + previous_segment
                w[index] = 1 - p

            # have to invert the weight ordering the matching the original input y ordering
            # since it will be sorted within _setup_spline
            w = _sort_array(w, self._inverted_order)

        _, weight_array = self._setup_spline(y, w, spline_degree, num_knots, True, diff_order, lam)
        baseline = self.pspline.solve_pspline(y, weight_array)

        params = {'weights': weight_array, 'half_window': half_wind}
        return baseline, params


_spline_wrapper = _class_wrapper(_Spline)


@_spline_wrapper
def mixture_model(data, lam=1e5, p=1e-2, num_knots=100, spline_degree=3, diff_order=3,
                  max_iter=50, tol=1e-3, weights=None, symmetric=False, num_bins=None,
                  x_data=None):
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
        will be given `p - 1` weight. Used to set the initial weights before performing
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
    num_bins : int, optional, deprecated

        .. deprecated:: 1.1.0
            ``num_bins`` is deprecated since it is no longer necessary for performing
            the expectation-maximization and will be removed in pybaselines version 1.3.0.

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


@jit(nopython=True, cache=True)
def _quadratic_bezier(y_points, t):
    """
    Makes a single quadratic Bezier curve weighted by y-values.

    Parameters
    ----------
    y_points : Container
        A container of the three y-values that define the y-values of the
        Bezier curve.
    t : numpy.ndarray, shape (N,)
        The array of values between 0 and 1 defining the Bezier curve.

    Returns
    -------
    output : numpy.ndarray, shape (N,)
        The Bezier curve for `t`, using the three points in `y_points` as weights
        in order to shift the curve to match the desired y-values.

    References
    ----------
    https://pomax.github.io/bezierinfo (if the link is dead, the GitHub repo for the
    website is https://github.com/Pomax/BezierInfo-2).

    """
    one_minus_t = 1 - t
    output = (
        y_points[0] * one_minus_t**2 + y_points[1] * 2 * one_minus_t * t + y_points[2] * t**2
    )
    return output


@jit(nopython=True, cache=True)
def _quadratic_bezier_spline(x, y, indices):
    """
    Creates a spline from multiple quadratic Bezier curves.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values for the spline.
    y : numpy.ndarray, shape (N,)
        The y-values for the spline.
    indices : numpy.ndarray, shape (M,)
        An array of the indices of the control points for the Bezier
        spline within `x` and `y`.

    Returns
    -------
    output : numpy.ndarray, shape (N,)
        The curve constructed from the control points.

    Raises
    ------
    ValueError
        Raised if the number of indices, `M`, is less than 2, or if `x` and `y` do not
        have the same number of points.

    Notes
    -----
    If `M` is 2, then the output is a linear interpolation. If `M` is 3, the output is
    a single Bezier curve using the three control points. Otherwise, a Bezier spline
    is constructed by inserting additional control points between each index in
    `indices[2:-2]` to join the individual Bezier curves, following the reference.

    The resulting spline is only guaranteed to be C0 continuous (ie. no discontinuities).
    Any derivatives are not guaranteed to be continuous.

    References
    ----------
    Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline
    Construction. Analyst, 2015, 140(23), 7984-7996.

    """
    num_indices = indices.shape[0]
    num_x = x.shape[0]
    if num_x != y.shape[0]:
        raise ValueError('x and y must have the same number of points')
    if num_indices < 2:
        raise ValueError('indices must have at least two points for a Bezier curve')
    elif num_indices < 4:
        left_idx = indices[0]
        right_idx = indices[-1]
        left_x = x[left_idx]
        right_x = x[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        if num_indices == 2:  # perform linear interpolation
            output = left_y + (x - left_x) * (right_y - left_y) / (right_x - left_x)
        else:  # create a single Bezier curve from the three points
            center_y = y[indices[1]]
            y_points = [left_y, center_y, right_y]
            t = (x - left_x) / (right_x - left_x)
            output = _quadratic_bezier(y_points, t)

        return output

    output = np.empty(num_x)

    # first segment uses first two indices and an added halfway-point
    center_idx = indices[1]
    next_idx = indices[2]
    left_x = x[indices[0]]
    center_x = x[center_idx]
    right_idx = (
        center_idx + np.argmin(np.abs(x[center_idx:next_idx + 1] - 0.5 * (center_x + x[next_idx])))
    )
    right_x = x[right_idx]

    left_y = y[indices[0]]
    center_y = y[center_idx]
    right_y = center_y + (right_x - center_x) * (y[next_idx] - center_y) / (x[next_idx] - center_x)
    y_points = [left_y, center_y, right_y]
    t = (x[:next_idx + 1] - left_x) / (right_x - left_x)
    output[:next_idx + 1] = _quadratic_bezier(y_points, t)

    for i, center_idx in enumerate(indices[2:-2], 2):
        left_idx = right_idx
        left_x = right_x
        next_idx = indices[i + 1]
        center_x = x[center_idx]
        right_idx = (
            center_idx
            + np.argmin(np.abs(x[center_idx:next_idx + 1] - 0.5 * (center_x + x[next_idx])))
        )
        right_x = x[right_idx]
        if right_x - left_x == 0:
            continue

        left_y = right_y
        center_y = y[center_idx]
        right_y = (
            center_y + (right_x - center_x) * (y[next_idx] - center_y) / (x[next_idx] - center_x)
        )
        y_points = [left_y, center_y, right_y]

        t = (x[left_idx:right_idx + 1] - left_x) / (right_x - left_x)
        output[left_idx:right_idx + 1] = _quadratic_bezier(y_points, t)

    # last segment uses last two indices and the last added halfway-point
    y_points = [right_y, y[indices[-2]], y[indices[-1]]]
    t = (x[right_idx:] - right_x) / (x[indices[-1]] - right_x)
    output[right_idx:] = _quadratic_bezier(y_points, t)

    return output


@_spline_wrapper
def corner_cutting(data, x_data=None, max_iter=100):
    """
    Iteratively removes corner points and creates a Bezier spline from the remaining points.

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
    dict
        An empty dictionary, just to match the output of all other algorithms.

    References
    ----------
    Liu, Y.J., et al. A Concise Iterative Method with Bezier Technique for Baseline
    Construction. Analyst, 2015, 140(23), 7984-7996.

    """


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
        will be given `p - 1` weight. Default is 1e-2.
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
    Leiden University Medical Centre Report, 2005, 1(1).

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
        will be given `p - 1` weight. Default is 1e-2.
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
                   max_iter=50, tol=1e-3, weights=None, x_data=None):
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
                  max_iter=100, tol=1e-3, weights=None, alpha=None, x_data=None):
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

    See Also
    --------
    pybaselines.whittaker.aspls

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

    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
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
        will be given `p - 1` weight. Default is 0.5.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :meth:`~Baseline.asls`.
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
                      smooth_half_window=None, num_smooths=16, x_data=None, **pad_kwargs):
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
        will be given `p - 1` weight. Default is 1e-2.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :meth:`~Baseline.asls`.
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
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from smoothing.

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
                 spline_degree=3, diff_order=2, tol=1e-3, max_iter=50, weights=None,
                 **window_kwargs):
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
        half-window size using :func:`.optimize_window` and `window_kwargs`.
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
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be
        calculated following the procedure in [1]_.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

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
