# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on Feb. 27, 2021
@author: Donald Erb


The function penalized_poly was adapted from MATLAB code from
https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction
(accessed March 18, 2021), which was licensed under the BSD-2-clause below.

License: 2-clause BSD

Copyright (c) 2012, Vincent Mazet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


The function loess was adapted from code from https://gist.github.com/agramfort/850437
(accessed March 25, 2021), which was licensed under the BSD-3-clause below.

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)
Copyright (c) 2015, Alexandre Gramfort
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from math import ceil
import warnings

import numpy as np

from . import _weighting
from ._algorithm_setup import _Algorithm, _class_wrapper
from ._compat import jit, prange
from .utils import _MIN_FLOAT, ParameterWarning, _convert_coef, _interp_inplace, relative_difference


class _Polynomial(_Algorithm):
    """A base class for all polynomial algorithms."""

    @_Algorithm._register(sort_keys=('weights',))
    def poly(self, data, poly_order=2, weights=None, return_coef=False):
        """
        Computes a polynomial that fits the baseline of the data.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'coef': numpy.ndarray, shape (poly_order,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Notes
        -----
        To only fit regions without peaks, supply a weight array with zero values
        at the indices where peaks are located.

        """
        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True
        )
        sqrt_w = np.sqrt(weight_array)

        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self.vandermonde @ coef
        params = {'weights': weight_array}
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def modpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                use_original=False, mask_initial_peaks=False, return_coef=False):
        """
        The modified polynomial (ModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 with N points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [8]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [9]_.
        mask_initial_peaks : bool, optional
            If True, will mask any data where the initial baseline fit + the standard
            deviation of the residual is less than measured data [10]_. Default is False.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.

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
            * 'coef': numpy.ndarray, shape (poly_order + 1,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Notes
        -----
        Algorithm originally developed in [9]_ and then slightly modified in [8]_.

        References
        ----------
        .. [8] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [9] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [10] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True, copy_weights=True
        )
        sqrt_w = np.sqrt(weight_array)
        if use_original:
            y0 = y

        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self.vandermonde @ coef
        if mask_initial_peaks:
            # use baseline + deviation since without deviation, half of y should be above baseline
            weight_array[baseline + np.std(y - baseline) < y] = 0
            sqrt_w = np.sqrt(weight_array)
            pseudo_inverse = np.linalg.pinv(sqrt_w[:, None] * self.vandermonde)

        tol_history = np.empty(max_iter)
        for i in range(max_iter):
            baseline_old = baseline
            y = np.minimum(y0 if use_original else y, baseline)
            coef = pseudo_inverse @ (sqrt_w * y)
            baseline = self.vandermonde @ coef
            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def imodpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                 use_original=False, mask_initial_peaks=True, return_coef=False, num_std=1.):
        """
        The improved modofied polynomial (IModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [11]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [12]_.
        mask_initial_peaks : bool, optional
            If True (default), will mask any data where the initial baseline fit +
            the standard deviation of the residual is less than measured data [13]_.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Default
            is 1. Must be greater or equal to 0.

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
            * 'coef': numpy.ndarray, shape (poly_order + 1,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Raises
        ------
        ValueError
            Raised if `num_std` is less than 0.

        Notes
        -----
        Algorithm originally developed in [13]_.

        References
        ----------
        .. [11] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [12] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [13] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        if num_std < 0:
            raise ValueError('num_std must be greater than or equal to 0')

        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True, copy_weights=True
        )
        sqrt_w = np.sqrt(weight_array)
        if use_original:
            y0 = y

        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self.vandermonde @ coef
        deviation = np.std(y - baseline)
        if mask_initial_peaks:
            weight_array[baseline + deviation < y] = 0
            sqrt_w = np.sqrt(weight_array)
            pseudo_inverse = np.linalg.pinv(sqrt_w[:, None] * self.vandermonde)

        tol_history = np.empty(max_iter)
        for i in range(max_iter):
            y = np.minimum(y0 if use_original else y, baseline + num_std * deviation)
            coef = pseudo_inverse @ (sqrt_w * y)
            baseline = self.vandermonde @ coef
            new_deviation = np.std(y - baseline)
            # use new_deviation as dividing term in relative difference
            calc_difference = relative_difference(new_deviation, deviation)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            deviation = new_deviation

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    # adapted from
    # https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction;
    # see license above
    @_Algorithm._register(sort_keys=('weights',))
    def penalized_poly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                       cost_function='asymmetric_truncated_quadratic', threshold=None,
                       alpha_factor=0.99, return_coef=False):
        """
        Fits a polynomial baseline using a non-quadratic cost function.

        The non-quadratic cost functions penalize residuals with larger values,
        giving a more robust fit compared to normal least-squares.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        cost_function : str, optional
            The non-quadratic cost function to minimize. Must indicate symmetry of the
            method by appending 'a' or 'asymmetric' for asymmetric loss, and 's' or
            'symmetric' for symmetric loss. Default is 'asymmetric_truncated_quadratic'.
            Available methods, and their associated reference, are:

                * 'asymmetric_truncated_quadratic'[14]_
                * 'symmetric_truncated_quadratic'[14]_
                * 'asymmetric_huber'[14]_
                * 'symmetric_huber'[14]_
                * 'asymmetric_indec'[15]_
                * 'symmetric_indec'[15]_

        threshold : float, optional
            The threshold value for the loss method, where the function goes from
            quadratic loss (such as used for least squares) to non-quadratic. For
            symmetric loss methods, residual values with absolute value less than
            threshold will have quadratic loss. For asymmetric loss methods, residual
            values less than the threshold will have quadratic loss. Default is None,
            which sets `threshold` to one-tenth of the standard deviation of the input
            data.
        alpha_factor : float, optional
            A value between 0 and 1 that controls the value of the penalty. Default is
            0.99. Typically should not need to change this value.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.

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
            * 'coef': numpy.ndarray, shape (poly_order + 1,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Raises
        ------
        ValueError
            Raised if `alpha_factor` is not between 0 and 1.

        Notes
        -----
        In baseline literature, this procedure is sometimes called "backcor".

        References
        ----------
        .. [14] Mazet, V., et al. Background removal from spectra by designing and
            minimising a non-quadratic cost function. Chemometrics and Intelligent
            Laboratory Systems, 2005, 76(2), 121-133.
        .. [15] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
            Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

        """
        if not 0 < alpha_factor <= 1:
            raise ValueError('alpha_factor must be between 0 and 1')
        symmetric_loss, method = _identify_loss_method(cost_function)
        loss_function = {
            'huber': _huber_loss,
            'truncated_quadratic': _truncated_quadratic_loss,
            'indec': _indec_loss
        }[method]

        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True
        )
        if threshold is None:
            threshold = np.std(y) / 10
        loss_kwargs = {
            'threshold': threshold, 'alpha_factor': alpha_factor, 'symmetric': symmetric_loss
        }

        sqrt_w = np.sqrt(weight_array)
        y = sqrt_w * y

        coef = pseudo_inverse @ y
        baseline = self.vandermonde @ coef
        tol_history = np.empty(max_iter)
        for i in range(max_iter):
            baseline_old = baseline
            coef = pseudo_inverse @ (y + loss_function(y - sqrt_w * baseline, **loss_kwargs))
            baseline = self.vandermonde @ coef
            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

        params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('weights', 'coef'))
    def loess(self, data, fraction=0.2, total_points=None, poly_order=1, scale=3.0,
              tol=1e-3, max_iter=10, symmetric_weights=False, use_threshold=False,
              num_std=1, use_original=False, weights=None, return_coef=False,
              conserve_memory=True, delta=0.0):
        """
        Locally estimated scatterplot smoothing (LOESS).

        Performs polynomial regression at each data point using the nearest points.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        fraction : float, optional
            The fraction of N data points to include for the fitting on each point.
            Default is 0.2. Not used if `total_points` is not None.
        total_points : int, optional
            The total number of points to include for the fitting on each point. Default
            is None, which will use `fraction` * N to determine the number of points.
        scale : float, optional
            A scale factor applied to the weighted residuals to control the robustness
            of the fit. Default is 3.0, as used in [16]_. Note that the original loess
            procedure in [17]_ used a `scale` of ~4.05.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 1.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 10.
        symmetric_weights : bool, optional
            If False (default), will apply weighting asymmetrically, with residuals
            < 0 having a weight of 1, according to [16]_. If True, will apply weighting
            the same for both positive and negative residuals, which is regular LOESS.
            If `use_threshold` is True, this parameter is ignored.
        use_threshold : bool, optional
            If False (default), will compute weights each iteration to perform the
            robust fitting, which is regular LOESS. If True, will apply a threshold
            on the data being fit each iteration, based on the maximum values of the
            data and the fit baseline, as proposed by [18]_, similar to the modpoly
            and imodpoly techniques.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Default
            is 1, which is the value used for the imodpoly technique. Only used if
            `use_threshold` is True.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [19]_ when choosing minimum values for
            thresholding. If True, will compare the baseline with the original
            y-values given by `data` [20]_. Only used if `use_threshold` is True.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.
        conserve_memory : bool, optional
            If False, will cache the distance-weighted kernels for each value
            in `x_data` on the first iteration and reuse them on subsequent iterations to
            save time. The shape of the array of kernels is (len(`x_data`), `total_points`).
            If True (default), will recalculate the kernels each iteration, which uses very
            little memory, but is slower. Can usually set to False unless `x_data` and`total_points`
            are quite large and the function causes memory issues when cacheing the kernels. If
            numba is installed, there is no significant time difference since the calculations are
            sped up.
        delta : float, optional
            If `delta` is > 0, will skip all but the last x-value in the range x_last + `delta`,
            where x_last is the last x-value to be fit using weighted least squares, and instead
            use linear interpolation to calculate the fit for those x-values (same behavior as in
            statsmodels [21]_ and Cleveland's original Fortran lowess implementation [22]_).
            Fits all x-values if `delta` is <= 0. Default is 0.0. Note that `x_data` is scaled to
            fit in the range [-1, 1], so `delta` should likewise be scaled. For example, if the
            desired `delta` value was ``0.01 * (max(x_data) - min(x_data))``, then the
            correctly scaled `delta` would be 0.02 (ie. ``0.01 * (1 - (-1))``).

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data. Does NOT contain the
                individual distance-weighted kernels for each x-value.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'coef': numpy.ndarray, shape (N, poly_order + 1)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a polynomial
                using :class:`numpy.polynomial.polynomial.Polynomial`. If `delta` is > 0,
                the coefficients for any skipped x-value will all be 0.

        Raises
        ------
        ValueError
            Raised if the number of points per window for the fitting is less than
            `poly_order` + 1 or greater than the total number of points, or if the
            values in `self.x` are not strictly increasing.

        Notes
        -----
        The iterative, robust, aspect of the fitting can be achieved either through
        reweighting based on the residuals (the typical usage), or thresholding the
        fit data based on the residuals, as proposed by [18]_, similar to the modpoly
        and imodpoly techniques.

        In baseline literature, this procedure is sometimes called "rbe", meaning
        "robust baseline estimate".

        References
        ----------
        .. [16] Ruckstuhl, A.F., et al. Baseline subtraction using robust local
            regression estimation. J. Quantitative Spectroscopy and Radiative
            Transfer, 2001, 68, 179-193.
        .. [17] Cleveland, W. Robust locally weighted regression and smoothing
                scatterplots. Journal of the American Statistical Association,
                1979, 74(368), 829-836.
        .. [18] Komsta, Ł. Comparison of Several Methods of Chromatographic
                Baseline Removal with a New Approach Based on Quantile Regression.
                Chromatographia, 2011, 73, 721-731.
        .. [19] Gan, F., et al. Baseline correction by improved iterative polynomial
                fitting with automatic threshold. Chemometrics and Intelligent
                Laboratory Systems, 2006, 82, 59-65.
        .. [20] Lieber, C., et al. Automated method for subtraction of fluorescence
                from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
                1363-1367.
        .. [21] https://github.com/statsmodels/statsmodels.
        .. [22] https://www.netlib.org/go (lowess.f is the file).

        """
        if np.any(self.x[1:] < self.x[:-1]):
            raise ValueError('x must be strictly increasing')

        if total_points is None:
            total_points = ceil(fraction * self._len)
        if total_points < poly_order + 1:
            raise ValueError('total points must be greater than polynomial order + 1')
        elif total_points > self._len:
            raise ValueError((
                'points per window is higher than total number of points; lower either '
                '"fraction" or "total_points"'
            ))
        elif poly_order > 2:
            warnings.warn(
                ('polynomial orders greater than 2 can have numerical issues;'
                 ' consider using a polynomial order of 1 or 2 instead'),
                ParameterWarning, stacklevel=2
            )

        y, weight_array = self._setup_polynomial(data, weights, poly_order, calc_vander=True)
        if use_original:
            y0 = y

        # x is the scaled version of self.x to fit within the [-1, 1] domain
        x = np.polynomial.polyutils.mapdomain(self.x, self.x_domain, np.array([-1., 1.]))
        # find the indices for fitting beforehand so that the fitting can be done
        # in parallel; cast delta as float so numba does not have to compile for
        # both int and float
        windows, fits, skips = _determine_fits(x, self._len, total_points, float(delta))

        # np.polynomial.polynomial.polyvander returns a Fortran-ordered array, which
        # when matrix multiplied with the C-ordered coefficient array gives a warning
        # when using numba, so convert Vandermonde matrix to C-ordering.
        self.vandermonde = np.ascontiguousarray(self.vandermonde)

        baseline = y
        coefs = np.zeros((self._len, poly_order + 1))
        tol_history = np.empty(max_iter + 1)
        sqrt_w = np.sqrt(weight_array)
        # do max_iter + 1 since a max_iter of 0 would return y as baseline otherwise
        for i in range(max_iter + 1):
            baseline_old = baseline
            if conserve_memory:
                baseline = _loess_low_memory(
                    x, y, sqrt_w, coefs, self.vandermonde, self._len, windows, fits
                )
            elif i == 0:
                kernels, baseline = _loess_first_loop(
                    x, y, sqrt_w, coefs, self.vandermonde, total_points, self._len, windows, fits
                )
            else:
                baseline = _loess_nonfirst_loops(
                    y, sqrt_w, coefs, self.vandermonde, kernels, windows, self._len, fits
                )

            _fill_skips(x, baseline, skips)

            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

            if use_threshold:
                y = np.minimum(
                    y0 if use_original else y, baseline + num_std * np.std(y - baseline)
                )
            else:
                residual = y - baseline
                # TODO median_absolute_value can be 0 if more than half of residuals are
                # 0 (perfect fit); can that ever really happen? if so, should prevent dividing by 0
                sqrt_w = _tukey_square(
                    residual / _median_absolute_value(residual), scale, symmetric_weights
                )

        params = {'weights': sqrt_w**2, 'tol_history': tol_history[:i + 1]}
        if return_coef:
            # TODO maybe leave out the coefficients from the rest of the calculations
            # since they are otherwise unused, and just fit x vs baseline here; would
            # save a little memory; is providing coefficients for loess even useful?
            params['coef'] = np.array([_convert_coef(coef, self.x_domain) for coef in coefs])

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def quant_reg(self, data, poly_order=2, quantile=0.05, tol=1e-6, max_iter=250,
                  weights=None, eps=None, return_coef=False):
        """
        Approximates the baseline of the data using quantile regression.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 0.05.
        tol : float, optional
            The exit criteria. Default is 1e-6. For extreme quantiles (`quantile` < 0.01
            or `quantile` > 0.99), may need to use a lower value to get a good fit.
        max_iter : int, optional
            The maximum number of iterations. Default is 250. For extreme quantiles
            (`quantile` < 0.01 or `quantile` > 0.99), may need to use a higher value to
            ensure convergence.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        eps : float, optional
            A small value added to the square of the residual to prevent dividing by 0.
            Default is None, which uses the square of the maximum-absolute-value of the
            fit each iteration multiplied by 1e-6.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input `x_data` and return them in the params dictionary.
            Default is False, since the conversion takes time.

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
            * 'coef': numpy.ndarray, shape (poly_order + 1,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Raises
        ------
        ValueError
            Raised if `quantile` is not between 0 and 1.

        Notes
        -----
        Application of quantile regression for baseline fitting ss described in [23]_.

        Performs quantile regression using iteratively reweighted least squares (IRLS)
        as described in [24]_.

        References
        ----------
        .. [23] Komsta, Ł. Comparison of Several Methods of Chromatographic
                Baseline Removal with a New Approach Based on Quantile Regression.
                Chromatographia, 2011, 73, 721-731.
        .. [24] Schnabel, S., et al. Simultaneous estimation of quantile curves using
                quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

        """
        # TODO provide a way to estimate best poly_order based on AIC like in Komsta? could be
        # useful for all polynomial methods; maybe could be an optimizer function
        if not 0 < quantile < 1:
            raise ValueError('quantile must be between 0 and 1.')

        y, weight_array = self._setup_polynomial(data, weights, poly_order, calc_vander=True)
        # estimate first iteration using least squares
        sqrt_w = np.sqrt(weight_array)
        coef = np.linalg.lstsq(self.vandermonde * sqrt_w[:, None], y * sqrt_w, None)[0]
        baseline = self.vandermonde @ coef
        tol_history = np.empty(max_iter)
        for i in range(max_iter):
            baseline_old = baseline
            sqrt_w = np.sqrt(_weighting._quantile(y, baseline, quantile, eps))
            coef = np.linalg.lstsq(self.vandermonde * sqrt_w[:, None], y * sqrt_w, None)[0]
            baseline = self.vandermonde @ coef
            # relative_difference(baseline_old, baseline, 1) gives nearly same result and
            # the l2 norm is faster to calculate, so use that instead of l1 norm
            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

        params = {'weights': sqrt_w**2, 'tol_history': tol_history[:i + 1]}
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('weights',))
    def goldindec(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                  cost_function='asymmetric_indec', peak_ratio=0.5, alpha_factor=0.99,
                  tol_2=1e-3, tol_3=1e-6, max_iter_2=100, return_coef=False):
        """
        Fits a polynomial baseline using a non-quadratic cost function.

        The non-quadratic cost functions penalize residuals with larger values,
        giving a more robust fit compared to normal least-squares.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 2.
        tol : float, optional
            The exit criteria for the fitting with a given threshold value. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations for fitting a threshold value. Default is 250.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        cost_function : str, optional
            The non-quadratic cost function to minimize. Unlike :func:`.penalized_poly`,
            this function only works with asymmetric cost functions, so the symmetry prefix
            ('a' or 'asymmetric') is optional (eg. 'indec' and 'a_indec' are the same). Default
            is 'asymmetric_indec'. Available methods, and their associated reference, are:

                * 'asymmetric_indec'[25]_
                * 'asymmetric_truncated_quadratic'[26]_
                * 'asymmetric_huber'[26]_

        peak_ratio : float, optional
            A value between 0 and 1 that designates how many points in the data belong
            to peaks. Values are valid within ~10% of the actual peak ratio. Default is 0.5.
        alpha_factor : float, optional
            A value between 0 and 1 that controls the value of the penalty. Default is
            0.99. Typically should not need to change this value.
        tol_2 : float, optional
            The exit criteria for the difference between the optimal up-down ratio (number of
            points above 0 in the residual compared to number of points below 0) and the up-down
            ratio for a given threshold value. Default is 1e-3.
        tol_3 : float, optional
            The exit criteria for the relative change in the threshold value. Default is 1e-6.
        max_iter_2 : float, optional
            The number of iterations for iterating between different threshold values.
            Default is 100.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input x_data and return them in the params dictionary.
            Default is False, since the conversion takes time.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'tol_history': numpy.ndarray, shape (J, K)
                An array containing the calculated tolerance values for each iteration
                of both threshold values and fit values. Index 0 are the tolerence values
                for the difference in up-down ratios, index 1 are the tolerance values for
                the relative change in the threshold, and indices >= 2 are the tolerance values
                for each fit. All values that were not used in fitting have values of 0. Shape J
                is 2 plus the number of iterations for the threshold to converge (related to
                `max_iter_2`, `tol_2`, `tol_3`), and shape K is the maximum of the number of
                iterations for the threshold and the maximum number of iterations for all of
                the fits of the various threshold values (related to `max_iter` and `tol`).
            * 'threshold' : float
                The optimal threshold value. Could be used in :func:`.penalized_poly`
                for fitting other similar data.
            * 'coef': numpy.ndarray, shape (poly_order + 1,)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

        Raises
        ------
        ValueError
            Raised if `alpha_factor` or `peak_ratio` are not between 0 and 1, or if the
            specified cost function is symmetric.

        References
        ----------
        .. [25] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
                Correction. Applied Spectroscopy, 2015, 69(7), 834-842.
        .. [26] Mazet, V., et al. Background removal from spectra by designing and
                minimising a non-quadratic cost function. Chemometrics and Intelligent
                Laboratory Systems, 2005, 76(2), 121-133.

        """
        if not 0 < alpha_factor <= 1:
            raise ValueError('alpha_factor must be between 0 and 1')
        elif not 0 < peak_ratio < 1:
            raise ValueError('peak_ratio must be between 0 and 1')
        try:
            symmetric_loss, method = _identify_loss_method(cost_function)
        except ValueError:  # do not require a prefix since cost must be asymmetric
            symmetric_loss, method = _identify_loss_method('a_' + cost_function)
        if symmetric_loss:
            # symmetric cost functions don't work due to how the up-down ratio vs
            # peak_ratio function was created in the reference; in theory, could simulate
            # spectra with both positive and negative peaks following the reference
            # and build another empirical function, but would likely need to also
            # add other parameters detailing the percent of positive vs negative peaks,
            # etc., so it's not worth the effort
            raise ValueError('goldindec only works for asymmetric cost functions')

        loss_function = {
            'huber': _huber_loss,
            'truncated_quadratic': _truncated_quadratic_loss,
            'indec': _indec_loss
        }[method]
        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True
        )
        up_down_ratio_goal = (
            0.7679 + 11.2358 * peak_ratio - 39.7064 * peak_ratio**2 + 92.3583 * peak_ratio**3
        )
        # TODO reference states threshold must be <= 2 for half-quadratic minimization to
        # be valid for indec cost function, and normalized y so that threshold is always <= 2;
        # however, it seems to work fine without normalization; just be aware in case errors
        # occur, may have to normalize y in both this function and penalized_poly
        sqrt_w = np.sqrt(weight_array)
        y_fit = sqrt_w * y

        coef = pseudo_inverse @ y_fit
        initial_baseline = self.vandermonde @ coef

        a = 0
        # reference used b=1, but normalized y before fitting; instead, set b as max of
        # initial residual
        b = abs((y - initial_baseline).max())
        threshold = a + 0.618 * (b - a)
        loss_kwargs = {
            'threshold': threshold, 'alpha_factor': alpha_factor,
            'symmetric': symmetric_loss
        }
        # have to use zeros rather than empty for tol_history since each inner fit may
        # have a different number of iterations
        tol_history = np.zeros((max_iter_2 + 2, max(max_iter, max_iter_2)))
        j_max = 0
        for i in range(max_iter_2):
            baseline = initial_baseline
            for j in range(max_iter):
                baseline_old = baseline
                coef = pseudo_inverse @ (
                    y_fit + loss_function(y_fit - sqrt_w * baseline, **loss_kwargs)
                )
                baseline = self.vandermonde @ coef
                calc_difference = relative_difference(baseline_old, baseline)
                tol_history[i + 2, j] = calc_difference
                if calc_difference < tol:
                    break
            if j > j_max:
                j_max = j

            up_count = (y > baseline).sum()
            up_down_ratio = up_count / max(1, self._len - up_count)
            calc_difference = up_down_ratio - up_down_ratio_goal
            tol_history[0, i] = calc_difference
            if calc_difference > tol_2:
                a = threshold
            elif calc_difference < -tol_2:
                b = threshold
            else:
                break
            threshold = a + 0.618 * (b - a)
            # this exit criteria was not stated in the reference, but the change in threshold
            # becomes zero fairly quickly, so need to also exit rather than needlessly
            # continuing to calculate with the same threshold value
            calc_difference = relative_difference(loss_kwargs['threshold'], threshold)
            tol_history[1, i] = calc_difference
            if calc_difference < tol_3:
                break
            loss_kwargs['threshold'] = threshold

        params = {
            'weights': weight_array, 'tol_history': tol_history[:i + 3, :max(i, j_max) + 1],
            'threshold': loss_kwargs['threshold']
        }
        if return_coef:
            params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params


_polynomial_wrapper = _class_wrapper(_Polynomial)


@_polynomial_wrapper
def poly(data, x_data=None, poly_order=2, weights=None, return_coef=False):
    """
    Computes a polynomial that fits the baseline of the data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'coef': numpy.ndarray, shape (poly_order,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Notes
    -----
    To only fit regions without peaks, supply a weight array with zero values
    at the indices where peaks are located.

    """


@_polynomial_wrapper
def modpoly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250, weights=None,
            use_original=False, mask_initial_peaks=False, return_coef=False):
    """
    The modified polynomial (ModPoly) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 250.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    use_original : bool, optional
        If False (default), will compare the baseline of each iteration with
        the y-values of that iteration [1]_ when choosing minimum values. If True,
        will compare the baseline with the original y-values given by `data` [2]_.
    mask_initial_peaks : bool, optional
        If True, will mask any data where the initial baseline fit + the standard
        deviation of the residual is less than measured data [3]_. Default is False.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.

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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Notes
    -----
    Algorithm originally developed in [2]_ and then slightly modified in [1]_.

    References
    ----------
    .. [1] Gan, F., et al. Baseline correction by improved iterative polynomial
           fitting with automatic threshold. Chemometrics and Intelligent
           Laboratory Systems, 2006, 82, 59-65.
    .. [2] Lieber, C., et al. Automated method for subtraction of fluorescence
           from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
           1363-1367.
    .. [3] Zhao, J., et al. Automated Autofluorescence Background Subtraction
           Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
           2007, 61(11), 1225-1232.

    """


@_polynomial_wrapper
def imodpoly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250, weights=None,
             use_original=False, mask_initial_peaks=True, return_coef=False, num_std=1):
    """
    The improved modofied polynomial (IModPoly) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 250.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    use_original : bool, optional
        If False (default), will compare the baseline of each iteration with
        the y-values of that iteration [4]_ when choosing minimum values. If True,
        will compare the baseline with the original y-values given by `data` [5]_.
    mask_initial_peaks : bool, optional
        If True (default), will mask any data where the initial baseline fit +
        the standard deviation of the residual is less than measured data [6]_.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Default
        is 1.

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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Notes
    -----
    Algorithm originally developed in [6]_.

    References
    ----------
    .. [4] Gan, F., et al. Baseline correction by improved iterative polynomial
           fitting with automatic threshold. Chemometrics and Intelligent
           Laboratory Systems, 2006, 82, 59-65.
    .. [5] Lieber, C., et al. Automated method for subtraction of fluorescence
           from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
           1363-1367.
    .. [6] Zhao, J., et al. Automated Autofluorescence Background Subtraction
           Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
           2007, 61(11), 1225-1232.

    """


# adapted from (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction);
# see license above
def _huber_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Huber non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the huber loss function, phi(x).

    References
    ----------
    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for huber is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
        weights = (
            mask * residual * (2 * alpha - 1)
            + (~mask) * 2 * alpha * threshold * np.sign(residual)
        )
    else:
        mask = (residual < threshold)
        weights = (
            mask * residual * (2 * alpha - 1)
            + (~mask) * (2 * alpha * threshold - residual)
        )
    return weights


# adapted from (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction);
# see license above
def _truncated_quadratic_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Truncated-Quadratic non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the truncated quadratic function, phi(x).

    References
    ----------
    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for truncated quadratic is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
    else:
        mask = (residual < threshold)
    return mask * residual * (2 * alpha - 1) - (~mask) * residual


def _indec_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Indec non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the Indec function, phi(x).

    References
    ----------
    Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
    Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for indec is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
        multiple = np.sign(residual)
    else:
        mask = (residual < threshold)
        # multiple=1 is same as sign(residual) since residual is always > 0
        # for asymmetric case, but this allows not doing the sign calculation
        multiple = 1
    weights = (
        mask * residual * (2 * alpha - 1)
        - (~mask) * (
            residual + alpha * multiple * threshold**3 / np.maximum(2 * residual**2, _MIN_FLOAT)
        )
    )
    return weights


def _identify_loss_method(loss_method):
    """
    Identifies the symmetry for the given loss method.

    Parameters
    ----------
    loss_method : str
        The loss method to use. Should have the symmetry identifier as
        the prefix.

    Returns
    -------
    symmetric : bool
        True if `loss_method` had 's_' or 'symmetric_' as the prefix, else False.
    str
        The input `loss_method` value without the first section that indicated
        the symmetry.

    Raises
    ------
    ValueError
        Raised if the loss method does not have the correct form.

    """
    prefix, *split_method = loss_method.lower().split('_')
    if prefix not in ('a', 's', 'asymmetric', 'symmetric') or not split_method:
        raise ValueError('must specify loss function symmetry by appending "a_" or "s_"')
    if prefix in ('a', 'asymmetric'):
        symmetric = False
    else:
        symmetric = True
    return symmetric, '_'.join(split_method)


# adapted from (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction);
# see license above
@_polynomial_wrapper
def penalized_poly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250,
                   weights=None, cost_function='asymmetric_truncated_quadratic',
                   threshold=None, alpha_factor=0.99, return_coef=False):
    """
    Fits a polynomial baseline using a non-quadratic cost function.

    The non-quadratic cost functions penalize residuals with larger values,
    giving a more robust fit compared to normal least-squares.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 250.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    cost_function : str, optional
        The non-quadratic cost function to minimize. Must indicate symmetry of the
        method by appending 'a' or 'asymmetric' for asymmetric loss, and 's' or
        'symmetric' for symmetric loss. Default is 'asymmetric_truncated_quadratic'.
        Available methods, and their associated reference, are:

            * 'asymmetric_truncated_quadratic'[7]_
            * 'symmetric_truncated_quadratic'[7]_
            * 'asymmetric_huber'[7]_
            * 'symmetric_huber'[7]_
            * 'asymmetric_indec'[8]_
            * 'symmetric_indec'[8]_

    threshold : float, optional
        The threshold value for the loss method, where the function goes from
        quadratic loss (such as used for least squares) to non-quadratic. For
        symmetric loss methods, residual values with absolute value less than
        threshold will have quadratic loss. For asymmetric loss methods, residual
        values less than the threshold will have quadratic loss. Default is None,
        which sets `threshold` to one-tenth of the standard deviation of the input
        data.
    alpha_factor : float, optional
        A value between 0 and 1 that controls the value of the penalty. Default is
        0.99. Typically should not need to change this value.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.

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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Raises
    ------
    ValueError
        Raised if `alpha_factor` is not between 0 and 1.

    Notes
    -----
    In baseline literature, this procedure is sometimes called "backcor".

    References
    ----------
    .. [7] Mazet, V., et al. Background removal from spectra by designing and
           minimising a non-quadratic cost function. Chemometrics and Intelligent
           Laboratory Systems, 2005, 76(2), 121-133.
    .. [8] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
           Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

    """


def _tukey_square(residual, scale=3, symmetric=False):
    """
    The square root of Tukey's bisquare function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array of the fit.
    scale : float, optional
        A scale factor applied to the weighted residuals to control the
        robustness of the fit. Default is 3.0.
    symmetric : bool, optional
        If False (default), will apply weighting asymmetrically, with residuals
        < 0 having full weight. If True, will apply weighting the same for both
        positive and negative residuals, which is regular LOESS.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weighting array.

    Notes
    -----
    The function is technically sqrt(Tukey's bisquare) since the outer
    power of 2 is not performed. This is intentional, so that the square
    root for weighting in least squares does not need to be done, speeding
    up the calculation.

    References
    ----------
    Ruckstuhl, A.F., et al., Baseline subtraction using robust local regression
    estimation. J. Quantitative Spectroscopy and Radiative Transfer, 2001, 68,
    179-193.

    """
    if symmetric:
        inner = residual / scale
        weights = np.maximum(0, 1 - inner * inner)
    else:
        weights = np.ones_like(residual)
        mask = residual > 0
        inner = residual[mask] / scale
        weights[mask] = np.maximum(0, 1 - inner * inner)
    return weights


def _median_absolute_value(values):
    """
    Computes the median absolute value (MAV) of an array.

    Parameters
    ----------
    values : array-like
        The array of values to use for the calculation.

    Returns
    -------
    float
        The scaled median absolute value for the input array.

    Notes
    -----
    The 1/0.6744897501960817 scale factor is to make the result comparable to the
    standard deviation of a Gaussian distribution. The divisor is obtained by
    calculating the value at which the cumulative distribution function of a Gaussian
    distribution is 0.75 (see https://en.wikipedia.org/wiki/Median_absolute_deviation),
    which can be obtained by::

        from scipy.special import ndtri
        ndtri(0.75)  # equals 0.6744897501960817

    To calculate the median absolute difference (MAD) using this function, simply do::

        _median_absolute_value(values - np.median(values))

    References
    ----------
    Ruckstuhl, A.F., et al., Baseline subtraction using robust local regression
    estimation. J. Quantitative Spectroscopy and Radiative Transfer, 2001, 68,
    179-193.

    https://en.wikipedia.org/wiki/Median_absolute_deviation.

    """
    return np.median(np.abs(values)) / 0.6744897501960817


@jit(nopython=True, cache=True)
def _loess_solver(AT, b):
    """
    Solves the equation `A x = b` given `A.T` and `b`.

    Parameters
    ----------
    AT : numpy.ndarray, shape (M, N)
        The transposed `A` matrix.
    b : numpy.ndarray, shape (N,)
        The `b` array.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The solution to the normal equation.

    Notes
    -----
    Uses np.linalg.solve (which uses LU decomposition) rather than np.linalg.lstsq
    (which uses SVD) since solve is ~30-60% faster. np.linalg.solve requires ``A.T * A``,
    which squares the condition number of ``A``, but on tested datasets the relative
    difference when using solve vs lstsq (using np.allclose) is ~1e-10 to 1e-13 for
    poly_orders of 1 or 2, which seems fine; the relative differences increase to
    ~1e-6 to 1e-9 for a poly_order of 3, and ~1e-4 to 1e-6 for a poly_order of 4, but
    loess should use a poly_order <= 2, so that should not be a problem.

    """
    return np.linalg.solve(AT.dot(AT.T), AT.dot(b))


@jit(nopython=True, cache=True, parallel=True)
def _fill_skips(x, baseline, skips):
    """
    Fills in the skipped baseline points using linear interpolation.

    Parameters
    ----------
    x : numpy.ndarray
        The array of x-values.
    baseline : numpy.ndarray
        The array of baseline values with all fit points allocated. All skipped points
        will be filled in using interpolation.
    skips : numpy.ndarray, shape (G, 2)
        The array of left and right indices that define the windows for interpolation,
        with length G being the number of interpolation segments. Indices are set such
        that `baseline[skips[i][0]:skips[i][1]]` will have fitted values at the first
        and last indices and all other values (the slice [1:-1]) will be calculated by
        interpolation.

    Notes
    -----
    All changes to `baseline` are done inplace.

    """
    for i in prange(skips.shape[0]):
        window = skips[i]
        left = window[0]
        right = window[1]
        _interp_inplace(x[left:right], baseline[left:right], baseline[left], baseline[right - 1])


# adapted from (https://gist.github.com/agramfort/850437); see license above
@jit(nopython=True, cache=True, parallel=True)
def _loess_low_memory(x, y, weights, coefs, vander, num_x, windows, fits):
    """
    A version of loess that uses near constant memory.

    The distance-weighted kernel for each x-value is computed each loop, rather
    than cached, so memory usage is low but the calculation is slightly slower.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the measured data, with N data points.
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    num_x : int
        The number of data points in `x`, also known as N.
    windows : numpy.ndarray, shape (F, 2)
        An array of left and right indices that define the fitting window for each fit
        x-value. The length is F, which is the total number of fit points. If `fit_dx`
        is <= 0, F is equal to N, the total number of x-values.
    fits : numpy.ndarray, shape (F,)
        The array of indices indicating which x-values to fit.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    for idx in prange(fits.shape[0]):
        i = fits[idx]
        window = windows[idx]
        left = window[0]
        right = window[1]

        difference = np.abs(x[left:right] - x[i])
        difference = difference / max(difference[0], difference[-1])
        difference = difference * difference * difference
        difference = 1 - difference
        kernel = np.sqrt(difference * difference * difference)

        coef = _loess_solver(
            kernel * vander_fit[:, left:right], kernel * y_fit[left:right]
        )
        baseline[i] = vander[i].dot(coef)
        coefs[i] = coef

    return baseline


# adapted from (https://gist.github.com/agramfort/850437); see license above
@jit(nopython=True, cache=True, parallel=True)
def _loess_first_loop(x, y, weights, coefs, vander, total_points, num_x, windows, fits):
    """
    The initial fit for loess that also caches the window values for each x-value.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the measured data, with N data points.
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    total_points : int
        The number of points to include when fitting each x-value.
    num_x : int
        The number of data points in `x`, also known as N.
    windows : numpy.ndarray, shape (F, 2)
        An array of left and right indices that define the fitting window for each fit
        x-value. The length is F, which is the total number of fit points. If `fit_dx`
        is <= 0, F is equal to N, the total number of x-values.
    fits : numpy.ndarray, shape (F,)
        The array of indices indicating which x-values to fit.

    Returns
    -------
    kernels : numpy.ndarray, shape (num_x, total_points)
        The array containing the distance-weighted kernel for each x-value.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    kernels = np.empty((num_x, total_points))
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    for idx in prange(fits.shape[0]):
        i = fits[idx]
        window = windows[idx]
        left = window[0]
        right = window[1]

        difference = np.abs(x[left:right] - x[i])
        difference = difference / max(difference[0], difference[-1])
        difference = difference * difference * difference
        difference = 1 - difference
        kernel = np.sqrt(difference * difference * difference)

        kernels[i] = kernel
        coef = _loess_solver(
            kernel * vander_fit[:, left:right], kernel * y_fit[left:right]
        )
        baseline[i] = vander[i].dot(coef)
        coefs[i] = coef

    return kernels, baseline


@jit(nopython=True, cache=True, parallel=True)
def _loess_nonfirst_loops(y, weights, coefs, vander, kernels, windows, num_x, fits):
    """
    The loess fit to use after the first loop that uses the cached window values.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    weights : numpy.ndarray, shape (N,)
        The array of weights.
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    kernels : numpy.ndarray, shape (N, total_points)
        The array containing the distance-weighted kernel for each x-value. Each
        kernel has a length of total_points.
    windows : numpy.ndarray, shape (F, 2)
        An array of left and right indices that define the fitting window for each fit
        x-value. The length is F, which is the total number of fit points. If `fit_dx`
        is <= 0, F is equal to N, the total number of x-values.
    num_x : int
        The total number of values, N.
    fits : numpy.ndarray, shape (F,)
        The array of indices indicating which x-values to fit.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    baseline = np.empty(num_x)
    y_fit = y * weights
    vander_fit = vander.T * weights
    for idx in prange(fits.shape[0]):
        i = fits[idx]
        window = windows[idx]
        left = window[0]
        right = window[1]
        kernel = kernels[i]
        coef = _loess_solver(
            kernel * vander_fit[:, left:right], kernel * y_fit[left:right]
        )
        baseline[i] = vander[i].dot(coef)
        coefs[i] = coef

    return baseline


@jit(nopython=True, cache=True)
def _determine_fits(x, num_x, total_points, delta):
    """
    Determines the x-values to fit and the left and right indices for each fit x-value.

    The windows are set before fitting so that fitting can be done in parallel
    when numba is installed, since the left and right indices would otherwise
    need to be determined in order. Similarly, determining which x-values to fit would
    not be able to be done in parallel since it requires knowledge of the last x-value
    fit.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The array of x-values.
    num_x : int
        The total number of x-values, N.
    total_points : int
        The number of values to include in each fitting window.
    delta : float
        If `delta` is > 0, will skip all but the last x-value in the range x_last + `delta`,
        where x_last is the last x-value to be fit. Fits all x-values if `delta` is <= 0.

    Returns
    -------
    windows : numpy.ndarray, shape (F, 2)
        An array of left and right indices that define the fitting window for each fit
        x-value. The length is F, which is the total number of fit points. If `fit_dx`
        is <= 0, F is equal to N, the total number of x-values. Indices are set such
        that the number of values in `x[windows[i][0]:windows[i][1]] is equal to
        `total_points`.
    fits : numpy.ndarray, shape (F,)
        The array of indices indicating which x-values to fit.
    skips : numpy.ndarray, shape (G, 2)
        The array of left and right indices that define the windows for interpolation,
        with length G being the number of interpolation segments. G is 0 if `fit_dx` is
        <= 0. Indices are set such that `baseline[skips[i][0]:skips[i][1]]` will have
        fitted values at the first and last indices and all other values (the slice [1:-1])
        will be calculated by interpolation.

    Notes
    -----
    The dtype `np.intp` is used for `fits`, `skips`, and `windows` to be consistent with
    numpy since numpy internally uses that type when referring to indices.

    """
    # faster to allocate array and return only filled in sections
    # rather than constanly appending to a list
    if delta > 0:
        check_fits = True
        fits = np.empty(num_x, dtype=np.intp)
        fits[0] = 0  # always fit first item
        skips = np.empty((num_x, 2), dtype=np.intp)
    else:
        # TODO maybe use another function when fitting all points in order
        # to skip the if check_fits check for every x-value; does it affect
        # calculation time that much?
        check_fits = False
        # TODO once numba minimum version is >= 0.47, can use dtype kwarg in np.arange
        fits = np.arange(num_x).astype(np.intp)
        # numba cannot compile in nopython mode when directly creating
        # np.array([], dtype=np.intp), so work-around by creating np.array([[0, 0]])
        # and then index with [:total_skips], which becomes np.array([])
        # since total_skips is 0 when delta is <= 0.
        skips = np.array([[0, 0]], dtype=np.intp)

    windows = np.empty((num_x, 2), dtype=np.intp)
    windows[0] = (0, total_points)
    total_fits = 1
    total_skips = 0
    skip_start = 0
    skip_range = x[0] + delta
    left = 0
    right = total_points
    for i in range(1, num_x - 1):
        x_val = x[i]
        if check_fits:
            # use x[i+1] rather than x[i] since it ensures that the last value within
            # the range x_last_fit + delta is used; x[i+1] is also guranteed to be >= x[i]
            if x[i + 1] < skip_range:
                if not skip_start:
                    skip_start = i
                continue
            else:
                skip_range = x_val + delta
                fits[total_fits] = i
                if skip_start:
                    skips[total_skips] = (skip_start - 1, i + 1)
                    total_skips += 1
                    skip_start = 0

        while right < num_x and x_val - x[left] > x[right] - x_val:
            left += 1
            right += 1
        window = windows[total_fits]
        window[0] = left
        window[1] = right
        total_fits += 1

    if skip_start:  # fit second to last x-value
        fits[total_fits] = num_x - 2
        if x[-1] - x[-2] < x[-2] - x[num_x - total_points]:
            windows[total_fits] = (num_x - total_points, num_x)
        else:
            windows[total_fits] = (num_x - total_points - 1, num_x - 1)
        total_fits += 1
        skips[total_skips] = (skip_start - 1, num_x - 1)
        total_skips += 1

    # always fit last item
    fits[total_fits] = num_x - 1
    windows[total_fits] = (num_x - total_points, num_x)
    total_fits += 1

    return windows[:total_fits], fits[:total_fits], skips[:total_skips]


@_polynomial_wrapper
def loess(data, x_data=None, fraction=0.2, total_points=None, poly_order=1, scale=3.0,
          tol=1e-3, max_iter=10, symmetric_weights=False, use_threshold=False, num_std=1,
          use_original=False, weights=None, return_coef=False, conserve_memory=True, delta=0.0):
    """
    Locally estimated scatterplot smoothing (LOESS).

    Performs polynomial regression at each data point using the nearest points.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    fraction : float, optional
        The fraction of N data points to include for the fitting on each point.
        Default is 0.2. Not used if `total_points` is not None.
    total_points : int, optional
        The total number of points to include for the fitting on each point. Default
        is None, which will use `fraction` * N to determine the number of points.
    scale : float, optional
        A scale factor applied to the weighted residuals to control the robustness
        of the fit. Default is 3.0, as used in [9]_. Note that the original loess
        procedure in [10]_ used a `scale` of ~4.05.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 1.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 10.
    symmetric_weights : bool, optional
        If False (default), will apply weighting asymmetrically, with residuals
        < 0 having a weight of 1, according to [9]_. If True, will apply weighting
        the same for both positive and negative residuals, which is regular LOESS.
        If `use_threshold` is True, this parameter is ignored.
    use_threshold : bool, optional
        If False (default), will compute weights each iteration to perform the
        robust fitting, which is regular LOESS. If True, will apply a threshold
        on the data being fit each iteration, based on the maximum values of the
        data and the fit baseline, as proposed by [11]_, similar to the modpoly
        and imodpoly techniques.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Default
        is 1, which is the value used for the imodpoly technique. Only used if
        `use_threshold` is True.
    use_original : bool, optional
        If False (default), will compare the baseline of each iteration with
        the y-values of that iteration [12]_ when choosing minimum values for
        thresholding. If True, will compare the baseline with the original
        y-values given by `data` [13]_. Only used if `use_threshold` is True.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.
    conserve_memory : bool, optional
        If False, will cache the distance-weighted kernels for each value
        in `x_data` on the first iteration and reuse them on subsequent iterations to
        save time. The shape of the array of kernels is (len(`x_data`), `total_points`).
        If True (default), will recalculate the kernels each iteration, which uses very
        little memory, but is slower. Can usually set to False unless `x_data` and`total_points`
        are quite large and the function causes memory issues when cacheing the kernels. If
        numba is installed, there is no significant time difference since the calculations are
        sped up.
    delta : float, optional
        If `delta` is > 0, will skip all but the last x-value in the range x_last + `delta`,
        where x_last is the last x-value to be fit using weighted least squares, and instead
        use linear interpolation to calculate the fit for those x-values (same behavior as in
        statsmodels [14]_ and Cleveland's original Fortran lowess implementation [15]_).
        Fits all x-values if `delta` is <= 0. Default is 0.0. Note that `x_data` is scaled to
        fit in the range [-1, 1], so `delta` should likewise be scaled. For example, if the
        desired `delta` value was ``0.01 * (max(x_data) - min(x_data))``, then the
        correctly scaled `delta` would be 0.02 (ie. ``0.01 * (1 - (-1))``).

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data. Does NOT contain the
            individual distance-weighted kernels for each x-value.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.
        * 'coef': numpy.ndarray, shape (N, poly_order + 1)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a polynomial
            using :class:`numpy.polynomial.polynomial.Polynomial`. If `delta` is > 0,
            the coefficients for any skipped x-value will all be 0.

    Raises
    ------
    ValueError
        Raised if the number of points per window for the fitting is less than
        `poly_order` + 1 or greater than the total number of points.

    Notes
    -----
    The iterative, robust, aspect of the fitting can be achieved either through
    reweighting based on the residuals (the typical usage), or thresholding the
    fit data based on the residuals, as proposed by [11]_, similar to the modpoly
    and imodpoly techniques.

    In baseline literature, this procedure is sometimes called "rbe", meaning
    "robust baseline estimate".

    References
    ----------
    .. [9] Ruckstuhl, A.F., et al. Baseline subtraction using robust local
           regression estimation. J. Quantitative Spectroscopy and Radiative
           Transfer, 2001, 68, 179-193.
    .. [10] Cleveland, W. Robust locally weighted regression and smoothing
            scatterplots. Journal of the American Statistical Association,
            1979, 74(368), 829-836.
    .. [11] Komsta, Ł. Comparison of Several Methods of Chromatographic
            Baseline Removal with a New Approach Based on Quantile Regression.
            Chromatographia, 2011, 73, 721-731.
    .. [12] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
    .. [13] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
    .. [14] https://github.com/statsmodels/statsmodels.
    .. [15] https://www.netlib.org/go (lowess.f is the file).

    """


@_polynomial_wrapper
def quant_reg(data, x_data=None, poly_order=2, quantile=0.05, tol=1e-6, max_iter=250,
              weights=None, eps=None, return_coef=False):
    """
    Approximates the baseline of the data using quantile regression.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    quantile : float, optional
        The quantile at which to fit the baseline. Default is 0.05.
    tol : float, optional
        The exit criteria. Default is 1e-6. For extreme quantiles (`quantile` < 0.01
        or `quantile` > 0.99), may need to use a lower value to get a good fit.
    max_iter : int, optional
        The maximum number of iterations. Default is 250. For extreme quantiles
        (`quantile` < 0.01 or `quantile` > 0.99), may need to use a higher value to
        ensure convergence.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    eps : float, optional
        A small value added to the square of the residual to prevent dividing by 0.
        Default is None, which uses the square of the maximum-absolute-value of the
        fit each iteration multiplied by 1e-6.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input `x_data` and return them in the params dictionary.
        Default is False, since the conversion takes time.

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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Raises
    ------
    ValueError
        Raised if `quantile` is not between 0 and 1.

    Notes
    -----
    Application of quantile regression for baseline fitting as described in [16]_.

    Performs quantile regression using iteratively reweighted least squares (IRLS)
    as described in [17]_.

    References
    ----------
    .. [16] Komsta, Ł. Comparison of Several Methods of Chromatographic
            Baseline Removal with a New Approach Based on Quantile Regression.
            Chromatographia, 2011, 73, 721-731.
    .. [17] Schnabel, S., et al. Simultaneous estimation of quantile curves using
            quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

    """


@_polynomial_wrapper
def goldindec(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250, weights=None,
              cost_function='asymmetric_indec', peak_ratio=0.5, alpha_factor=0.99,
              tol_2=1e-3, tol_3=1e-6, max_iter_2=100, return_coef=False):
    """
    Fits a polynomial baseline using a non-quadratic cost function.

    The non-quadratic cost functions penalize residuals with larger values,
    giving a more robust fit compared to normal least-squares.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 2.
    tol : float, optional
        The exit criteria for the fitting with a given threshold value. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations for fitting a threshold value. Default is 250.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    cost_function : str, optional
        The non-quadratic cost function to minimize. Unlike :func:`.penalized_poly`,
        this function only works with asymmetric cost functions, so the symmetry prefix
        ('a' or 'asymmetric') is optional (eg. 'indec' and 'a_indec' are the same). Default
        is 'asymmetric_indec'. Available methods, and their associated reference, are:

            * 'asymmetric_indec'[18]_
            * 'asymmetric_truncated_quadratic'[19]_
            * 'asymmetric_huber'[19]_

    peak_ratio : float, optional
        A value between 0 and 1 that designates how many points in the data belong
        to peaks. Values are valid within ~10% of the actual peak ratio. Default is 0.5.
    alpha_factor : float, optional
        A value between 0 and 1 that controls the value of the penalty. Default is
        0.99. Typically should not need to change this value.
    tol_2 : float, optional
        The exit criteria for the difference between the optimal up-down ratio (number of
        points above 0 in the residual compared to number of points below 0) and the up-down
        ratio for a given threshold value. Default is 1e-3.
    tol_3 : float, optional
        The exit criteria for the relative change in the threshold value. Default is 1e-6.
    max_iter_2 : float, optional
        The number of iterations for iterating between different threshold values.
        Default is 100.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'tol_history': numpy.ndarray, shape (J, K)
            An array containing the calculated tolerance values for each iteration
            of both threshold values and fit values. Index 0 are the tolerence values
            for the difference in up-down ratios, index 1 are the tolerance values for
            the relative change in the threshold, and indices >= 2 are the tolerance values
            for each fit. All values that were not used in fitting have values of 0. Shape J
            is 2 plus the number of iterations for the threshold to converge (related to
            `max_iter_2`, `tol_2`, `tol_3`), and shape K is the maximum of the number of
            iterations for the threshold and the maximum number of iterations for all of
            the fits of the various threshold values (related to `max_iter` and `tol`).
        * 'threshold' : float
            The optimal threshold value. Could be used in :func:`.penalized_poly`
            for fitting other similar data.
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.

    Raises
    ------
    ValueError
        Raised if `alpha_factor` or `peak_ratio` are not between 0 and 1, or if the
        specified cost function is symmetric.

    References
    ----------
    .. [18] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
            Correction. Applied Spectroscopy, 2015, 69(7), 834-842.
    .. [19] Mazet, V., et al. Background removal from spectra by designing and
            minimising a non-quadratic cost function. Chemometrics and Intelligent
            Laboratory Systems, 2005, 76(2), 121-133.

    """
