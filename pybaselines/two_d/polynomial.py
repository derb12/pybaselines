# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on April 16, 2023
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

"""

import numpy as np

from .. import _weighting
from ..utils import _MIN_FLOAT, _convert_coef2d, relative_difference
from ._algorithm_setup import _Algorithm2D


class _Polynomial(_Algorithm2D):
    """A base class for all polynomial algorithms."""

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def poly(self, data, poly_order=2, weights=None, return_coef=False, max_cross=None):
        """
        Computes a polynomial that fits the baseline of the data.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for x and z. If a single value, will use that for both x and
            z. Default is 2.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the x and z values and return them in the params dictionary.
            Default is False, since the conversion takes time.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Notes
        -----
        To only fit regions without peaks, supply a weight array with zero values
        at the indices where peaks are located.

        """
        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True, max_cross=max_cross
        )
        sqrt_w = np.sqrt(weight_array)

        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self.vandermonde @ coef
        params = {'weights': weight_array}
        if return_coef:
            params['coef'] = _convert_coef2d(
                coef, self.poly_order[0], self.poly_order[1], self.x_domain, self.z_domain
            )

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def modpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                use_original=False, mask_initial_peaks=False, return_coef=False, max_cross=None):
        """
        The modified polynomial (ModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 with N points.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for x and z. If a single value, will use that for both x and
            z. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [33]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [34]_.
        mask_initial_peaks : bool, optional
            If True, will mask any data where the initial baseline fit + the standard
            deviation of the residual is less than measured data [35]_. Default is False.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the x and z values and return them in the params dictionary.
            Default is False, since the conversion takes time.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

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
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Notes
        -----
        Algorithm originally developed in [34]_ and then slightly modified in [33]_.

        References
        ----------
        .. [33] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [34] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [35] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True, copy_weights=True,
            max_cross=max_cross
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
            params['coef'] = _convert_coef2d(
                coef, self.poly_order[0], self.poly_order[1], self.x_domain, self.z_domain
            )

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def imodpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                 use_original=False, mask_initial_peaks=True, return_coef=False,
                 num_std=1., max_cross=None):
        """
        The improved modofied polynomial (IModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for x and z. If a single value, will use that for both x and
            z. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [36]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [37]_.
        mask_initial_peaks : bool, optional
            If True (default), will mask any data where the initial baseline fit +
            the standard deviation of the residual is less than measured data [38]_.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the x and z values and return them in the params dictionary.
            Default is False, since the conversion takes time.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Default
            is 1. Must be greater or equal to 0.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

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
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `num_std` is less than 0.

        Notes
        -----
        Algorithm originally developed in [38]_.

        References
        ----------
        .. [36] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [37] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [38] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        if num_std < 0:
            raise ValueError('num_std must be greater than or equal to 0')

        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True,
            copy_weights=True, max_cross=max_cross
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
            params['coef'] = _convert_coef2d(
                coef, self.poly_order[0], self.poly_order[1], self.x_domain, self.z_domain
            )

        return baseline, params

    # adapted from
    # https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction;
    # see license above
    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def penalized_poly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                       cost_function='asymmetric_truncated_quadratic', threshold=None,
                       alpha_factor=0.99, return_coef=False, max_cross=None):
        """
        Fits a polynomial baseline using a non-quadratic cost function.

        The non-quadratic cost functions penalize residuals with larger values,
        giving a more robust fit compared to normal least-squares.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for x and z. If a single value, will use that for both x and
            z. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        cost_function : str, optional
            The non-quadratic cost function to minimize. Must indicate symmetry of the
            method by appending 'a' or 'asymmetric' for asymmetric loss, and 's' or
            'symmetric' for symmetric loss. Default is 'asymmetric_truncated_quadratic'.
            Available methods, and their associated reference, are:

                * 'asymmetric_truncated_quadratic'[39]_
                * 'symmetric_truncated_quadratic'[39]_
                * 'asymmetric_huber'[39]_
                * 'symmetric_huber'[39]_
                * 'asymmetric_indec'[40]_
                * 'symmetric_indec'[40]_

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
            a form that fits the x and z values and return them in the params dictionary.
            Default is False, since the conversion takes time.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

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
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `alpha_factor` is not between 0 and 1.

        Notes
        -----
        In baseline literature, this procedure is sometimes called "backcor".

        References
        ----------
        .. [39] Mazet, V., et al. Background removal from spectra by designing and
            minimising a non-quadratic cost function. Chemometrics and Intelligent
            Laboratory Systems, 2005, 76(2), 121-133.
        .. [40] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
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
            data, weights, poly_order, calc_vander=True, calc_pinv=True, max_cross=max_cross
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
            params['coef'] = _convert_coef2d(
                coef, self.poly_order[0], self.poly_order[1], self.x_domain, self.z_domain
            )

        return baseline, params

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def quant_reg(self, data, poly_order=2, quantile=0.05, tol=1e-6, max_iter=250,
                  weights=None, eps=None, return_coef=False, max_cross=None):
        """
        Approximates the baseline of the data using quantile regression.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for x and z. If a single value, will use that for both x and
            z. Default is 2.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 0.05.
        tol : float, optional
            The exit criteria. Default is 1e-6. For extreme quantiles (`quantile` < 0.01
            or `quantile` > 0.99), may need to use a lower value to get a good fit.
        max_iter : int, optional
            The maximum number of iterations. Default is 250. For extreme quantiles
            (`quantile` < 0.01 or `quantile` > 0.99), may need to use a higher value to
            ensure convergence.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        eps : float, optional
            A small value added to the square of the residual to prevent dividing by 0.
            Default is None, which uses the square of the maximum-absolute-value of the
            fit each iteration multiplied by 1e-6.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the x and z values and return them in the params dictionary.
            Default is False, since the conversion takes time.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

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
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `quantile` is not between 0 and 1.

        Notes
        -----
        Application of quantile regression for baseline fitting ss described in [41]_.

        Performs quantile regression using iteratively reweighted least squares (IRLS)
        as described in [42]_.

        References
        ----------
        .. [41] Komsta, Ł. Comparison of Several Methods of Chromatographic
                Baseline Removal with a New Approach Based on Quantile Regression.
                Chromatographia, 2011, 73, 721-731.
        .. [42] Schnabel, S., et al. Simultaneous estimation of quantile curves using
                quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

        """
        # TODO provide a way to estimate best poly_order based on AIC like in Komsta? could be
        # useful for all polynomial methods; maybe could be an optimizer function
        if not 0 < quantile < 1:
            raise ValueError('quantile must be between 0 and 1.')

        y, weight_array = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, max_cross=max_cross
        )
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
            params['coef'] = _convert_coef2d(
                coef, self.poly_order[0], self.poly_order[1], self.x_domain, self.z_domain
            )

        return baseline, params


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
