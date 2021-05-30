# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on Feb. 27, 2021
@author: Donald Erb


The function penalized_poly is adapted from MATLAB code from
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


The function loess is adapted from code from https://gist.github.com/agramfort/850437
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

import numpy as np

from ._algorithm_setup import _get_vander, _setup_polynomial
from .utils import _MIN_FLOAT, relative_difference


def _convert_coef(coef, original_domain):
    """
    Scales the polynomial coefficients back to the original domain of the data.

    For fitting, the x-values are scaled from their original domain, [min(x),
    max(x)], to [-1, 1] in order to improve the numerical stability of fitting.
    This function rescales the retrieved polynomial coefficients for the fit
    x-values back to the original domain.

    Parameters
    ----------
    coef : array-like
        The array of coefficients for the polynomial. Should increase in
        order, for example (c0, c1, c2) from `y = c0 + c1 * x + c2 * x**2`.
    original_domain : array-like, shape (2,)
        The domain, [min(x), max(x)], of the original data used for fitting.

    Returns
    -------
    numpy.ndarray
        The array of coefficients scaled for the original domain.

    """
    fit_polynomial = np.polynomial.Polynomial(coef, domain=original_domain)
    return fit_polynomial.convert().coef


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
            polynomial using numpy.polynomial.polynomial.Polynomial().

    Notes
    -----
    To only fit regions without peaks, supply a weight array with zero values
    at the indices where peaks are located.

    """
    y, x, weight_array, original_domain = _setup_polynomial(data, x_data, weights)
    fit_polynomial = np.polynomial.Polynomial.fit(x, y, poly_order, w=np.sqrt(weight_array))
    baseline = fit_polynomial(x)
    params = {'weights': weight_array}
    if return_coef:
        params['coef'] = fit_polynomial.convert(window=original_domain).coef

    return baseline, params


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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using numpy.polynomial.polynomial.Polynomial().

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
    y, x, weight_array, original_domain, vander, pseudo_inverse = _setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(weight_array)
    if use_original:
        y0 = y

    coef = np.dot(pseudo_inverse, sqrt_w * y)
    baseline = np.dot(vander, coef)
    if mask_initial_peaks:
        # use baseline + deviation since without deviation, half of y should be above baseline
        weight_array[baseline + np.std(y - baseline) < y] = 0
        sqrt_w = np.sqrt(weight_array)
        vander, pseudo_inverse = _get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, baseline)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        baseline_new = np.dot(vander, coef)
        if relative_difference(baseline, baseline_new) < tol:
            break
        baseline = baseline_new

    params = {'weights': weight_array}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return baseline, params


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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using numpy.polynomial.polynomial.Polynomial().

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
    y, x, weight_array, original_domain, vander, pseudo_inverse = _setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(weight_array)
    if use_original:
        y0 = y

    coef = np.dot(pseudo_inverse, sqrt_w * y)
    baseline = np.dot(vander, coef)
    deviation = np.std(y - baseline)
    if mask_initial_peaks:
        weight_array[baseline + deviation < y] = 0
        sqrt_w = np.sqrt(weight_array)
        vander, pseudo_inverse = _get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, baseline + num_std * deviation)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        baseline = np.dot(vander, coef)
        new_deviation = np.std(y - baseline)
        # use new_deviation as dividing term in relative difference
        if relative_difference(new_deviation, deviation) < tol:
            break
        deviation = new_deviation

    params = {'weights': weight_array}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return baseline, params


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
    Laboratory Systems, 2005, 76(2), 121–133.

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
    Laboratory Systems, 2005, 76(2), 121–133.

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
    Laboratory Systems, 2005, 76(2), 121–133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for goldindec is 0.5
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
        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using numpy.polynomial.polynomial.Polynomial().

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
           Laboratory Systems, 2005, 76(2), 121–133.
    .. [8] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
           Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

    """
    if alpha_factor < 0 or alpha_factor > 1:
        raise ValueError('alpha_factor must be between 0 and 1')
    symmetric_loss, method = _identify_loss_method(cost_function)
    loss_function = {
        'huber': _huber_loss,
        'truncated_quadratic': _truncated_quadratic_loss,
        'indec': _indec_loss
    }[method]
    y, x, weight_array, original_domain, vander, pseudo_inverse = _setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    if threshold is None:
        threshold = np.std(y) / 10
    loss_kwargs = {
        'threshold': threshold, 'alpha_factor': alpha_factor, 'symmetric': symmetric_loss
    }

    sqrt_w = np.sqrt(weight_array)
    y = sqrt_w * y

    coef = np.dot(pseudo_inverse, y)
    baseline = np.dot(vander, coef)
    for _ in range(max_iter):
        coef = np.dot(pseudo_inverse, y + loss_function(y - sqrt_w * baseline, **loss_kwargs))
        baseline_new = np.dot(vander, coef)

        if relative_difference(baseline, baseline_new) < tol:
            break
        baseline = baseline_new

    params = {'weights': weight_array}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return baseline, params


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
        mask = residual >= 0
        inner = residual[mask] / scale
        weights[mask] = np.maximum(0, 1 - inner * inner)
    return weights


def _median_absolute_differences(array_1, array_2=None, errors=None):
    """
    Computes the median absolute difference (MAD) between two arrays.

    Parameters
    ----------
    array_1 : numpy.ndarray
        The first array. If `array_2` is None, then `array_1` is assumed to
        be the difference of the two arrays.
    array_2 : numpy.ndarray, optional
        The second array. If None (default), then the function assumes the
        difference of the two arrays was already computed and input as `array_1`.
    errors : numpy.ndarray, optional
        The array of errors associated with the measurement.

    Returns
    -------
    float
        The scaled median absolute difference for the input arrays.

    Notes
    -----
    The 1/0.6745 scale factor is to make the result comparable to the
    standard deviation of a Gaussian distribution.

    Reference
    ---------
    Ruckstuhl, A.F., et al., Baseline subtraction using robust local regression
    estimation. J. Quantitative Spectroscopy and Radiative Transfer, 2001, 68,
    179-193.

    """
    if array_2 is None:
        difference = array_1
    else:
        difference = array_1 - array_2
    if errors is not None:
        return np.nanmedian(np.sqrt(errors) * np.abs(difference)) / 0.6745
    else:
        return np.nanmedian(np.abs(difference)) / 0.6745


def _loess_low_memory(x, y, coefs, vander, total_points, num_x, use_threshold, weights):
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
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    total_points : int
        The number of points to include when fitting each x-value.
    num_x : int
        The number of data points in `x`, also known as N.
    use_threshold : bool
        If False, will also use `weights` when calculating the total weighting
        for each window.
    weights : numpy.ndarray, shape (N,)
        The array of weights for the data. Only used if `use_threshold` is False.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    left = 0
    right = total_points - 1
    max_right = num_x - 1
    for i, x_val in enumerate(x):
        difference = abs(x - x_val)
        while right < max_right and difference[left] > difference[right + 1]:
            left += 1
            right += 1
        window_slice = slice(left, right + 1)
        inner = difference[window_slice] / max(difference[left], difference[right])
        inner = inner * inner * inner
        inner = 1 - inner
        kernel = np.sqrt(inner * inner * inner)
        if use_threshold:
            weight_array = kernel
        else:
            weight_array = kernel * weights[window_slice]

        coefs[i] = np.linalg.lstsq(
            weight_array[:, np.newaxis] * vander[window_slice],
            weight_array * y[window_slice],
            None
        )[0]


def _loess_first_loop(x, y, coefs, vander, total_points, num_x, use_threshold, weights):
    """
    The initial fit for loess that also caches the window values for each x-value.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values of the measured data, with N data points.
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    total_points : int
        The number of points to include when fitting each x-value.
    num_x : int
        The number of data points in `x`, also known as N.
    use_threshold : bool
        If False, will also use `weights` when calculating the total weighting
        for each window.
    weights : numpy.ndarray, shape (N,)
        The array of weights for the data. Only used if `use_threshold` is False.

    Returns
    -------
    kernels : numpy.ndarray, shape (num_x, total_points)
        The array containing the distance-weighted kernel for each x-value.
    windows : list(slice)
        A list of slices that define the indices for each window to use for
        fitting each x-value. Has a length of `num_x`.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    left = 0
    right = total_points - 1
    max_right = num_x - 1
    kernels = np.empty((num_x, total_points))
    windows = [None] * num_x
    for i, x_val in enumerate(x):
        difference = abs(x - x_val)
        while right < max_right and difference[left] > difference[right + 1]:
            left += 1
            right += 1
        window_slice = slice(left, right + 1)

        inner = difference[window_slice] / max(difference[left], difference[right])
        inner = inner * inner * inner
        inner = 1 - inner
        kernel = np.sqrt(inner * inner * inner)
        if use_threshold:
            weight_array = kernel
        else:
            weight_array = kernel * weights[window_slice]

        windows[i] = window_slice
        kernels[i] = kernel
        coefs[i] = np.linalg.lstsq(
            weight_array[:, np.newaxis] * vander[window_slice],
            weight_array * y[window_slice],
            None
        )[0]

    return kernels, windows


def _loess_nonfirst_loops(y, coefs, vander, use_threshold, weights, kernels, windows):
    """
    The loess fit to use after the first loop that uses the cached window values.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N points.
    coefs : numpy.ndarray, shape (N, poly_order + 1)
        The array of polynomial coefficients (with polynomial order poly_order),
        for each value in `x`.
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the `x` array.
    use_threshold : bool
        If False, will also use `weights` when calculating the total weighting
        for each window.
    weights : numpy.ndarray, shape (N,)
        The array of weights for the data. Only used if `use_threshold` is False.
    kernels : numpy.ndarray, shape (N, total_points)
        The array containing the distance-weighted kernel for each x-value. Each
        kernel has a length of total_points.
    windows : list(slice)
        A list of slices that define the indices for each window to use for
        fitting each x-value. Has a length of N.

    Notes
    -----
    The coefficient array, `coefs`, is modified inplace.

    """
    for i, kernel in enumerate(kernels):
        window_slice = windows[i]
        if use_threshold:
            weight_array = kernel
        else:
            weight_array = kernel * weights[window_slice]

        coefs[i] = np.linalg.lstsq(
            weight_array[:, np.newaxis] * vander[window_slice],
            weight_array * y[window_slice],
            None
        )[0]


def loess(data, x_data=None, fraction=0.2, total_points=None, poly_order=1, scale=3.0,
          tol=1e-3, max_iter=10, symmetric_weights=False, use_threshold=False, num_std=1,
          use_original=False, weights=None, return_coef=False, conserve_memory=False):
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
        procedure in [10]_ used a `scale` of 4.05.
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
        size equal to N and all values set to 1. Only used if `use_threshold` is
        False.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.
    conserve_memory : bool, optional
        If False (default), will cache the distance-weighted kernels for each value
        in `x_data` on the first iteration and reuse them on subsequent iterations to
        save time. The shape of the array of kernels is (len(`x_data`), `total_points`).
        If True, will recalculate the kernels each iteration, which uses very little memory,
        but is slower. Only need to set to True if `x_data` and `total_points` are quite
        large and the function causes memory issues when cacheing the kernels.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data. Does NOT contain the
            individual distance-weighted kernels for each x-value.
        * 'coef': numpy.ndarray, shape (N, poly_order + 1)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using numpy.polynomial.polynomial.Polynomial().

    Raises
    ------
    ValueError
        Raised if the number of points for the fitting is less than 2.

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
    .. [9] Ruckstuhl, A.F., et al., Baseline subtraction using robust local
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

    """
    y, x, weight_array, original_domain = _setup_polynomial(data, x_data, weights, poly_order)
    num_x = x.shape[0]
    if total_points is None:
        total_points = ceil(fraction * num_x)
    if total_points < 2:  #TODO probably also dictated by the polynomial order
        raise ValueError('total points must be greater than 1')
    if use_original:
        y0 = y

    sort_order = np.argsort(x)  # to ensure x is increasing
    x = x[sort_order]
    y = y[sort_order]
    vander = _get_vander(x, poly_order, calc_pinv=False)

    baseline = y
    coefs = np.empty((num_x, poly_order + 1))
    for i in range(max_iter):
        if conserve_memory:
            _loess_low_memory(
                x, y, coefs, vander, total_points, num_x, use_threshold, weight_array
            )
        elif i == 0:
            kernels, windows = _loess_first_loop(
                x, y, coefs, vander, total_points, num_x, use_threshold, weight_array
            )
        else:
            _loess_nonfirst_loops(
                y, coefs, vander, use_threshold, weight_array, kernels, windows
            )

        # einsum is same as np.array([np.dot(vander[i], coefs[i]) for i in range(num_x)])
        baseline_new = np.einsum('ij,ij->i', vander, coefs)
        if relative_difference(baseline, baseline_new) < tol:
            if i == 0:
                # in case tol was reached on first iteration
                baseline = baseline_new
            break

        baseline = baseline_new
        if use_threshold:
            y = np.minimum(
                y0 if use_original else y,
                baseline_new + num_std * np.std(y - baseline_new)
            )
        else:
            residual = y - baseline_new
            weight_array = _tukey_square(
                residual / _median_absolute_differences(residual), scale, symmetric_weights
            )

    params = {'weights': weight_array}
    if return_coef:
        params['coef'] = np.array([_convert_coef(coef, original_domain) for coef in coefs])

    return baseline[np.argsort(sort_order)], params
