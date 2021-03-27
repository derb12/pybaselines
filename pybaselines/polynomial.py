# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Polynomial
    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)

Created on Feb. 27, 2021
@author: Donald Erb


The function penalized_poly contains ported MATLAB code from
https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction
licensed under the BSD-2-Clause license, included below.

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

from . import utils


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
    z : numpy.ndarray, shape (N,)
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
    y, x, w, original_domain = utils._setup_polynomial(data, x_data, weights)
    fit_polynomial = np.polynomial.Polynomial.fit(x, y, poly_order, w=np.sqrt(w))
    z = fit_polynomial(x)
    params = {'weights': w}
    if return_coef:
        params['coef'] = fit_polynomial.convert(window=original_domain).coef

    return z, params


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
    z : numpy.ndarray, shape (N,)
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
    y, x, w, original_domain, vander, pseudo_inverse = utils._setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(w)
    if use_original:
        y0 = y

    coef = np.dot(pseudo_inverse, sqrt_w * y)
    z = np.dot(vander, coef)
    if mask_initial_peaks:
        # use z + deviation since without deviation, half of y should be above z
        w[z + np.std(y - z) < y] = 0
        sqrt_w = np.sqrt(w)
        vander, pseudo_inverse = utils._get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        z_new = np.dot(vander, coef)
        if utils.relative_difference(z, z_new) < tol:
            break
        z = z_new

    params = {'weights': w}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return z, params


def imodpoly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250, weights=None,
             use_original=False, mask_initial_peaks=True, return_coef=False):
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

    Returns
    -------
    z : numpy.ndarray, shape (N,)
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
    y, x, w, original_domain, vander, pseudo_inverse = utils._setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(w)
    if use_original:
        y0 = y

    coef = np.dot(pseudo_inverse, sqrt_w * y)
    z = np.dot(vander, coef)
    deviation = np.std(y - z)
    if mask_initial_peaks:
        w[z + deviation < y] = 0
        sqrt_w = np.sqrt(w)
        vander, pseudo_inverse = utils._get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        z = np.dot(vander, coef)
        new_deviation = np.std((y0 if use_original else y) - z)
        # use new_deviation as dividing term in relative difference
        if utils.relative_difference(new_deviation, deviation) < tol:
            break
        deviation = new_deviation

    params = {'weights': w}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return z, params


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
        - (~mask) * (residual + alpha * multiple * threshold**3 / np.maximum(2 * residual**2, utils._MIN_FLOAT))
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
    split_method = loss_method.lower().split('_')
    if (split_method[0] not in ('a', 's', 'asymmetric', 'symmetric')
            or len(split_method) < 2):
        raise ValueError('must specify loss function symmetry by appending "a_" or "s_"')
    if split_method[0] in ('a', 'asymmetric'):
        symmetric = False
    else:
        symmetric = True
    return symmetric, '_'.join(split_method[1:])


def penalized_poly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=250,
                   cost_function='asymmetric_truncated_quadratic', threshold=1.0,
                   return_coef=False):
    """
    Fits a polynomial baseline using a non-quadratic cost function.

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
    cost_function : str, optional
        The non-quadratic cost function to minimize. Must indicate symmetry of the
        method by appending 'a' or 'asymmetric' for asymmetric loss, and 's' or
        'symmetric' for symmetric loss. Default is 'asymmetric_truncated_quadratic'.
        Available methods, and their associated reference, are:

            * 'asymmetric_truncated_quadratic'[7]_
            * 'symmetric_truncated_quadratic'[7]_
            * 'asymmetric_huber'[7]_
            * 'symmetric_huber'[7]_
            * 'asymmetric_indec'[9]_
            * 'symmetric_indec'[9]_

    threshold : float, optional
        The threshold value for the loss method, where the function goes from
        quadratic loss (such as used for least squares) to non-quadratic. For
        symmetric loss methods, residual values with absolute value less than
        threshold will have quadratic loss. For asymmetric loss methods, residual
        values less than the threshold will have quadratic loss. Default is 1.0.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input x_data and return them in the params dictionary.
        Default is False, since the conversion takes time.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'coef': numpy.ndarray, shape (poly_order + 1,)
            Only if `return_coef` is True. The array of polynomial parameters
            for the baseline, in increasing order. Can be used to create a
            polynomial using numpy.polynomial.polynomial.Polynomial().

    Notes
    -----
    Code was partially adapted from MATLAB code from [8]_.

    References
    ----------
    .. [7] Mazet, V., et al. Background removal from spectra by designing and
           minimising a non-quadratic cost function. Chemometrics and Intelligent
           Laboratory Systems, 2005, 76(2), 121–133.
    .. [8] Vincent Mazet (2021). Background correction
           (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction),
           MATLAB Central File Exchange. Retrieved March 18, 2021.
    .. [9] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
           Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

    """
    #TODO should scale y between [-1, 1] so that abs(residual) is roughly <~ 1 and threshold <= 1
    # this would allow using same threshold regardless of y scale; would need to scale output
    # coefficients, though
    alpha_factor = 0.99  #TODO should alpha_factor be a param?
    symmetric_loss, method = _identify_loss_method(cost_function)
    loss_kwargs = {
        'threshold': threshold, 'alpha_factor': alpha_factor, 'symmetric': symmetric_loss
    }
    loss_function = {
        'huber': _huber_loss,
        'truncated_quadratic': _truncated_quadratic_loss,
        'indec': _indec_loss
    }[method]

    y, x, _, original_domain, vander, pseudo_inverse = utils._setup_polynomial(
        data, x_data, None, poly_order, return_vander=True, return_pinv=True
    )

    coef = np.dot(pseudo_inverse, y)
    z = np.dot(vander, coef)
    for _ in range(max_iter):
        coef = np.dot(pseudo_inverse, y + loss_function(y - z, **loss_kwargs))
        z_new = np.dot(vander, coef)

        if utils.relative_difference(z, z_new) < tol:
            break
        z = z_new

    params = {}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return z, params
