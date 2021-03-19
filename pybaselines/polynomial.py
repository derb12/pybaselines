# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Polynomial
    1) ModPoly (Modified Polynomial)
    2) IModPoly (Improved Modified Polynomial)

Created on Feb. 27, 2021

@author: Donald Erb

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
    np.ndarray
        The array of coefficients scaled for the original domain.

    """
    fit_polynomial = np.polynomial.Polynomial(coef, domain=original_domain)
    return fit_polynomial.convert().coef


def poly(data, x_data=None, poly_order=2, weights=None, return_coef=False):
    """
    Computes a polynomial that fits the data.

    Parameters
    ----------
    data : [type]
        [description]
    x_data : [type]
        [description]
    poly_order : int, optional
        [description]. Default is 2.
    weights : [type], optional
        [description]. Default is None.

    Returns
    -------
    np.ndarray
        The background data from the polynomial.
    dict
        A dictionary containing the keys:

        * 'weights'
              The array of weights used.

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


def modpoly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=500, weights=None,
            mask_initial_peaks=False, use_original=False, return_coef=False):
    """
    The modified polynomial (ModPoly) baseline algorithm.

    Adapted from:
    C.A. Lieber, A. Mahadevan-Jansen, Automated method
    for subtraction of fluorescence from biological raman spectra,
    Applied Spectroscopy 57(11) (2003) 1363-1367.

    and

    Gan, F., et al. Baseline correction by improved iterative polynomial fitting
    with automatic threshold. Chemometrics and Intelligent Laboratory Systems, 82
    (2006) 59-65.

    The mask_initial_peaks idea was adopted from the IModPoly:
    Zhao, J., et al., Automated Autofluorescence Background Subtraction
    Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy
    61(11) (2007) 1225-1232.

    use_original=True is Lieber's method, and use_original=False is Gan's method.

    """
    y, x, w, original_domain, vander, vander_pinv = utils._setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(w)
    if use_original:
        y0 = y

    coef = np.dot(vander_pinv, sqrt_w * y)
    z = np.dot(vander, coef)
    if mask_initial_peaks:
        # use z + deviation since without deviation, half of y should be above z
        w[z + np.std(y - z) < y] = 0
        sqrt_w = np.sqrt(w)
        vander, vander_pinv = utils._get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(vander_pinv, sqrt_w * y)
        z_new = np.dot(vander, coef)
        if utils.relative_difference(z, z_new) < tol:
            break
        z = z_new

    params = {'weights': w}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return z, params


def imodpoly(data, x_data=None, poly_order=2, tol=1e-3, max_iter=500, weights=None,
             mask_initial_peaks=True, use_original=False, return_coef=False):
    """
    The improved modofied polynomial (IModPoly) baseline algorithm.

    Adapted from:
    Zhao, J., et al., Automated Autofluorescence Background Subtraction
    Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy
    61(11) (2007) 1225-1232.

    The use_original term is adapted from the ModPoly algorithm (is False for
    Zhao's original IModPoly implementation):
    C.A. Lieber, A. Mahadevan-Jansen, Automated method
    for subtraction of fluorescence from biological raman spectra,
    Applied Spectroscopy 57(11) (2003) 1363-1367.

    """
    y, x, w, original_domain, vander, vander_pinv = utils._setup_polynomial(
        data, x_data, weights, poly_order, return_vander=True, return_pinv=True
    )
    sqrt_w = np.sqrt(w)
    if use_original:
        y0 = y

    coef = np.dot(vander_pinv, sqrt_w * y)
    z = np.dot(vander, coef)
    deviation = np.std(y - z)
    if mask_initial_peaks:
        w[z + deviation < y] = 0
        sqrt_w = np.sqrt(w)
        vander, vander_pinv = utils._get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(vander_pinv, sqrt_w * y)
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
