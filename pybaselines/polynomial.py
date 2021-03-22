# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Polynomial
    1) poly (regular polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)



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

