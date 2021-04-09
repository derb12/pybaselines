# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Polynomial
    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)
    5) loess (Locally Estimated Scatterplot Smoothing)

Created on Feb. 27, 2021
@author: Donald Erb

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
    y, x, w, original_domain = _setup_polynomial(data, x_data, weights)
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
    y, x, w, original_domain, vander, pseudo_inverse = _setup_polynomial(
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
        vander, pseudo_inverse = _get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        z_new = np.dot(vander, coef)
        if relative_difference(z, z_new) < tol:
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
    y, x, w, original_domain, vander, pseudo_inverse = _setup_polynomial(
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
        vander, pseudo_inverse = _get_vander(x, poly_order, sqrt_w)

    for _ in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        coef = np.dot(pseudo_inverse, sqrt_w * y)
        z = np.dot(vander, coef)
        new_deviation = np.std((y0 if use_original else y) - z)
        # use new_deviation as dividing term in relative difference
        if relative_difference(new_deviation, deviation) < tol:
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
        - (~mask) * (residual + alpha * multiple * threshold**3 / np.maximum(2 * residual**2, _MIN_FLOAT))
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
    In baseline literature, this procedure is sometimes called "backcor".

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

    y, x, _, original_domain, vander, pseudo_inverse = _setup_polynomial(
        data, x_data, None, poly_order, return_vander=True, return_pinv=True
    )

    coef = np.dot(pseudo_inverse, y)
    z = np.dot(vander, coef)
    for _ in range(max_iter):
        coef = np.dot(pseudo_inverse, y + loss_function(y - z, **loss_kwargs))
        z_new = np.dot(vander, coef)

        if relative_difference(z, z_new) < tol:
            break
        z = z_new

    params = {}
    if return_coef:
        params['coef'] = _convert_coef(coef, original_domain)

    return z, params


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
        weights = np.maximum(0, 1 - (residual / scale)**2)
    else:
        weights = np.ones_like(residual)
        mask = residual >= 0
        weights[mask] = np.maximum(0, 1 - (residual[mask] / scale)**2)
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


def loess(data, x_data=None, fraction=0.2, total_points=None, poly_order=1,
          scale=3.0, tol=1e-3, max_iter=10, symmetric_weights=False,
          use_threshold=False, include_stdev=True):
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
        is None, which will use fraction * N to determine the number of points.
    scale : float, optional
        A scale factor applied to the weighted residuals to control the robustness
        of the fit. Default is 3.0, as used in [10]_. Note that the original loess
        procedure in [11]_ used a `scale` of 4.05.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 1.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 10.
    symmetric_weights : bool, optional
        If False (default), will apply weighting asymmetrically, with residuals
        < 0 having a weight of 1, according to [10]_. If True, will apply weighting
        the same for both positive and negative residuals, which is regular LOESS.
        If `use_threshold` is True, this parameter is ignored.
    use_threshold : bool, optional
        If False (default), will compute weights each iteration to perform the
        robust fitting, which is regular LOESS. If True, will apply a threshold
        on the data being fit each iteration, based on the maximum values of the
        data and the fit baseline, as proposed by [12]_, similar to the ModPoly
        and IModPoly techniques.
    include_stdev : bool, optional
        If True (default), then will include the standard devitation of the
        residual when performing the thresholding, similar to the IModPoly
        technique [13]_. A value of True performs better when the input data
        has a high amount of noise. Only used if `use_threshold` is True.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Raises
    ------
    ValueError
        Raised if the number of points for the fitting is less than 2.

    Notes
    -----
    The iterative, robust, aspect of the fitting can be achieved either through
    reweighting based on the residuals (the typical usage), or thresholding the
    fit data based on the residuals, as proposed by [12]_, similar to the ModPoly
    and IModPoly techniques.

    In baseline literature, this procedure is sometimes called "rbe", meaning
    "robust baseline estimate".

    Code partially adapted from https://gist.github.com/agramfort/850437
    (accessed March 25, 2021).

    References
    ----------
    .. [10] Ruckstuhl, A.F., et al., Baseline subtraction using robust local
            regression estimation. J. Quantitative Spectroscopy and Radiative
            Transfer, 2001, 68, 179-193.
    .. [11] Cleveland, W. Robust locally weighted regression and smoothing
            scatterplots. Journal of the American Statistical Association,
            1979, 74(368), 829-836.
    .. [12] Komsta, Ł. Comparison of Several Methods of Chromatographic
            Baseline Removal with a New Approach Based on Quantile Regression.
            Chromatographia, 2011, 73, 721-731.
    .. [13] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

    """
    y, x, weights, _, vander = _setup_polynomial(data, x_data, None, poly_order, True)
    num_x = x.shape[0]
    if total_points is None:
        total_points = ceil(fraction * num_x)
    if total_points < 2:  #TODO probably also dictated by the polynomial order
        raise ValueError('total points must be greater than 1')

    #TODO potentially use k-d trees instead, following Cleveland, et al., Regression by local fitting. 1988.
    # or sort x before, and just slide the window to the right as you go along
    kernels = np.empty((num_x, num_x))
    for i, x_val in enumerate(x):
        difference = np.abs(x - x_val)
        kernels[i] = np.sqrt(
            (1 - np.clip(difference / np.sort(difference)[total_points - 1], 0, 1)**3)**3
        )

    z = np.zeros(num_x)
    z_new = np.empty(num_x)
    for _ in range(max_iter):
        for i, kernel in enumerate(kernels):
            if use_threshold:
                weight_array = kernel
            else:
                weight_array = kernel * weights
            coef = np.linalg.lstsq(
                weight_array[:, np.newaxis] * vander, weight_array * y, None
            )[0]
            #TODO should the coefficients be returned? probably not since it would slow it down a lot
            z_new[i] = np.dot(vander[i], coef)
        if relative_difference(z, z_new) < tol:
            break

        z = z_new.copy()
        if not use_threshold:
            residual = y - z_new
            weights = _tukey_square(
                residual / _median_absolute_differences(residual), scale, symmetric_weights
            )
        else:
            y = np.minimum(y, z_new + (0 if not include_stdev else np.std(y - z_new)))

    return z, {}
