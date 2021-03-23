# -*- coding: utf-8 -*-
"""Helper functions for pybaselines.

Created on March 5, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags


# the minimum positive float values such that a + _MIN_FLOAT != a
_MIN_FLOAT = np.finfo(float).eps


def difference_matrix(data_size, diff_order=2):
    """
    Creates an n-order differential matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : {2, 1, 3, 4, 5}, optional
        The integer differential order; either 1, 2, 3, 4, or 5. Default is 2.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        The sparse diagonal matrix of the differential.

    Raises
    ------
    ValueError
        Raised if diff_order is not 1, 2, 3, 4, or 5.

    Notes
    -----
    Most baseline algorithms use 2nd order differential matrices when
    doing penalized least squared fitting.

    It would be possible to support any differential order by doing
    np.diff(np.eye(data_size), diff_order), but the resulting matrix could
    cause issues if data_size is large. Therefore, it's better to only
    provide sparse arrays for the most commonly used differential orders.

    The resulting matrices are transposes of the result of
    np.diff(np.eye(data_size), diff_order). Not sure why there is a discrepancy,
    but this implementation allows using the differential matrices are they
    are written in various publications, ie. D.T * D rather than having to
    do D * D.T like most code, such as those adapted from stack overflow:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    """
    if diff_order not in (1, 2, 3, 4, 5):
        raise ValueError('The differential order must be 1, 2, 3, 4, or 5')
    diagonals = {
        1: [-1, 1],
        2: [1, -2, 1],
        3: [-1, 3, -3, 1],
        4: [1, -4, 6, -4, 1],
        5: [-1, 5, -10, 10, -5, 1]
    }[diff_order]

    return diags(diagonals, list(range(diff_order + 1)), shape=(data_size - diff_order, data_size))


def _setup_whittaker(data, lam, diff_order=2, weights=None):
    """
    Sets the starting parameters for doing penalized least squares.

    Parameters
    ----------
    data : array-like, shape (M,)
        The y-values of the measured data, with M data points.
    lam : float
        The smoothing parameter, lambda. Typical values are between 10 and
        1e8, but it strongly depends on the penalized least square method
        and the differential order.
    diff_order : {2, 1, 3, 4, 5}, optional
        The integer differential order; either 1, 2, 3, 4, or 5. Default is 2.
    weights : array-like, shape (M,), optional
        The weighting array. If None (default), then will be an array with
        shape (M,) and all values set to 1.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    scipy.sparse.dia.dia_matrix
        The product of lam * D.T * D, where D is the sparse diagonal matrix of
        the differential, and D.T is the transpose of D.
    scipy.sparse.dia.dia_matrix
        The sparse weight matrix with the weighting array as the diagonal values.
    weight_array : numpy.ndarray, shape (N,), optional
        The weighting array.

    """
    y = np.asarray(data)
    diff_matrix = difference_matrix(y.shape[0], diff_order)
    if weights is None:
        weight_array = np.ones(y.shape[0])
    else:
        weight_array = np.asarray(weights).copy()
    return y, lam * diff_matrix.T * diff_matrix, diags(weight_array), weight_array


def _get_vander(x, poly_order=2, weights=None, calc_pinv=True):
    """
    Calculates the Vandermonde matrix and its pseudo-inverse.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values for the polynomial with N data points.
    poly_order : int, optional
        The polynomial order. Default is 2.
    weights : np.ndarray, shape (N,), optional
        The weighting array. If None (default), will ignore. Otherwise,
        will multiply the Vandermonde by the weighting array before calculating
        the pseudo-inverse.
    calc_pinv : bool, optional
        If True (default), will calculate and return the pseudo-inverse of the
        Vandermonde, after applying weights.

    Returns
    -------
    vander : numpy.ndarray
        The Vandermonde matrix for the polynomial.
    vander_pinv : numpy.ndarray
        The pseudo-inverse of the Vandermonde, with weights applied if input.
        Calculated using singular value decomposition (SVD).

    Notes
    -----
    If weights are supplied, they should be the square-root of the total weights.

    """
    vander = np.polynomial.polynomial.polyvander(x, poly_order)
    if not calc_pinv:
        return vander

    if weights is not None:
        vander_pinv = np.linalg.pinv(diags(weights) * vander)
    else:
        vander_pinv = np.linalg.pinv(vander)

    return vander, vander_pinv


def _setup_polynomial(data, x_data=None, weights=None, poly_order=2,
                      return_vander=False, return_pinv=False):
    """
    Sets the starting parameters for doing polynomial fitting.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    poly_order : int, optional
        The polynomial order. Default is 2.
    return_vander : bool, optional
        If True, will calculate and return the Vandermonde matrix. Default is False.
    return_pinv : bool, optional
        If True, and if `return_vander` is True, will calculate and return the
        pseudo-inverse of the Vandermonde matrix. Default is False.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    x : numpy.ndarray, shape (N,)
        The x-values for fitting the polynomial, converted to fit within
        the domain [-1, 1].
    w : numpy.ndarray, shape (N,)
        The weight array for fitting a polynomial to the data.
    original_domain : numpy.ndarray, shape (2,)
        The minimum and maximum values of the original x_data values. Can
        be used to convert the coefficents found during least squares
        minimization using the normalized x into usable polynomial coefficients
        for the original x_data.
    vander : numpy.ndarray
        Only returned if return_vander is True. The Vandermonde matrix for the
        normalized x values.
    vander_pinv : numpy.ndarray
        Only returned if return_pinv is True. The pseudo-inverse of the
        Vandermonde matrix, calculated with singular value decomposition (SVD).

    Notes
    -----
    If x_data is given, its domain is reduced from [min(x_data), max(x_data)]
    to [-1, 1] to improve the numerical stability of calculations; since the
    Vandermonde matrix goes from x^0 to x^poly_order, large values of x would
    otherwise cause difficulty when doing least squares minimization.

    """
    y = np.asarray(data)
    if x_data is None:
        x = np.linspace(-1, 1, y.shape[0])
        original_domain = np.array([-1, 1])
    else:
        x = np.asarray(x_data)
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1, 1]))
    if weights is not None:
        w = np.asarray(weights).copy()
    else:
        w = np.ones(y.shape[0])

    output = [y, x, w, original_domain]
    if return_vander:
        vander_output = _get_vander(x, poly_order, np.sqrt(w), return_pinv)
        if return_pinv:
            output.extend(vander_output)
        else:
            output.append(vander_output)

    return output


def relative_difference(old, new, norm_order=None):
    """
    Calculates the relative difference (norm(new-old) / norm(old)) of two values.

    Used as an exit criteria in many baseline algorithms.

    Parameters
    ----------
    old : numpy.ndarray or float
        The array or single value from the previous iteration.
    new : numpy.ndarray or float
        The array or single value from the current iteration.
    norm_order : int, optional
        The type of norm to calculate. Default is None, which is l2
        norm for arrays, abs for scalars.

    Returns
    -------
    float
        The relative difference between the old and new values.

    """
    numerator = np.linalg.norm(new - old, norm_order)
    denominator = np.maximum(np.linalg.norm(old, norm_order), _MIN_FLOAT)
    return numerator / denominator


def gaussian(x, height=1.0, center=0.0, sigma=1.0):
    """
    Generates a gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center : float, optional
        The center of the distribution. Default is 0.0.
    sigma : float, optional
        The standard deviation of the distribution. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The gaussian distribution evaluated with x.

    """
    return height * np.exp(-0.5 * ((x - center)**2) / sigma**2)


def gaussian_kernel(window_size, sigma=1.0):
    """
    Creates an area-normalized gaussian kernel for convolution.

    Parameters
    ----------
    window_size : int
        The number of points for the entire kernel.
    sigma : float, optional
        The standard deviation of the gaussian model.

    Returns
    -------
    numpy.ndarray, shape (window_size,)
        The area-normalized gaussian kernel.

    Notes
    -----
    Return gaus/sum(gaus) rather than creating a unit-area gaussian
    since the unit-area gaussian would have an area smaller than 1
    for window_size < ~ 6 * sigma.

    """
    # centers distribution from -half_window to half_window
    x = np.arange(0, window_size) - (window_size - 1) / 2
    gaus = gaussian(x, 1, 0, sigma)
    return gaus / np.sum(gaus)


def mollify(data, kernel):
    """
    Smooths data by convolving with a area-normalized kernal.

    Parameters
    ----------
    data : numpy.ndarray
        The data to smooth.
    kernel : numpy.ndarray
        A pre-computed, normalized kernel for the mollification. Indices should
        span from -half_window to half_window.

    Returns
    -------
    numpy.ndarray
        The smoothed input array.

    Notes
    -----
    Mirrors the data near the edges so that convolution does not
    produce edge effects.

    """
    pad = (min(data.shape[0], kernel.shape[0]) // 2) + 1
    convolution = np.convolve(
        np.concatenate((data[pad:1:-1], data, data[-1:-pad:-1])),
        kernel, mode='valid'
    )
    return convolution
