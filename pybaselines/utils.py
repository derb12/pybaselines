# -*- coding: utf-8 -*-
"""Helper functions for pybaselines.

Parameters
----------
MIN_FLOAT : float
    The minimum positive float value.

Created on March 5, 2021

@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags


MIN_FLOAT = np.finfo(float).eps


def difference_matrix(data_size, order=2):
    """
    Creates an n-order differential matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    order : {2, 1, 3, 4, 5}, optional
        The integer differential order; either 1, 2, 3, 4, or 5. Default is 2.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        The sparse diagonal matrix of the differential.

    Raises
    ------
    ValueError
        Raised if order is not 1, 2, 3, 4, or 5.

    Notes
    -----
    Most baseline algorithms use 2nd order differential matrices when
    doing penalized least squared fitting.

    It would be possible to support any differential order by doing
    np.diff(np.eye(data_size), order), but the resulting matrix could
    cause issues if data_size is large. Therefore, it's better to only
    provide sparse arrays for the most commonly used differential orders.

    The resulting matrices are transposes of the result of
    np.diff(np.eye(data_size), order). Not sure why there is a discrepancy,
    but this implementation allows using the differential matrices are they
    are written in various publications, ie. D.T * D rather than having to
    do D * D.T like most code, such as those adapted from stack overflow:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    """
    if order not in (1, 2, 3, 4, 5):
        raise ValueError('The differential order must be 1, 2, 3, 4, or 5')
    diagonals = {
        1: [-1, 1],
        2: [1, -2, 1],
        3: [-1, 3, -3, 1],
        4: [1, -4, 6, -4, 1],
        5: [-1, 5, -10, 10, -5, 1]
    }[order]

    return diags(diagonals, list(range(order + 1)), shape=(data_size - order, data_size))


def _setup_pls(data_size, lam, order=2, weights=None):
    """
    Sets the starting parameters for doing penalized least squares.

    Parameters
    ----------
    data_size : int
        The number of data points.
    lam : float
        The smoothing parameter, lambda. Typical values are between 10 and
        1e8, but it strongly depends on the penalized least square method
        and the differential order.
    order : {2, 1, 3, 4, 5}, optional
        The integer differential order; either 1, 2, 3, 4, or 5. Default is 2.
    weights : np.ndarray, optional
        The weighting array. If None (default), then will be an array with
        size equal to data_size and all values set to 1.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        The product of lam * D.T * D, where D is the sparse diagonal matrix of
        the differential, and D.T is the transpose of D.
    scipy.sparse.dia.dia_matrix
        The sparse weight matrix with the weighting array as the diagonal values.
    weight_array : np.ndarray
        The weighting array.

    """
    diff_matrix = difference_matrix(data_size, order)
    if weights is None:
        weight_array = np.ones(data_size)
    else:
        weight_array = weights
    return lam * diff_matrix.T * diff_matrix, diags(weight_array), weight_array


def _get_vander(x, poly_order=2, weights=None):
    """
    Calculates the Vandermonde matrix and its pseudo-inverse.

    Parameters
    ----------
    x : [type]
        [description]
    poly_order : int, optional
        [description]. Default is 2.
    weights : np.ndarray, optional


    Returns
    -------
    vander : np.ndarray
        The Vandermonde matrix.
    vander_pinv : np.ndarray
        The pseudo-inverse of the Vandermonde, with weights applied if input.
        Calculated using singular value decomposition (SVD).

    """
    vander = np.polynomial.polynomial.polyvander(x, poly_order)
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
    data : [type]
        [description]
    x_data : [type], optional
        [description]. Default is None.
    weights : [type], optional
        [description]. Default is None.
    poly_order : int, optional
        [description]. Default is 2.
    return_vander : bool, optional
        [description]. Default is False.
    return_pinv : bool, optional
        [description]. Default is False.

    Returns
    -------
    y : np.ndarray

    x : np.ndarray

    w : np.ndarray

    original_domain : np.ndarray, shape (2,)
        The minimum and maximum values of the original x_data values. Can
        be used to convert the coefficents found during least squares
        minimization using the normalized x into usable polynomial coefficients
        for the original x_data.
    vander : np.ndarray
        Only returned if return_vander is True. The Vandermonde matrix for the
        normalized x values.
    vander_pinv : np.ndarray
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
        vander, vander_pinv = _get_vander(x, poly_order, np.sqrt(w))
        output.append(vander)
        if return_pinv:
            output.append(vander_pinv)

    return output


def relative_difference(old, new):
    """
    Calculates the relative difference (norm(new-old) / norm(old)) of two values.

    Used as an exit criteria in many baseline algorithms.

    Parameters
    ----------
    old : np.ndarray or float
        The array or single value from the previous iteration.
    new : np.ndarray or float
        The array or single value from the current iteration.

    Returns
    -------
    float
        The relative difference between the old and new values.

    """
    return np.linalg.norm(new - old) / max(np.linalg.norm(old), MIN_FLOAT)


def gaussian(x, height=1.0, center=0.0, sigma=1.0):
    """
    Generates a gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : np.ndarray
        The x-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center : float, optional
        The center of the distribution. Default is 0.0.
    sigma : float, optional
        The standard deviation of the distribution. Default is 1.0.

    Returns
    -------
    np.ndarray
        The gaussian distribution evaluated with x.

    """
    return height * np.exp(-0.5 * ((x - center) / sigma)**2)


def gaussian_kernel(window_size, sigma=1.0):
    """
    Creates an area-normalized gaussian kernel for convolution.

    Parameters
    ----------
    window_size : int
        [description]
    sigma : float, optional
        [description]

    Returns
    -------
    np.ndarray
        The normalized gaussian kernel with size=window_size.

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
    data : np.ndarray
        The data to smooth.
    kernel : np.ndarray
        A pre-computed, normalized kernel for the mollification. Indices should
        span from -half_window to half_window.

    Returns
    -------
    np.ndarray
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
