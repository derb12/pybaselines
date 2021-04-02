# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on March 31, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags

from .utils import pad_edges


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
    pseudo_inverse : numpy.ndarray
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
        pseudo_inverse = np.linalg.pinv(weights[:, np.newaxis] * vander)
    else:
        pseudo_inverse = np.linalg.pinv(vander)

    return vander, pseudo_inverse


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
    pseudo_inverse : numpy.ndarray
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


def _setup_window(data, half_window, **pad_kwargs):
    """
    Sets the starting parameters for doing window-based algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the moving window functions. Used
        to pad the left and right edges of the data to reduce edge
        effects.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    numpy.ndarray, shape (N + 2 * half_window)
        [description]
    """
    return pad_edges(np.asarray(data), half_window, **pad_kwargs)
