# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on March 31, 2021
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import grey_opening
from scipy.sparse import diags

from .utils import pad_edges, relative_difference


def difference_matrix(data_size, diff_order=2):
    """
    Creates an n-order differential matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : int, optional
        The integer differential order; must be >= 0. Default is 2.

    Returns
    -------
    scipy.sparse.dia.dia_matrix
        The sparse diagonal matrix of the differential.

    Raises
    ------
    ValueError
        Raised if diff_order is negative.

    Notes
    -----
    Most baseline algorithms use 2nd order differential matrices when
    doing penalized least squared fitting.

    The resulting matrices are transposes of the result of
    np.diff(np.eye(data_size), diff_order). This implementation allows using
    the differential matrices are they are written in various publications,
    ie. D.T * D rather than having to do D * D.T.

    """
    if diff_order < 0:
        raise ValueError('the differential order must be >= 0')
    if diff_order > data_size:
        # do not issue warning or exception to maintain parity with np.diff
        diff_order = data_size

    diagonals = np.zeros(2 * diff_order + 1)
    diagonals[diff_order] = 1
    for _ in range(diff_order):
        diagonals = diagonals[:-1] - diagonals[1:]

    return diags(diagonals, np.arange(diff_order + 1), shape=(data_size - diff_order, data_size))


def _yx_arrays(data, x_data=None, x_min=-1., x_max=1.):
    """
    Converts input data into numpy arrays and provides x data if none is given.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1. to 1. with N points.
    x_min : float, optional
        The minimum x-value if `x_data` is None. Default is -1.
    x_max : float, optional
        The maximum x-value if `x_data` is None. Default is 1.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        A numpy array of the y-values of the measured data.
    x : numpy.ndarray, shape (N,)
        A numpy array of the x-values of the measured data, or a created array.

    Notes
    -----
    Does not change the scale/domain of the input `x_data` if it is given, only
    converts it to an array.

    """
    y = np.asarray(data)
    if x_data is None:
        x = np.linspace(x_min, x_max, y.shape[0])
    else:
        x = np.asarray(x_data)

    return y, x


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
    diff_order : int, optional
        The integer differential order; must be greater than 0. Default is 2.
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

    Raises
    ------
    ValueError
        Raised is `diff_order` is less than 1.

    Warns
    -----
    UserWarning
        Raised if `diff_order` is greater than 3.

    """
    y = np.asarray(data)
    if diff_order < 1:
        raise ValueError(
            'the differential order must be > 0 for Whittaker-smoothing-based methods'
        )
    elif diff_order > 3:
        warnings.warn((
            'differential orders greater than 3 can have numerical issues;'
            ' consider using a differential order of 2 or 1 instead'
        ))
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
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the polynomial.
    pseudo_inverse : numpy.ndarray, shape (poly_order + 1, N)
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
        the domain [-1., 1.].
    weight_array : numpy.ndarray, shape (N,)
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
    to [-1., 1.] to improve the numerical stability of calculations; since the
    Vandermonde matrix goes from x^0 to x^poly_order, large values of x would
    otherwise cause difficulty when doing least squares minimization.

    """
    y, x = _yx_arrays(data, x_data)
    if x_data is None:
        original_domain = np.array([-1., 1.])
    else:
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1., 1.]))
    if weights is not None:
        weight_array = np.asarray(weights).copy()
    else:
        weight_array = np.ones(y.shape[0])

    output = [y, x, weight_array, original_domain]
    if return_vander:
        vander_output = _get_vander(x, poly_order, np.sqrt(weight_array), return_pinv)
        if return_pinv:
            output.extend(vander_output)
        else:
            output.append(vander_output)

    return output


def _optimize_window(data, increment=1, max_hits=3, window_tol=1e-6,
                     max_half_window=None, min_half_window=None):
    """
    Optimizes the morphological half-window size.

    Parameters
    ----------
    data : array-like, shape (N,)
        The measured data values.
    increment : int, optional
        The step size for iterating half windows. Default is 1.
    max_hits : int, optional
        The number of consecutive half windows that must produce the same
        morphological opening before accepting the half window as the optimum
        value. Default is 3.
    window_tol : float, optional
        The tolerance value for considering two morphological openings as
        equivalent. Default is 1e-6.
    max_half_window : int, optional
        The maximum allowable half-window size. If None (default), will be set
        to (len(data) - 1) / 2.
    min_half_window : int, optional
        The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    half_window : int
        The optimized half window size.

    Notes
    -----
    May only provide good results for some morphological algorithms, so use with
    caution.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

    """
    y = np.asarray(data)
    if max_half_window is None:
        max_half_window = (y.shape[0] - 1) // 2
    if min_half_window is None:
        min_half_window = 1

    opening = grey_opening(y, [2 * min_half_window + 1])
    hits = 0
    best_half_window = min_half_window
    for half_window in range(min_half_window + increment, max_half_window, increment):
        new_opening = grey_opening(y, [half_window * 2 + 1])
        if relative_difference(opening, new_opening) < window_tol:
            if hits == 0:
                # keep just the first window that fits tolerance
                best_half_window = half_window - increment
            hits += 1
            if hits >= max_hits:
                half_window = best_half_window
                break
        elif hits:
            hits = 0
        opening = new_opening

    return half_window


def _setup_morphology(data, half_window=None, **window_kwargs):
    """
    Sets the starting parameters for morphology-based methods.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using pybaselines.morphological.optimize_window.
    **window_kwargs
        Keyword arguments to pass to :func:`.optimize_window`.
        Possible items are:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 3.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable half-window size. If None (default), will be
                set to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    output_half_window : int
        The accepted half window size.

    Notes
    -----
    Ensures that window size is odd since morphological operations operate in
    the range [-output_half_window, ..., output_half_window].

    Half windows are dealt with rather than full window sizes to clarify their
    usage. SciPy morphology operations deal with full window sizes.

    """
    y = np.asarray(data)
    if half_window is not None:
        output_half_window = half_window
    else:
        output_half_window = _optimize_window(y, **window_kwargs)

    return y, output_half_window


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
        The padded array of data.

    """
    return pad_edges(data, half_window, **pad_kwargs)
