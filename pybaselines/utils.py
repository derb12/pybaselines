# -*- coding: utf-8 -*-
"""Helper functions for pybaselines.

Created on March 5, 2021
@author: Donald Erb

"""

import numpy as np


try:
    from pentapy import solve as _pentapy_solve
    _HAS_PENTAPY = True
except ImportError:
    _HAS_PENTAPY = False

    def _pentapy_solve(*args, **kwargs):
        """Dummy function in case pentapy is not installed."""
        raise NotImplementedError('must have pentapy installed to use its solver')


# Note: the triple quotes are for including the attributes within the documentation
PENTAPY_SOLVER = 2
"""An integer designating the solver to use if pentapy is installed.
pentapy's solver can be used for solving pentadiagonal linear systems, such
as those used for the Whittaker-smoothing-based algorithms. Should be 2 (default)
or 1. See :func:`pentapy.core.solve` for more details.
"""

PERMC_SPEC = None
"""A deprecated constant used in previous versions. Will be removed in v0.6.0."""

# the minimum positive float values such that a + _MIN_FLOAT != a
_MIN_FLOAT = np.finfo(float).eps


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


def _get_edges(data, pad_length, mode='extrapolate', extrapolate_window=None, **pad_kwargs):
    """
    Provides the left and right edges for padding data.

    Parameters
    ----------
    data : array-like
        The array of the data.
    pad_length : int
        The number of points to add to the left and right edges.
    mode : str, optional
        The method for padding. Default is 'extrapolate'. Any method other than
        'extrapolate' will use numpy.pad.
    extrapolate_window : int, optional
        The number of values to use for linear fitting on the left and right
        edges. Default is None, which will set the extrapolate window size equal
        to the `half_window` size.
    **pad_kwargs
        Any keyword arguments to pass to numpy.pad, which will be used if `mode`
        is not 'extrapolate'.

    Returns
    -------
    left_edge : numpy.ndarray, shape(pad_length,)
        The array of data for the left padding.
    right_edge : numpy.ndarray, shape(pad_length,)
        The array of data for the right padding.

    Notes
    -----
    If mode is 'extrapolate', then the left and right edges will be fit with
    a first order polynomial and then extrapolated. Otherwise, uses numpy.pad.

    """
    y = np.asarray(data)
    if pad_length == 0:
        return y

    mode = mode.lower()
    if mode == 'extrapolate':
        if extrapolate_window is None:
            extrapolate_window = 2 * pad_length + 1
        x = np.arange(-pad_length, y.shape[0] + pad_length)
        left_poly = np.polynomial.Polynomial.fit(
            x[pad_length:-pad_length][:extrapolate_window],
            y[:extrapolate_window], 1
        )
        right_poly = np.polynomial.Polynomial.fit(
            x[pad_length:-pad_length][-extrapolate_window:],
            y[-extrapolate_window:], 1
        )

        left_edge = left_poly(x[:pad_length])
        right_edge = right_poly(x[-pad_length:])
    else:
        padded_data = np.pad(y, pad_length, mode, **pad_kwargs)
        left_edge = padded_data[:pad_length]
        right_edge = padded_data[-pad_length:]

    return left_edge, right_edge


def pad_edges(data, pad_length, mode='extrapolate',
              extrapolate_window=None, **pad_kwargs):
    """
    Adds left and right edges to the data.

    Parameters
    ----------
    data : array-like
        The array of the data.
    pad_length : int
        The number of points to add to the left and right edges.
    mode : str, optional
        The method for padding. Default is 'extrapolate'. Any method other than
        'extrapolate' will use numpy.pad.
    extrapolate_window : int, optional
        The number of values to use for linear fitting on the left and right
        edges. Default is None, which will set the extrapolate window size equal
        to the `half_window` size.
    **pad_kwargs
        Any keyword arguments to pass to numpy.pad, which will be used if `mode`
        is not 'extrapolate'.

    Returns
    -------
    padded_data : numpy.ndarray, shape (N + 2 * half_window,)
        The data with padding on the left and right edges.

    Notes
    -----
    If mode is 'extrapolate', then the left and right edges will be fit with
    a first order polynomial and then extrapolated. Otherwise, uses numpy.pad.

    """
    y = np.asarray(data)
    if pad_length == 0:
        return y

    if mode.lower() == 'extrapolate':
        left_edge, right_edge = _get_edges(y, pad_length, mode, extrapolate_window)
        padded_data = np.concatenate((left_edge, y, right_edge))
    else:
        padded_data = np.pad(y, pad_length, mode.lower(), **pad_kwargs)

    return padded_data


def padded_convolve(data, kernel, mode='reflect', **pad_kwargs):
    """
    Pads data before convolving to reduce edge effects.

    Parameters
    ----------
    data : numpy.ndarray, shape (N,)
        The data to smooth.
    kernel : numpy.ndarray, shape (M,)
        A pre-computed, normalized kernel for the convolution. Indices should
        span from -half_window to half_window.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The smoothed input array.

    """
    #TODO need to revisit this and ensure everything is correct
    padding = min(data.shape[0], kernel.shape[0]) // 2
    convolution = np.convolve(
        pad_edges(data, padding, mode, **pad_kwargs), kernel, mode='valid'
    )
    return convolution


def _safe_std(array, **kwargs):
    """
    Calculates the standard deviation and protects against nan and 0.

    Used to prevent propogating nan or dividing by 0.

    Parameters
    ----------
    array : numpy.ndarray
        The array of values for calculating the standard deviation.
    **kwargs
        Additional keyword arguments to pass to :func:`numpy.std`.

    Returns
    -------
    std : float
        The standard deviation of the array, or `_MIN_FLOAT` if the
        calculated standard deviation was 0 or if `array` was empty.

    Notes
    -----
    Does not protect against the calculated standard deviation of a non-empty
    array being nan because that would indicate that nan or inf was within the
    array, which should not be protected.

    """
    if array.size < 2:  # std would be 0 for an array with size of 1
        std = _MIN_FLOAT
    else:
        std = np.std(array, **kwargs)
        if std == 0:
            std = _MIN_FLOAT

    return std
