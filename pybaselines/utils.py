# -*- coding: utf-8 -*-
"""Helper functions for pybaselines.

Created on March 5, 2021
@author: Donald Erb

"""

from math import ceil

import numpy as np
from scipy.ndimage import grey_opening
from scipy.signal import convolve
from scipy.special import binom

from ._banded_utils import PenalizedSystem
from ._banded_utils import difference_matrix as _difference_matrix
from ._compat import jit
from ._spline_utils import PSpline
from ._validation import (
    _check_array, _check_optional_array, _check_scalar, _get_row_col_values, _yx_arrays
)


# the minimum positive float values such that a + _MIN_FLOAT != a
# TODO this is mostly used to prevent dividing by 0; is there a better way to do that?
# especially since it is usually max(value, _MIN_FLOAT) and in some cases value could be
# < _MIN_FLOAT but still > 0 and useful; think about it
_MIN_FLOAT = np.finfo(float).eps


class ParameterWarning(UserWarning):
    """
    Warning issued when a parameter value is outside of the recommended range.

    For cases where a parameter value is valid and will not cause errors, but is
    outside of the recommended range of values and as a result may cause issues
    such as numerical instability that would otherwise be hard to diagnose.
    """


def relative_difference(old, new, norm_order=None):
    """
    Calculates the relative difference, ``(norm(new-old) / norm(old))``, of two values.

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
    Generates a Gaussian distribution based on height, center, and sigma.

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
        The Gaussian distribution evaluated with x.

    Raises
    ------
    ValueError
        Raised if `sigma` is not greater than 0.

    """
    if sigma <= 0:
        raise ValueError('sigma must be greater than 0')
    return height * np.exp(-0.5 * ((x - center)**2) / sigma**2)


def gaussian2d(x, z, height=1.0, center_x=0.0, center_z=0.0, sigma_x=1.0, sigma_z=1.0):
    """
    Generates a Gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : numpy.ndarray, shape (M, N)
        The x-values at which to evaluate the distribution.
    z : numpy.ndarray, shape (M, N)
        The z-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center_x : float, optional
        The center of the distribution in the x-axis. Default is 0.0.
    sigma_x : float, optional
        The standard deviation of the distribution in the x-axis. Default is 1.0.
    center_z : float, optional
        The center of the distribution in the z-axis. Default is 0.0.
    sigma_z : float, optional
        The standard deviation of the distribution in the z-axis. Default is 1.0.

    Returns
    -------
    numpy.ndarray, shape (M, N)
        The Gaussian distribution evaluated with x and z.

    Raises
    ------
    ValueError
        Raised if the input `x` or `z` are not two dimensional.

    Notes
    -----
    The input `x` and `z` should be two dimensional arrays, which can be gotten
    from their one dimensional counterparts by using :func:`numpy.meshgrid`.

    """
    if x.ndim != 2 or z.ndim != 2:
        raise ValueError('x and z should be two dimensional')
    return height * gaussian(x, 1, center_x, sigma_x) * gaussian(z, 1, center_z, sigma_z)


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
    window_size = max(1, window_size)
    x = np.arange(window_size) - (window_size - 1) / 2
    gaus = gaussian(x, 1, 0, sigma)
    return gaus / np.sum(gaus)


def _mollifier_kernel(window_size):
    """
    A kernel for smoothing/mollification.

    Parameters
    ----------
    window_size : int
        The number of points for the entire kernel.

    Returns
    -------
    numpy.ndarray, shape (2 * window_size + 1,)
        The area normalized kernel.

    References
    ----------
    Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

    """
    x = (np.arange(0, 2 * window_size + 1) - window_size) / window_size
    kernel = np.zeros_like(x)
    # x[1:-1] is same as x[abs(x) < 1]
    kernel[1:-1] = np.exp(-1 / (1 - (x[1:-1])**2))
    return kernel / kernel.sum()


def _get_edges(data, pad_length, mode='extrapolate', extrapolate_window=None, **pad_kwargs):
    """
    Provides the left and right edges for padding data.

    Parameters
    ----------
    data : array-like
        The array of the data.
    pad_length : int
        The number of points to add to the left and right edges.
    mode : str or Callable, optional
        The method for padding. Default is 'extrapolate'. Any method other than
        'extrapolate' will use numpy.pad.
    extrapolate_window : int, optional
        The number of values to use for linear fitting on the left and right
        edges. Default is None, which will set the extrapolate window size equal
        to `pad_length`.
    **pad_kwargs
        Any keyword arguments to pass to numpy.pad, which will be used if `mode`
        is not 'extrapolate'.

    Returns
    -------
    left_edge : numpy.ndarray, shape(pad_length,)
        The array of data for the left padding.
    right_edge : numpy.ndarray, shape(pad_length,)
        The array of data for the right padding.

    Raises
    ------
    ValueError
        Raised if `pad_length` is < 0, or if `extrapolate_window` is <= 0 and
        `mode` is `extrapolate`.

    Notes
    -----
    If mode is 'extrapolate', then the left and right edges will be fit with
    a first order polynomial and then extrapolated. Otherwise, uses :func:`numpy.pad`.

    """
    y = np.asarray(data)
    if pad_length == 0:
        return np.array([]), np.array([])
    elif pad_length < 0:
        raise ValueError('pad length must be greater or equal to 0')

    if isinstance(mode, str):
        mode = mode.lower()
    if mode == 'extrapolate':
        if extrapolate_window is None:
            extrapolate_window = pad_length
        extrapolate_windows = _check_scalar(extrapolate_window, 2, True, dtype=int)[0]

        if np.any(extrapolate_windows <= 0):
            raise ValueError('extrapolate_window must be greater than 0')
        left_edge = np.empty(pad_length)
        right_edge = np.empty(pad_length)
        # use x[pad_length:-pad_length] for fitting to ensure x and y are
        # same shape regardless of extrapolate window value
        x = np.arange(len(y) + 2 * pad_length)
        for i, array in enumerate((left_edge, right_edge)):
            extrapolate_window_i = extrapolate_windows[i]
            if extrapolate_window_i == 1:
                # just use the edges rather than trying to fit a line
                array[:] = y[0] if i == 0 else y[-1]
            elif i == 0:
                poly = np.polynomial.Polynomial.fit(
                    x[pad_length:-pad_length][:extrapolate_window_i],
                    y[:extrapolate_window_i], 1
                )
                array[:] = poly(x[:pad_length])
            else:
                poly = np.polynomial.Polynomial.fit(
                    x[pad_length:-pad_length][-extrapolate_window_i:],
                    y[-extrapolate_window_i:], 1
                )
                array[:] = poly(x[-pad_length:])
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
    mode : str or Callable, optional
        The method for padding. Default is 'extrapolate'. Any method other than
        'extrapolate' will use :func:`numpy.pad`.
    extrapolate_window : int, optional
        The number of values to use for linear fitting on the left and right
        edges. Default is None, which will set the extrapolate window size equal
        to `pad_length`.
    **pad_kwargs
        Any keyword arguments to pass to :func:`numpy.pad`, which will be used if `mode`
        is not 'extrapolate'.

    Returns
    -------
    padded_data : numpy.ndarray, shape (N + 2 * half_window,)
        The data with padding on the left and right edges.

    Notes
    -----
    If mode is 'extrapolate', then the left and right edges will be fit with
    a first order polynomial and then extrapolated. Otherwise, uses :func:`numpy.pad`.

    """
    y = np.asarray(data)
    if pad_length == 0:
        return y

    if isinstance(mode, str):
        mode = mode.lower()
    if mode == 'extrapolate':
        left_edge, right_edge = _get_edges(y, pad_length, mode, extrapolate_window)
        padded_data = np.concatenate((left_edge, y, right_edge))
    else:
        padded_data = np.pad(y, pad_length, mode, **pad_kwargs)

    return padded_data


def _extrapolate2d(y, total_padding, extrapolate_window=None):
    """
    Extrapolates each edge of two dimensional data.

    Corners are calculated by averaging linear fits of the extended data.

    Parameters
    ----------
    y : numpy.ndarray
        _description_
    total_padding : Sequence[int, int, int, int]
        The padding for the top, bottom, left, and right. The padding of top and
        bottom are assumed to be equal, as are the left and right.
    extrapolate_window : int or Sequence[int, int] or Sequence[int, int, int, int], optional
        The number of values to use for linear fitting on the top, bottom, left, and right
        edges. Default is None, which will set the extrapolate window size equal
        to `total_padding`.

    Returns
    -------
    output : numpy.ndarray
        The data with padding

    Raises
    ------
    NotImplementedError
        Raised if any value in `total_padding` is zero.
    ValueError
        Raised if any extrapolation window is less than 1.

    Notes
    -----
    Uses the Moore-Penrose pseudo-inverse to speed up the calculation of the linear fits
    for each edge. Using the Vandermonde with `numpy.linalg.lstsq` would also work but is
    a little slower.

    """
    if np.equal(total_padding, 0).any():
        raise NotImplementedError('pad length of 0 is not supported in 2D')
    elif np.less(total_padding, 0).any():
        raise ValueError('pad length must be greater or equal to 0')

    if extrapolate_window is None:
        extrapolate_windows = total_padding
    else:
        extrapolate_windows = _get_row_col_values(extrapolate_window).reshape((2, 2))

    if np.less_equal(extrapolate_windows, 0).any():
        raise ValueError('extrapolate_window must be greater than 0')
    # pad length for left and right or top and bottom should be equal, so ignore the repeats
    total_padding = [total_padding[0][0], total_padding[1][0]]

    output = np.empty(
        (y.shape[0] + total_padding[0] * 2, y.shape[1] + total_padding[1] * 2)
    )
    output[total_padding[0]:-total_padding[0], total_padding[1]:-total_padding[1]] = y

    x = np.arange(y.shape[0] + 2 * total_padding[0])
    z = np.arange(y.shape[1] + 2 * total_padding[1])

    vander_x = np.polynomial.polynomial.polyvander(x, 1)
    vander_z = np.polynomial.polynomial.polyvander(z, 1)
    pinv_top = np.linalg.pinv(
        vander_x[total_padding[0]:-total_padding[0]][:extrapolate_windows[0][0]]
    )
    pinv_bottom = np.linalg.pinv(
        vander_x[total_padding[0]:-total_padding[0]][-extrapolate_windows[0][1]:]
    )
    pinv_left = np.linalg.pinv(
        vander_z[total_padding[1]:-total_padding[1]][:extrapolate_windows[1][0]]
    )
    pinv_right = np.linalg.pinv(
        vander_z[total_padding[1]:-total_padding[1]][-extrapolate_windows[1][1]:]
    )

    top = vander_x[:total_padding[0]] @ (pinv_top @ y[:extrapolate_windows[0][0]])
    bottom = vander_x[-total_padding[0]:] @ (pinv_bottom @ y[-extrapolate_windows[0][1]:])

    output[:total_padding[0], total_padding[1]:-total_padding[1]] = top
    output[-total_padding[0]:, total_padding[1]:-total_padding[1]] = bottom

    left = vander_z[:total_padding[1]] @ (pinv_left @ y[:, :extrapolate_windows[1][0]].T)
    right = vander_z[-total_padding[1]:] @ (pinv_right @ y[:, -extrapolate_windows[1][1]:].T)

    output[total_padding[0]:-total_padding[0], :total_padding[1]] = left.T
    output[total_padding[0]:-total_padding[0], -total_padding[1]:] = right.T

    # now fill the corners by averaging the extensions of the corners
    top_left = vander_z[:total_padding[1]] @ (
        pinv_left @ output[
            :total_padding[0], total_padding[1]:-total_padding[1]
        ][:, :extrapolate_windows[1][0]].T
    )
    top_right = vander_z[-total_padding[1]:] @ (
        pinv_right @ output[
            :total_padding[0], total_padding[1]:-total_padding[1]
        ][:, -extrapolate_windows[1][1]:].T
    )

    bottom_left = vander_z[:total_padding[1]] @ (
        pinv_left @ output[
            -total_padding[0]:, total_padding[1]:-total_padding[1]
        ][:, :extrapolate_windows[1][0]].T
    )
    bottom_right = vander_z[-total_padding[1]:] @ (
        pinv_right @ output[
            -total_padding[0]:, total_padding[1]:-total_padding[1]
        ][:, -extrapolate_windows[1][1]:].T
    )

    left_top = vander_x[:total_padding[0]] @ (
        pinv_top @ output[
            total_padding[0]:-total_padding[0], :total_padding[1]
        ][:extrapolate_windows[0][0]]
    )
    left_bottom = vander_x[-total_padding[0]:] @ (
        pinv_bottom @ output[
            total_padding[0]:-total_padding[0], :total_padding[1]:
        ][-extrapolate_windows[0][1]:]
    )

    right_top = vander_x[:total_padding[0]] @ (
        pinv_top @ output[
            total_padding[0]:-total_padding[0], -total_padding[1]:
        ][:extrapolate_windows[0][0]]
    )
    right_bottom = vander_x[-total_padding[0]:] @ (
        pinv_bottom @ output[
            total_padding[0]:-total_padding[0], -total_padding[1]:
        ][-extrapolate_windows[0][1]:]
    )

    output[:total_padding[0], :total_padding[1]] = 0.5 * (top_left.T + left_top)
    output[:total_padding[0], -total_padding[1]:] = 0.5 * (top_right.T + right_top)
    output[-total_padding[0]:, :total_padding[1]] = 0.5 * (bottom_left.T + left_bottom)
    output[-total_padding[0]:, -total_padding[1]:] = 0.5 * (bottom_right.T + right_bottom)

    return output


def pad_edges2d(data, pad_length, mode='edge', extrapolate_window=None, **pad_kwargs):
    """
    Adds left, right, top, and bottom edges to the data.

    Parameters
    ----------
    data : array-like, shape (M, N)
        The 2D array of the data.
    pad_length : int or Sequence[int, int]
        The number of points to add to the top, bottom, left, and right edges. If a single
        value is given, all edges have the same padding. If a sequence of two values is
        given, the first value will be the padding on the top and bottom (rows), and the second
        value will pad the left and right (columns).
    mode : str or Callable, optional
        The method for padding. Default is 'edge'. Any method other than
        'extrapolate' will use :func:`numpy.pad`.
    extrapolate_window : int or Sequence[int, int] or Sequence[int, int, int, int], optional
        The number of values to use for linear fitting on the top, bottom, left, and right
        edges. Default is None, which will set the extrapolate window size equal
        to `pad_length`.
    **pad_kwargs
        Any keyword arguments to pass to :func:`numpy.pad`, which will be used if `mode`
        is not 'extrapolate'.

    Returns
    -------
    padded_data : numpy.ndarray
        The data with padding on the top, bottom, left, and right edges.

    Notes
    -----
    If mode is 'extrapolate', then each edge will be extended by linear fits along each
    row and column, and the corners are calculated by averaging the linear sections.

    """
    y = np.asarray(data)
    if y.ndim != 2:
        raise ValueError('input data must be two dimensional')
    total_padding = _get_row_col_values(pad_length).reshape((2, 2))

    if isinstance(mode, str):
        mode = mode.lower()
    if mode == 'extrapolate':
        output = _extrapolate2d(y, total_padding, extrapolate_window)
    else:
        output = np.pad(data, total_padding, mode=mode, **pad_kwargs)

    return output


def padded_convolve(data, kernel, mode='reflect', **pad_kwargs):
    """
    Pads data before convolving to reduce edge effects.

    Parameters
    ----------
    data : array-like, shape (N,)
        The data to convolve.
    kernel : array-like, shape (M,)
        The convolution kernel.
    mode : str or Callable, optional
        The method for padding to pass to :func:`.pad_edges`. Default is 'reflect'.
    **pad_kwargs
        Any additional keyword arguments to pass to :func:`.pad_edges`.

    Returns
    -------
    convolution : numpy.ndarray, shape (N,)
        The convolution output.

    """
    # TODO need to revisit this and ensure everything is correct
    # TODO look at using scipy.ndimage.convolve1d instead, or at least
    # comparing the output in tests; that function should have a similar usage
    padding = ceil(min(len(data), len(kernel)) / 2)
    convolution = convolve(
        pad_edges(data, padding, mode, **pad_kwargs), kernel, mode='same'
    )
    return convolution[padding:-padding]


@jit(nopython=True, cache=True)
def _interp_inplace(x, y, y_start, y_end):
    """
    Interpolates values inplace between the two ends of an array.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values for interpolation. All values are assumed to be valid.
    y : numpy.ndarray
        The y-values. The two endpoints, y[0] and y[-1] are assumed to be valid,
        and all values inbetween (ie. y[1:-1]) will be replaced by interpolation.
    y_start : float, optional
        The initial y-value for interpolation.
    y_end : float, optional
        The end y-value for interpolation.

    Returns
    -------
    y : numpy.ndarray
        The input `y` array, with the interpolation performed inplace.

    """
    y[1:-1] = y_start + (x[1:-1] - x[0]) * ((y_end - y_start) / (x[-1] - x[0]))

    return y


def _poly_transform_matrix(num_coefficients, original_domain):
    """
    Creates the matrix that transforms polynomial coefficents from one domain to another.

    The polynomial coefficient array `d` computed with `v` can be transformed to the
    coefficient array `c` computed with `x` where ``v = scale * x + offset`` by applying
    ``c = T @ d``, where `T` is the transformation matrix.

    Parameters
    ----------
    num_coefficients : int
        The number of polynomial coefficients, ie. the polynomial degree + 1.
    original_domain : Sequence[float, float]
        The domain, [min(x), max(x)], of the original data used for fitting.

    Returns
    -------
    transformation : numpy.ndarray, shape (`num_coefficients`, `num_coefficients`)
        The transformation matrix to convert domains.

    Notes
    -----
    The calculation of the transformation matrix is based on the math from
    https://stackoverflow.com/questions/141422/how-can-a-transform-a-polynomial-to-another-coordinate-system#comment57358951_142436.

    This function assumes the original coefficients were computed with the domain [-1, 1].

    """
    offset, scale = np.polynomial.polyutils.mapparms(np.array([-1., 1.]), original_domain)
    transformation = np.zeros((num_coefficients, num_coefficients))
    skip_offset = np.equal(offset, 0)  # 0 raised to negative powers causes nan
    for i in range(num_coefficients):
        for j in range(num_coefficients):
            if skip_offset:
                if j == i:
                    transformation[i, j] = binom(j, i) * (scale)**(-j)
            else:
                transformation[i, j] = binom(j, i) * (scale)**(-j) * (-offset)**(j - i)

    return transformation


def _convert_coef(coef, original_domain):
    """
    Scales the polynomial coefficients back to the original domain of the data.

    For fitting, the x-values are scaled from their original domain, [min(x),
    max(x)], to [-1, 1] in order to improve the numerical stability of fitting.
    This function rescales the retrieved polynomial coefficients for the fit
    x-values back to the original domain.

    Parameters
    ----------
    coef : numpy.ndarray, shape (a,)
        The array of coefficients for the polynomial. Should increase in
        order, for example (c0, c1, c2) from `y = c0 + c1 * x + c2 * x**2`.
    original_domain : Sequence[float, float]
        The domain, [min(x), max(x)], of the original data used for fitting.

    Returns
    -------
    numpy.ndarray, shape (a,)
        The array of coefficients scaled for the original domain.

    Notes
    -----
    Could slightly reduce computation time by computing offset and scale once within
    the _Algorithm object, but doing it this way with `original_domain` is backwards
    compatible and this function is probably not called enough to justify the change.

    """
    transformation = _poly_transform_matrix(coef.shape[0], original_domain)
    return transformation @ coef


def _convert_coef2d(coef, poly_degree_x, poly_degree_z, original_x_domain, original_z_domain):
    """
    Scales the polynomial coefficients back to the original domain of the data.

    For fitting, the x-values and z-values are scaled from their original domain,
    [min(x), max(x)] and [min(z), max(z)], to [-1, 1] in order to improve the numerical
    stability of fitting. This function rescales the retrieved polynomial coefficients
    for the fit x-values and z-values back to their original domains.

    Parameters
    ----------
    coef : numpy.ndarray, shape (``a * b``,)
        The 1d array of coefficients for the polynomial. Should increase in
        order. The shape should be (``a * b``,), where `a` is the polynomial degree + 1 for
        the x-values and `b` is the polynomial degree + 1 for the z-values.
    poly_degree_x : int
        The polynomial degree for the x-values
    poly_degree_z : int
        The polynomial degree for the z-values
    original_x_domain : Sequence[float, float]
        The domain, [min(x), max(x)], of the original x-values used for fitting.
    original_z_domain : Sequence[float, float]
        The domain, [min(z), max(z)], of the original z-values used for fitting.

    Returns
    -------
    numpy.ndarray, shape (a, b)
        The 2D array of coefficients scaled for the original domains.

    Notes
    -----
    Reshapes the coefficient array into the correct shape for use with
    :func:`numpy.polynomial.polynomial.polyval2d`.

    """
    x_order = poly_degree_x + 1
    z_order = poly_degree_z + 1
    transformation_x = _poly_transform_matrix(x_order, original_x_domain)
    transformation_z = _poly_transform_matrix(z_order, original_z_domain)

    return transformation_x @ coef.reshape((x_order, z_order)) @ transformation_z.T


def difference_matrix(data_size, diff_order=2, diff_format=None):
    """
    Creates an n-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : int, optional
        The integer differential order; must be >= 0. Default is 2.
    diff_format : str or None, optional
        The sparse format to use for the difference matrix. Default is None,
        which will use the default specified in :func:`scipy.sparse.diags`.

    Returns
    -------
    diff_matrix : scipy.sparse.spmatrix or scipy.sparse._sparray
        The sparse difference matrix.

    Raises
    ------
    ValueError
        Raised if `diff_order` or `data_size` is negative.

    Notes
    -----
    The resulting matrices are sparse versions of::

        import numpy as np
        np.diff(np.eye(data_size), diff_order, axis=0)

    This implementation allows using the differential matrices are they
    are written in various publications, ie. ``D.T @ D``.

    Most baseline algorithms use 2nd order differential matrices when
    doing penalized least squared fitting or Whittaker-smoothing-based fitting.

    """
    # difference_matrix moved to pybaselines._banded_utils in version 1.0.0 in order to more
    # easily use it in other modules without creating circular imports; this function
    # exposes it through pybaselines.utils for backwards compatibility in user code
    return _difference_matrix(data_size, diff_order=diff_order, diff_format=diff_format)


def optimize_window(data, increment=1, max_hits=3, window_tol=1e-6,
                    max_half_window=None, min_half_window=None):
    """
    Optimizes the morphological half-window size.

    Parameters
    ----------
    data : array-like
        The measured data values. Can be one or two dimensional.
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
    half_window : int or numpy.ndarray[int, int]
        The optimized half window size(s). If `data` is one dimensional, the
        output is a single integer, and if `data` is two dimensional, the output
        is an array of two integers.

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
        max_half_window = (y.shape[-1] - 1) // 2
    if min_half_window is None:
        min_half_window = 1

    y_dims = y.ndim
    # TODO would it be better to allow padding the data?
    opening = grey_opening(y, [2 * min_half_window + 1] * y_dims)
    hits = 0
    half_window = 1  # in case min_half_window is set incorrectly
    best_half_window = min_half_window
    for half_window in range(min_half_window + increment, max_half_window, increment):
        new_opening = grey_opening(y, [half_window * 2 + 1] * y_dims)
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

    if y_dims == 2:
        output = np.maximum([half_window, half_window], [1, 1])
    else:
        output = max(half_window, 1)  # ensure half window is at least 1
    return output


def _inverted_sort(sort_order):
    """
    Finds the indices that invert a sorting.

    Given an array `a`, and the indices that sort the array, `sort_order`, the
    inverted sort is defined such that it gives the original index order of `a`,
    ie. ``a == a[sort_order][inverted_order]``.

    Parameters
    ----------
    sort_order : numpy.ndarray, shape (N,)
        The original index array for sorting.

    Returns
    -------
    inverted_order : numpy.ndarray, shape (N,)
        The array that inverts the sort given by `sort_order`.

    Notes
    -----
    This function is equivalent to doing::

        inverted_order = sort_order.argsort()

    but is faster for large arrays since no additional sorting is performed.

    """
    num_points = len(sort_order)
    inverted_order = np.empty(num_points, dtype=np.intp)
    inverted_order[sort_order] = np.arange(num_points, dtype=np.intp)

    return inverted_order


def _determine_sorts(data):
    """
    Provides the arrays for sorting and inverting sorting, if needed.

    Parameters
    ----------
    data : numpy.ndarray, shape (N,)
        The array to potentially sort.

    Returns
    -------
    output : tuple(numpy.ndarray, numpy.ndarray) or tuple(None, None)
        A tuple of the index array for sorting the input array and the array
        that inverts that sorting. If the input array is already sorted, then
        the output will be (None, None).

    """
    sort_order = data.argsort(kind='mergesort')
    skip_sorting = (sort_order[1:] > sort_order[:-1]).all()
    if skip_sorting:
        output = (None, None)
    else:
        output = (sort_order, _inverted_sort(sort_order))

    return output


def _sort_array2d(array, sort_order=None):
    """
    Sorts the input 2D array only if given a non-None sorting order.

    Parameters
    ----------
    array : numpy.ndarray
        The array to sort. Must be two or three dimensional.
    sort_order : numpy.ndarray, optional
        The array(s) defining the sort order for the input array. Default is None, which
        will not sort the input.

    Returns
    -------
    output : numpy.ndarray
        The input array after optionally sorting.

    Notes
    -----
    For all inputs, assumes the last 2 axes correspond to the data that needs sorted.

    Raises
    ------
    ValueError
        Raised if the input array is not two or three dimensional.

    """
    if sort_order is None:
        output = array
    else:
        n_dims = array.ndim
        if n_dims == 2:
            output = array[sort_order]
        elif n_dims == 3:
            if isinstance(sort_order, tuple):
                if sort_order[0] is Ellipsis:
                    output = array[sort_order]
                else:
                    output = array[:, sort_order[0], sort_order[1]]
            else:
                output = array[:, sort_order, :]
        else:
            raise ValueError('too many dimensions to sort the data')

    return output


def _sort_array(array, sort_order=None):
    """
    Sorts the input array only if given a non-None sorting order.

    Parameters
    ----------
    array : numpy.ndarray
        The array to sort.
    sort_order : numpy.ndarray, optional
        The array defining the sort order for the input array. Default is None, which
        will not sort the input.

    Returns
    -------
    output : numpy.ndarray
        The input array after optionally sorting.

    Notes
    -----
    For all inputs, assumes the last axis corresponds to the data that needs sorted.

    Raises
    ------
    ValueError
        Raised if the input array has more than two dimensions.

    """
    if sort_order is None:
        output = array
    else:
        n_dims = array.ndim
        if n_dims == 1:
            output = array[sort_order]
        elif n_dims == 2:
            output = array[:, sort_order]
        else:
            raise ValueError('too many dimensions to sort the data')

    return output


def whittaker_smooth(data, lam=1e6, diff_order=2, weights=None, check_finite=True):
    """
    Smooths the input data using Whittaker smoothing.

    The input is smoothed by solving the equation ``(W + lam * D.T @ D) y_smooth = W @ y``,
    where `W` is a matrix with `weights` on the diagonals and `D` is the finite difference
    matrix.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    diff_order : int, optional
        The order of the finite difference matrix. Must be greater than or equal to 0.
        Default is 2 (second order differential matrix). Typical values are 2 or 1.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    check_finite : bool, optional
        If True, will raise an error if any values if `data` or `weights` are not finite.
        Default is False, which skips the check.

    Returns
    -------
    y_smooth : numpy.ndarray, shape (N,)
        The smoothed data.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    y = _check_array(data, check_finite=check_finite, ensure_1d=True)
    len_y = len(y)
    penalized_system = PenalizedSystem(len_y, lam=lam, diff_order=diff_order)
    weight_array = _check_optional_array(len_y, weights, check_finite=check_finite)

    y_smooth = penalized_system.solve(
        penalized_system.add_diagonal(weight_array),
        weight_array * y, overwrite_ab=True, overwrite_b=True
    )

    return y_smooth


def pspline_smooth(data, x_data=None, lam=1e1, num_knots=100, spline_degree=3, diff_order=2,
                   weights=None, check_finite=True):
    """
    Smooths the input data using Penalized Spline smoothing.

    The input is smoothed by solving the equation
    ``(B.T @ W @ B + lam * D.T @ D) y_smooth = B.T @ W @ y``, where `W` is a matrix with
    `weights` on the diagonals, `D` is the finite difference matrix, and `B` is the
    spline basis matrix.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e1.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the finite difference matrix. Must be greater than or equal to 0.
        Default is 2 (second order differential matrix). Typical values are 2 or 1.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    check_finite : bool, optional
        If True, will raise an error if any values if `data` or `weights` are not finite.
        Default is False, which skips the check.

    Returns
    -------
    y_smooth : numpy.ndarray, shape (N,)
        The smoothed data.
    tuple(numpy.ndarray, numpy.ndarray, int)
        A tuple of the spline knots, spline coefficients, and spline degree, which can be used to
        reconstruct the fit spline. Useful if needing to recreate the spline with different
        x-values.

    References
    ----------
    Eilers, P., et al. Splines, knots, and penalties. Wiley Interdisciplinary
    Reviews: Computational Statistics, 2010, 2(6), 637-653.

    """
    y, x = _yx_arrays(data, x_data, check_finite=check_finite, ensure_1d=True)
    pspline = PSpline(x, num_knots, spline_degree, check_finite, lam, diff_order)

    weight_array = _check_optional_array(
        len(y), weights, dtype=float, order='C', check_finite=check_finite
    )
    y_smooth = pspline.solve_pspline(y, weight_array)

    return y_smooth, pspline.tck
