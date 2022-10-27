# -*- coding: utf-8 -*-
"""Helper functions for pybaselines.

Created on March 5, 2021
@author: Donald Erb

"""

from math import ceil

import numpy as np
from scipy.ndimage import grey_opening
from scipy.signal import convolve

from ._banded_utils import PenalizedSystem, difference_matrix as _difference_matrix
from ._compat import jit
from ._validation import _check_array, _check_scalar, _check_optional_array


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
    return height * np.exp(-0.5 * ((x - center)**2) / max(sigma, _MIN_FLOAT)**2)


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
    output_coefs : numpy.ndarray
        The array of coefficients scaled for the original domain.

    """
    zeros_mask = np.equal(coef, 0)
    if zeros_mask.any():
        # coefficients with one or several zeros sometimes get compressed
        # to leave out some of the coefficients, so replace zero with another value
        # and then fill in later
        coef = coef.copy()
        coef[zeros_mask] = _MIN_FLOAT  # could probably fill it with any non-zero value

    fit_polynomial = np.polynomial.Polynomial(coef, domain=original_domain)
    output_coefs = fit_polynomial.convert().coef
    output_coefs[zeros_mask] = 0

    return output_coefs


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
    diff_matrix : scipy.sparse.base.spmatrix
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

    # TODO would it be better to allow padding the data?
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

    return max(half_window, 1)  # ensure half window is at least 1


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


def whittaker_smooth(data, lam=1e6, diff_order=2, weights=None, check_finite=True,
                     penalized_system=None):
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
    penalized_system : pybaselines._banded_utils.PenalizedSystem, optional
        If None (default), will create a new PenalizedSystem object for solving the equation.
        If not None, will use the object's `reset_diagonals` method and then solve.

    Returns
    -------
    smooth_y : numpy.ndarray, shape (N,)
        The smoothed data.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    y = _check_array(data, check_finite=check_finite, ensure_1d=True)
    len_y = len(y)
    if penalized_system is not None:
        penalized_system.reset_diagonals(lam=lam, diff_order=diff_order)
    else:
        penalized_system = PenalizedSystem(len_y, lam=lam, diff_order=diff_order)
    weight_array = _check_optional_array(len_y, weights, check_finite=check_finite)

    penalized_system.penalty[penalized_system.main_diagonal_index] = (
        penalized_system.penalty[penalized_system.main_diagonal_index] + weight_array
    )
    smooth_y = penalized_system.solve(
        penalized_system.penalty, weight_array * y, overwrite_ab=True, overwrite_b=True
    )

    return smooth_y
