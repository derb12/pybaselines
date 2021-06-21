# -*- coding: utf-8 -*-
"""Window-based techniques for fitting baselines to experimental data.

Created on March 7, 2021
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from ._algorithm_setup import _get_vander, _setup_window
from .utils import gaussian, gaussian_kernel, padded_convolve


def noise_median(data, half_window, smooth_half_window=None, sigma=None, **pad_kwargs):
    """
    The noise-median method for baseline identification.

    Assumes the baseline can be considered as the median value within a moving
    window, and the resulting baseline is then smoothed with a Gaussian kernel.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int
        The index-based size to use for the median window. The total window
        size will range from [-half_window, ..., half_window] with size
        2 * half_window + 1.
    smooth_half_window : int, optional
        The half window to use for smoothing. Default is None, which will use
        the same value as `half_window`.
    sigma : float, optional
        The standard deviation of the smoothing Gaussian kernel. Default is None,
        which will use (2 * `smooth_half_window` + 1) / 6.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated and smoothed baseline.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    References
    ----------
    Friedrichs, M., A model-free algorithm for the removal of baseline
    artifacts. J. Biomolecular NMR, 1995, 5, 147-153.

    """
    window_size = 2 * half_window + 1
    median = median_filter(
        _setup_window(data, half_window, **pad_kwargs),
        [window_size], mode='nearest'
    )
    if smooth_half_window is None:
        smooth_window = window_size
    else:
        smooth_window = 2 * smooth_half_window + 1
    if sigma is None:
        # the gaussian kernel will includes +- 3 sigma
        sigma = smooth_window / 6
    baseline = padded_convolve(median, gaussian_kernel(smooth_window, sigma))
    return baseline[half_window:-half_window], {}


def snip(data, max_half_window, decreasing=False, smooth_half_window=None,
         filter_order=2, **pad_kwargs):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    max_half_window : int or Sequence(int, int)
        The maximum number of iterations. Should be set such that
        `max_half_window` is approxiamtely ``(w-1)/2``, where ``w`` is the index-based
        width of a feature or peak. `max_half_window` can also be a sequence of
        two integers for asymmetric peaks, with the first item corresponding to
        the `max_half_window` of the peak's left edge, and the second item
        for the peak's right edge [3]_.
    decreasing : bool, optional
        If False (default), will iterate through window sizes from 1 to
        `max_half_window`. If True, will reverse the order and iterate from
        `max_half_window` to 1, which gives a smoother baseline according to [3]_
        and [4]_.
    smooth_half_window : int, optional
        The half window to use for smoothing the data. If `smooth_half_window`
        is greater than 0, will perform a moving average smooth on the data for
        each window, which gives better results for noisy data [3]_. Default is
        None, which will not perform any smoothing.
    filter_order : {2, 4, 6, 8}, optional
        If the measured data has a more complicated baseline consisting of other
        elements such as Compton edges, then a higher `filter_order` should be
        selected [3]_. Default is 2, which works well for approximating a linear
        baseline.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Raises
    ------
    ValueError
        Raised if `filter_order` is not 2, 4, 6, or 8.

    Warns
    -----
    UserWarning
        Raised if max_half_window is greater than (len(data) - 1) // 2.

    Notes
    -----
    Algorithm initially developed by [1]_, and this specific version of the
    algorithm is adapted from [2]_, [3]_, and [4]_.

    If data covers several orders of magnitude, better results can be obtained
    by first transforming the data using log-log-square transform before
    using SNIP [2]_::

        transformed_data =  np.log(np.log(np.sqrt(data + 1) + 1) + 1)

    and then baseline can then be reverted back to the original scale using inverse::

        baseline = -1 + (np.exp(np.exp(snip(transformed_data)) - 1) - 1)**2

    References
    ----------
    .. [1] Ryan, C.G., et al. SNIP, A Statistics-Sensitive Background Treatment
           For The Quantitative Analysis Of Pixe Spectra In Geoscience Applications.
           Nuclear Instruments and Methods in Physics Research B, 1988, 934, 396-402.
    .. [2] Morháč, M., et al. Background elimination methods for multidimensional
           coincidence γ-ray spectra. Nuclear Instruments and Methods in Physics
           Research A, 1997, 401, 113-132.
    .. [3] Morháč, M., et al. Peak Clipping Algorithms for Background Estimation in
           Spectroscopic Data. Applied Spectroscopy, 2008, 62(1), 91-106.
    .. [4] Morháč, M. An algorithm for determination of peak regions and baseline
           elimination in spectroscopic data. Nuclear Instruments and Methods in
           Physics Research A, 2009, 60, 478-487.

    """
    #TODO potentially add adaptive window sizes from [4]_, or at least allow inputting
    # an array of max_half_windows; would need to have a separate function for array
    # windows since it would no longer be able to be vectorized
    if filter_order not in {2, 4, 6, 8}:
        raise ValueError('filter_order must be 2, 4, 6, or 8')

    if isinstance(max_half_window, int):
        half_windows = [max_half_window, max_half_window]
    elif len(max_half_window) == 1:
        half_windows = [max_half_window[0], max_half_window[0]]
    else:
        half_windows = [max_half_window[0], max_half_window[1]]

    num_y = len(data)
    for i, half_window in enumerate(half_windows):
        if half_window > (num_y - 1) // 2:
            warnings.warn(
                'max_half_window values greater than (len(data) - 1) / 2 have no effect.'
            )
            half_windows[i] = (num_y - 1) // 2

    max_of_half_windows = max(half_windows)
    if decreasing:
        range_args = (max_of_half_windows, 0, -1)
    else:
        range_args = (1, max_of_half_windows + 1, 1)

    y = _setup_window(data, max_of_half_windows, **pad_kwargs)
    num_y = y.shape[0]  # new num_y since y is now padded
    smooth = smooth_half_window is not None and smooth_half_window > 0
    baseline = y.copy()
    for i in range(*range_args):
        i_left = min(i, half_windows[0])
        i_right = min(i, half_windows[1])

        filters = (
            baseline[i - i_left:num_y - i - i_left] + baseline[i + i_right:num_y - i + i_right]
        ) / 2
        if filter_order > 2:
            filters_new = (
                - (
                    baseline[i - i_left:num_y - i - i_left]
                    + baseline[i + i_right:num_y - i + i_right]
                )
                + 4 * (
                    baseline[i - i_left // 2:-i - i_left // 2]
                    + baseline[i + i_right // 2:-i + i_right // 2]
                )
            ) / 6
            filters = np.maximum(filters, filters_new)
        if filter_order > 4:
            filters_new = (
                baseline[i - i_left:num_y - i - i_left] + baseline[i + i_right:num_y - i + i_right]
                - 6 * (
                    baseline[i - 2 * i_left // 3:-i - 2 * i_left // 3]
                    + baseline[i + 2 * i_right // 3:-i + 2 * i_right // 3]
                )
                + 15 * (
                    baseline[i - i_left // 3:-i - i_left // 3]
                    + baseline[i + i_right // 3:-i + i_right // 3]
                )
            ) / 20
            filters = np.maximum(filters, filters_new)
        if filter_order > 6:
            filters_new = (
                - (
                    baseline[i - i_left:num_y - i - i_left]
                    + baseline[i + i_right:num_y - i + i_right]
                )
                + 8 * (
                    baseline[i - 3 * i_left // 4:-i - 3 * i_left // 4]
                    + baseline[i + 3 * i_right // 4:-i + 3 * i_right // 4]
                )
                - 28 * (
                    baseline[i - i_left // 2:-i - i_left // 2]
                    + baseline[i + i_right // 2:-i + i_right // 2]
                )
                + 56 * (
                    baseline[i - i_left // 4:-i - i_left // 4]
                    + baseline[i + i_right // 4:-i + i_right // 4]
                )
            ) / 70
            filters = np.maximum(filters, filters_new)

        if smooth:
            previous_baseline = uniform_filter1d(baseline, 2 * smooth_half_window + 1)[i:-i]
        else:
            previous_baseline = baseline[i:-i]
        baseline[i:-i] = np.where(baseline[i:-i] > filters, filters, previous_baseline)

    return baseline[max_of_half_windows:-max_of_half_windows], {}


def _swima_loop(y, vander, pseudo_inverse, data_slice, max_half_window, min_half_window=3):
    """
    Computes an iterative moving average to smooth peaks and obtain the baseline.

    The internal loop of the small-window moving average (SWiMA) algorithm.

    Parameters
    ----------
    y : numpy.ndarray, shape (N + 2 * max_half_window,)
        The array of the measured data with N data points padded at each edge with
        `max_half_window` extra data points.
    vander : numpy.ndarray, shape (N - 1, 4)
        The Vandermonde matrix for computing the 3rd order polynomial fit of the
        differential of the residual. Used for the alternate exit criteria.
    pseudo_inverse : numpy.ndarray, shape (4, N - 1)
        The pseudo-inverse of the Vandermonde matrix for computing the 3rd order
        polynomial fit of the differential of the residual. Used for the alternate
        exit criteria.
    data_slice : slice
        The slice used for separating the actual values of `y` from the extended y
        array.
    max_half_window : int
        The maximum allowable half window.
    min_half_window : int, optional
        The minimum half window that must be reached before exit criteria are
        considered. Default is 3.

    Returns
    -------
    baseline : numpy.ndarray, shape (N + 2 * max_half_window,)
        The baseline with the padded edges.
    converged : bool or None
        Whether the main exit criteria was achieved. True if it was, False
        if the alternate exit criteria was achieved, and None if `max_half_window`
        was reached before either exit criteria.
    half_window : int
        The half window at which the exit criteria was reached.

    Notes
    -----
    Uses a moving average rather than a 0-degree Savitzky–Golay filter since
    they are equivalent and the moving average is faster.

    The second exit criteria is based on Figure 2 in the reference, since the
    slightly different definition of criteria two stated in the text was always
    reached before the main exit criteria, which is not the desired outcome.

    References
    ----------
    Schulze, H., et al. A Small-Window Moving Average-Based Fully Automated
    Baseline Estimation Method for Raman Spectra. Applied Spectroscopy, 2012,
    66(7), 757-764.

    """
    actual_y = y[data_slice]
    baseline = y
    min_half_window_check = min_half_window - 2
    area_current = -1
    area_old = -1
    converged = None
    for half_window in range(1, max_half_window + 1):
        baseline_new = np.minimum(baseline, uniform_filter1d(baseline, 2 * half_window + 1))
        # only begin calculating the area when near the lowest allowed half window
        if half_window > min_half_window_check:
            area_new = np.trapz(baseline[data_slice] - baseline_new[data_slice])
            # exit criteria 1
            if area_new > area_current and area_current < area_old:
                converged = True
                # subtract 1 since minimum area was reached the previous iteration
                half_window -= 1
                break
            if half_window > min_half_window:
                diff_current = np.diff(actual_y - baseline_new[data_slice])
                poly_diff_current = np.trapz(
                    abs(np.dot(vander, np.dot(pseudo_inverse, diff_current)))
                )
                # exit criteria 2, means baseline is not well fit
                if poly_diff_current > 0.15 * np.trapz(abs(diff_current)):
                    converged = False
                    break
            area_old = area_current
            area_current = area_new
        baseline = baseline_new

    return baseline, converged, half_window


def swima(data, min_half_window=3, max_half_window=None, smooth_half_window=None, **pad_kwargs):
    """
    Small-window moving average (SWiMA) baseline.

    Computes an iterative moving average to smooth peaks and obtain the baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    min_half_window : int, optional
        The minimum half window value that must be reached before the exit criteria
        is considered. Can be increased to reduce the calculation time. Default is 3.
    max_half_window : int, optional
        The maximum number of iterations. Default is None, which will use
        (N - 1) / 2. Typically does not need to be specified.
    smooth_half_window : int, optional
        The half window to use for smoothing the input data with a moving average.
        Default is None, which will use N / 50. Use a value of 0 or less to not
        smooth the data. See Notes below for more details.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': list(int)
            A list of the half windows at which the exit criteria was reached.
            Has a length of 1 if the main exit criteria was intially reached,
            otherwise has a length of 2.
        * 'converged': list(bool or None)
            A list of the convergence status. Has a length of 1 if the main
            exit criteria was intially reached, otherwise has a length of 2.
            Each convergence status is True if the main exit criteria was
            reached, False if the second exit criteria was reached, and None
            if `max_half_window` is reached before either exit criteria.

    Notes
    -----
    This algorithm requires the input data to be fairly smooth (noise-free), so it
    is recommended to either smooth the data beforehand, or specify a
    `smooth_half_window` value. Non-smooth data can cause the exit criteria to be
    reached prematurely (can be avoided by setting a larger `min_half_window`), while
    over-smoothed data can cause the exit criteria to be reached later than optimal.

    The half-window at which convergence occurs is roughly close to the index-based
    full-width-at-half-maximum of a peak or feature, but can vary. Therfore, it is
    better to set a `min_half_window` that is smaller than expected to not miss the
    exit criteria.

    If the main exit criteria is not reached on the initial fit, a gaussian baseline
    (which is well handled by this algorithm) is added to the data, and it is re-fit.

    References
    ----------
    Schulze, H., et al. A Small-Window Moving Average-Based Fully Automated
    Baseline Estimation Method for Raman Spectra. Applied Spectroscopy, 2012,
    66(7), 757-764.

    """
    if max_half_window is None:
        max_half_window = (len(data) - 1) // 2
    y = _setup_window(data, max_half_window, **pad_kwargs)

    data_slice = slice(max_half_window, -max_half_window)
    if smooth_half_window is None:
        smooth_half_window = max(1, y[data_slice].shape[0] // 50)
    if smooth_half_window > 0:
        y = uniform_filter1d(y, 2 * smooth_half_window + 1)

    vander, pseudo_inverse = _get_vander(np.linspace(-1, 1, y[data_slice].shape[0] - 1), 3)
    baseline, converged, half_window = _swima_loop(
        y, vander, pseudo_inverse, data_slice, max_half_window, min_half_window
    )
    converges = [converged]
    half_windows = [half_window]
    if not converged:
        residual = y - baseline
        gaussian_bkg = gaussian(
            np.arange(y.shape[0]), np.max(residual), y.shape[0] / 2, y.shape[0] / 6
        )
        baseline_2, converged, half_window = _swima_loop(
            residual + gaussian_bkg, vander, pseudo_inverse, data_slice, max_half_window, 3
        )
        baseline += baseline_2 - gaussian_bkg
        converges.append(converged)
        half_windows.append(half_window)

    return baseline[data_slice], {'half_window': half_windows, 'converged': converges}
