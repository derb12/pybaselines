# -*- coding: utf-8 -*-
"""Window-based techniques for fitting baselines to experimental data.

Window
    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    3) swima (Small-Window Moving Average)

Created on March 7, 2021
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import savgol_filter

from ._algorithm_setup import _setup_window
from .utils import gaussian_kernel, padded_convolve


def noise_median(data, half_window, smooth_half_window=1, sigma=5.0, **pad_kwargs):
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
        The half window to use for smoothing. Default is 1.
    sigma : float, optional
        The standard deviation of the smoothing Gaussian kernel. Default is 5.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated and smoothed baseline.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    References
    ----------
    Friedrichs, M., A model-free algorithm for the removal of baseline
    artifacts. J. Biomolecular NMR, 1995, 5, 147-153.

    """
    median = median_filter(
        _setup_window(data, half_window, **pad_kwargs),
        [2 * half_window + 1], mode='nearest'
    )
    z = padded_convolve(median, gaussian_kernel(2 * smooth_half_window + 1, sigma))
    return z[half_window:-half_window], {}


def snip(data, max_half_window, decreasing=False, smooth_half_window=0,
         filter_order=2,**pad_kwargs):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    max_half_window : int or Sequence(int, int)
        The maximum number of iterations. Should be set such that
        `max_half_window`=(w-1)/2, where w is the index-based width of a
        feature or peak. `max_half_window` can also be a sequence of two
        integers for asymmetric peaks, with the first item corresponding to
        the `max_half_window` of the peak's left edge, and the second item
        for the peak's right edge [3]_.
    decreasing : bool, optional
        If False (default), will iterate through window sizes from 1 to
        max_half_window. If True, will reverse the order and iterate from
        max_half_window to 1, which gives a smoother baseline according to [3]_
        and [4]_.
    smooth_half_window : int, optional
        The half window to use for smoothing the data. If `smooth_half_window`
        is greater than 0, will perform a moving average smooth on the data for
        each window, which gives better results for noisy data [3]_. Default is
        0, which will not perform any smoothing.
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
    z : numpy.ndarray, shape (N,)
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
    using SNIP [2]_:

        transformed_data =  np.log(np.log(np.sqrt(data + 1) + 1) + 1)

    and then baseline can then be reverted back to the original scale using inverse:

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

    #TODO potentially add adaptive window sizes from [4]_
    """
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
    smooth = smooth_half_window > 0
    z = y.copy()
    for i in range(*range_args):
        i_left = min(i, half_windows[0])
        i_right = min(i, half_windows[1])

        filters = (z[i - i_left:num_y - i - i_left] + z[i + i_right:num_y - i + i_right]) / 2
        if filter_order > 2:
            filters_new = (
                - (z[i - i_left:num_y - i - i_left] + z[i + i_right:num_y - i + i_right])
                + 4 * (z[i - i_left // 2:-i - i_left // 2] + z[i + i_right // 2:-i + i_right // 2])
            ) / 6
            filters = np.maximum(filters, filters_new)
        if filter_order > 4:
            filters_new = (
                z[i - i_left:num_y - i - i_left] + z[i + i_right:num_y - i + i_right]
                - 6 * (z[i - 2 * i_left // 3:-i - 2 * i_left // 3] + z[i + 2 * i_right // 3:-i + 2 * i_right // 3])
                + 15 * (z[i - i_left // 3:-i - i_left // 3] + z[i + i_right // 3:-i + i_right // 3])
            ) / 20
            filters = np.maximum(filters, filters_new)
        if filter_order > 6:
            filters_new = (
                - (z[i - i_left:num_y - i - i_left] + z[i + i_right:num_y - i + i_right])
                + 8 * (z[i - 3 * i_left // 4:-i - 3 * i_left // 4] + z[i + 3 * i_right // 4:-i + 3 * i_right // 4])
                - 28 * (z[i - i_left // 2:-i - i_left // 2] + z[i + i_right // 2:-i + i_right // 2])
                + 56 * (z[i - i_left // 4:-i - i_left // 4] + z[i + i_right // 4:-i + i_right // 4])
            ) / 70
            filters = np.maximum(filters, filters_new)

        if smooth:
            previous_baseline = uniform_filter1d(z, 2 * smooth_half_window + 1)[i:-i]
        else:
            previous_baseline = z[i:-i]
        z[i:-i] = np.where(z[i:-i] > filters, filters, previous_baseline)

    return z[max_of_half_windows:-max_of_half_windows], {}


def swima(data, max_half_window=None, min_half_window=3, smooth_half_window=None,
          **pad_kwargs):
    """
    Small-window moving average (SWiMA) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    max_half_window : int, optional
        The maximum number of iterations. Default is None, which will use
        (N - 1) / 2. Typically does not need to be specified.
    min_half_window : int, optional
        The minimum half window value that must be reached before the exit criteria
        is considered. Default is 3.
    smooth_half_window : int, optional
        The half window to use for smoothing the input data with a moving average.
        Default is None, which will use N / 50. Use a value of 0 or less to not
        smooth the data. See Notes below for more details.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window at which the exit criteria was reached.

    Notes
    -----
    This algorithm requires the input data to be fairly smooth (noise-free), so it
    is recommended to either smooth the data beforehand, or specify a
    `smooth_half_window` value. Non-smooth data can cause the exit criteria to be
    reached prematurely (can be avoided by setting a larger `min_half_window`), while
    over-smoothed data can cause the exit criteria to be reached later than optimal.

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
        z = uniform_filter1d(y, 2 * smooth_half_window + 1)
    else:
        z = y

    area_current = 0
    area_old = 0
    for half_window in range(1, max_half_window + 1):
        z_new = np.minimum(z, savgol_filter(z, 2 * half_window + 1, 0, mode='nearest'))

        area_new = np.trapz(z[data_slice] - z_new[data_slice])
        if half_window > min_half_window and area_new > area_current and area_current < area_old:
            break
        area_old = area_current
        area_current = area_new
        z = z_new
        #TODO need to implement the second exit condition? will increase time since
        # have to differentiate, fit with polynomial, and integrate

    return z[data_slice], {'half_window': half_window - 1}
