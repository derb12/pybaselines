# -*- coding: utf-8 -*-
"""Window-based techniques for fitting baselines to experimental data.

Window
    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)

Created on March 7, 2021
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

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
    artifacts. J. Biomolecular NMR. 5 (1995) 147-153.

    """
    median = median_filter(
        _setup_window(data, half_window, **pad_kwargs),
        [2 * half_window + 1], mode='nearest'
    )
    z = padded_convolve(median, gaussian_kernel(2 * smooth_half_window + 1, sigma))
    return z[half_window:-half_window], {}


def snip(data, max_half_window, decreasing=False, smooth=False,
         smooth_half_window=1, **pad_kwargs):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    max_half_window : int
        The maximum number of iterations. Should be set so that
        max_half_window=(w-1)/2, where w is the index-based width of a
        feature or peak.
    decreasing : bool, optional
        If False (default), will iterate through window sizes from 1 to
        max_half_window. If True, will reverse the order and iterate from
        max_half_window to 1, which gives a smoother baseline according to [3]_
        and [4]_.
    smooth : bool, optional
        If True, will perform a moving average smooth on the data for each window,
        which gives better results for noisy data [3]_. Default is False.
    smooth_half_window : int, optional
        The half window to use for the moving average if smooth=True. Default is 1,
        which gives a 3-point moving average.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Warns
    -----
    UserWarning
        Raised if max_half_window is greater than (data.shape[0] - 1) // 2.

    Notes
    -----
    Algorithm initially developed by [1]_ and this specific version of the
    algorithm is adapted from [2]_ and [4]_.

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
           Nuclear Instruments and Methods in Physics Research 934 (1988) 396-402.
    .. [2] Morháč, M., et al. Background elimination methods for multidimensional
           coincidence γ-ray spectra. Nuclear Instruments and Methods in Physics
           Research A 401 (1997) 113-132.
    .. [3] Morháč, M., et al. Peak Clipping Algorithms for Background Estimation in
           Spectroscopic Data. Applied Spectroscopy. 62(1) (2008) 91-106.
    .. [4] Morháč, M. An algorithm for determination of peak regions and baseline
           elimination in spectroscopic data. Nuclear Instruments and Methods in
           Physics Research A. 600 (2009) 478-487.

    #TODO potentially add filter orders 4, 6, and 8 from [3]_
    #TODO potentially add adaptive window sizes from [4]_
    """
    y = _setup_window(data, max_half_window, **pad_kwargs)
    if max_half_window > (y.shape[0] - 1) // 2:
        warnings.warn(
            'max_half_window values greater than (data.shape[0] - 1) / 2 have no effect.'
        )
        max_half_window = (y.shape[0] - 1) // 2

    if decreasing:
        range_args = (max_half_window, 0, -1)
    else:
        range_args = (1, max_half_window + 1, 1)

    z = y.copy()
    for i in range(*range_args):
        medians = 0.5 * (z[2 * i:] + z[:-2 * i])
        if smooth:
            #TODO could speed this up by not doing the entirety of z
            means = uniform_filter1d(z, 2 * smooth_half_window + 1)[i:-i]
        else:
            means = z[i:-i]
        z[i:-i] = np.where(z[i:-i] > medians, medians, means)

    return z[max_half_window:-max_half_window], {}
