# -*- coding: utf-8 -*-
"""Window-based techniques for fitting baselines to experimental data.

Created on March 7, 2021

@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage.filters import uniform_filter1d

from .utils import gaussian_kernel, mollify


def noise_median(data, half_window=1, smooth_half_window=1, sigma=5.0):
    """
    [summary]

    Assumes noise can be considered as the median value within a window.

    Parameters
    ----------
    data : [type]
        [description]
    half_window : int, optional
        [description], by default 1
    smooth_half_window : int, optional
        [description], by default 1
    sigma : float, optional
        [description], by default 5

    Returns
    -------
    np.ndarray
        The smoothed baseline.

    References
    ----------
    Friedrichs, M., A model-free algorithm for the removal of baseline
    artifacts. J. Biomolecular NMR. 5 (1995) 147-153.

    """
    median = median_filter(data, [2 * half_window + 1])
    return mollify(median, gaussian_kernel(2 * smooth_half_window + 1, sigma))


def snip(data, max_half_window, decreasing=False, smooth=False,
         smooth_half_window=1, transform=False):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Parameters
    ----------
    data : [type]
        [description]
    max_half_window : int
        The maximum number of iterations. Should be set so that
        max_half_window=(w-1)/2, where w is the index-based width of a feature.
    decreasing : bool, optional
        If False (default), will iterate through window sizes from 1 to
        max_half_window. If True, will reverse the order and iterate from
        max_half_window to 1, which gives a smoother baseline according to [3]_
        and [4]_.
    smooth : bool, optional
        If True, will perform a moving average on the data for each window, which
        gives better results for noisy data [3]_. Default is False.
    smooth_half_window : int, optional
        The half window to use for the moving average if smooth=True. Default is 1,
        which gives a 3-point moving average.
    transform : bool, optional
        [description], by default False

    Returns
    -------
    z : np.ndarray
        The baseline.

    Warns
    -----
    UserWarning
        Raised if max_half_window is greater than (data.shape[0] - 1) // 2.

    Notes
    -----
    Algorithm initially developed by [1]_ and this specific version of the
    algorithm is adapted from [2]_ and [4]_.

    References
    ----------
    .. [1] Ryan, C.G., et al. SNIP, A Statistics-Sensitive Background Treatment
           For The Quantitative Analysis Of Pixe Spectra In Geoscience Applications.
           Nuclear Instruments and Methods in Physics Research 934 (1988) 396-402.
    .. [2] Morhac, M., et al. Background elimination methods for multidimensional
           coincidence γ-ray spectra. Nuclear Instruments and Methods in Physics
           Research A 401 (1997) 113-132.
    .. [3] Morhac, M., et al. Peak Clipping Algorithms for Background Estimation in
           Spectroscopic Data. Applied Spectroscopy. 62(1) (2008) 91-106.
    .. [4] Morhac, M. An algorithm for determination of peak regions and baseline
           elimination in spectroscopic data. Nuclear Instruments and Methods in
           Physics Research A. 600 (2009) 478-487.

    """
    if max_half_window > (data.shape[0] - 1) // 2:
        warnings.warn(
            'max_half_window values greater than (data.shape[0] - 1) / 2 have no effect.'
        )
        max_half_window = (data.shape[0] - 1) // 2

    if transform:
        z = np.log(np.log(np.sqrt(data + 1) + 1) + 1)
    else:
        z = data.copy()

    if decreasing:
        min_val = max_half_window
        max_val = 0
        step = -1
    else:
        min_val = 1
        max_val = max_half_window + 1
        step = 1

    for i in range(min_val, max_val, step):
        medians = 0.5 * (z[2 * i:] + z[:-2 * i])
        if smooth:
            #TODO could speed this up by not doing the entirety of z
            means = uniform_filter1d(z, 2 * smooth_half_window + 1)[i:-i]
        else:
            means = z[i:-i]
        z[i:-i] = np.where(z[i:-i] > medians, medians, means)

    if transform:
        z = -1 + (np.exp(np.exp(z) - 1) - 1)**2

    return z
