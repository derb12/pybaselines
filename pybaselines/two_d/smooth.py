# -*- coding: utf-8 -*-
"""Smoothing-based techniques for fitting baselines to experimental data.

Created on April 8, 2023
@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from ._algorithm_setup import _Algorithm2D


class _Smooth(_Algorithm2D):
    """A base class for all smoothing algorithms."""

    @_Algorithm2D._register
    def noise_median(self, data, half_window=None, smooth_half_window=None, sigma=None,
                     **pad_kwargs):
        """
        The noise-median method for baseline identification.

        Assumes the baseline can be considered as the median value within a moving
        window, and the resulting baseline is then smoothed with a Gaussian kernel.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        half_window : int or Sequence[int, int], optional
            The index-based size to use for the median window on the rows and columns,
            respectively. The total window size in each dimension will range from
            [-half_window, ..., half_window] with size 2 * half_window + 1. Default is
            None, which will use twice the output from :func:`.optimize_window`,
            which is an okay starting value.
        smooth_half_window : int, optional
            The half window to use for smoothing. Default is None, which will use
            the average of the values in `half_window`.
        sigma : float, optional
            The standard deviation of the smoothing Gaussian kernel. Default is None,
            which will use (2 * `smooth_half_window` + 1) / 6.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges2d` for padding
            the edges of the data to prevent edge effects from convolution.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated and smoothed baseline.
        dict
            An empty dictionary, just to match the output of all other algorithms.

        References
        ----------
        Friedrichs, M., A model-free algorithm for the removal of baseline
        artifacts. J. Biomolecular NMR, 1995, 5, 147-153.

        """
        y, half_window = self._setup_smooth(data, half_window, False, 2, **pad_kwargs)
        window_size = 2 * half_window + 1
        median = median_filter(y, window_size, mode='nearest')
        if smooth_half_window is None:
            smooth_window = np.mean(window_size)  # truncate can only be a single value
        else:
            smooth_window = 2 * smooth_half_window + 1
        if sigma is None:
            # the gaussian kernel will includes +- 3 sigma
            sigma = smooth_window / 6

        baseline = gaussian_filter(median, sigma, truncate=smooth_window)  # TODO check truncate value
        return baseline[half_window[0]:-half_window[0], half_window[1]:-half_window[1]], {}
