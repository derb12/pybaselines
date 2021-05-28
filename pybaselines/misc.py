# -*- coding: utf-8 -*-
"""Miscellaneous functions for creating baselines.

Created on April 2, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.interpolate import interp1d


def interp_pts(x_data, baseline_points=(), interp_method='linear'):
    """
    Creates a baseline by interpolating through input points.

    Parameters
    ----------
    x_data : array-like, shape (N,)
        The x-values of the measured data.
    baseline_points : array-like, shape (n, 2)
        An array of ((x_1, y_1), (x_2, y_2), ..., (x_n, y_n)) values for
        each point representing the baseline.
    interp_method : string, optional
        The method to use for interpolation. See :func:`scipy.interpolation.interp1d`
        for all options. Default is 'linear', which connects each point with
        a line segment.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline array constructed from interpolating between
        each input baseline point.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Notes
    -----
    This method is only suggested for use within user-interfaces.

    """
    x = np.asarray(x_data)
    points = np.asarray(baseline_points).T

    interpolator = interp1d(
        *points, kind=interp_method, bounds_error=False, fill_value=0
    )
    baseline = interpolator(np.linspace(np.nanmin(x), np.nanmax(x), x.shape[0]))

    return baseline, {}
