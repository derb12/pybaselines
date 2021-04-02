# -*- coding: utf-8 -*-
"""Functions for creating manually-created baselines.

Manual
    1) linear_interp (Linear interpolation between points)


Created on April 2, 2021
@author: Donald Erb

"""

import warnings

import numpy as np


def linear_interp(x_data, baseline_points=()):
    """
    Creates a linear baseline constructed from points.

    Parameters
    ----------
    x_data : array-like, shape (N,)
        The x-values of the measured data.
    baseline_points : Iterable(Container(float, float))
        An iterable of ((x_1, y_1), (x_2, y_2), ..., (x_n, y_n)) values for
        each point representing the baseline. Must be at least two points
        to have a non-zero baseline.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The baseline array constructed from connecting line segments between
        each background point.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Warns
    -----
    UserWarning
        Raised if there are less than two points in baseline_points.

    Notes
    -----
    Assumes the background is represented by lines connecting each of the
    specified background points.

    """
    #TODO allow polynomial and spline interpolation
    x = np.asarray(x_data)
    z = np.zeros(x.shape[0])
    if len(baseline_points) < 2:
        warnings.warn('there must be at least 2 background points to create a baseline')
    else:
        points = sorted(baseline_points, key=lambda p: p[0])
        for i in range(len(points) - 1):
            x_points, y_points = zip(*points[i:i + 2])
            segment = (x >= x_points[0]) & (x <= x_points[1])
            z[segment] = np.linspace(*y_points, x[segment].shape[0])

    return z, {}
