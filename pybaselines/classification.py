# -*- coding: utf-8 -*-
"""Techniques that rely on classifying peak and/or baseline segments for fitting baselines.

Created on July 3, 2021
@author: Donald Erb

"""

from math import ceil
import warnings

import numpy as np
from scipy.ndimage import binary_erosion, grey_dilation, grey_erosion, uniform_filter1d

from ._algorithm_setup import _get_vander, _yx_arrays, _setup_polynomial
from .utils import _interp_inplace, pad_edges, relative_difference, ParameterWarning


def _remove_single_points(mask):
    """
    Removes lone True or False values from the mask.

    Removes the lone True values first since True values designate the baseline.
    That way, the approach is more conservative with assigning baseline points.

    """
    # TODO need to verify this works completely; check edges as well
    baseline_mask = np.asarray(mask, bool)
    # convert lone True values to False
    # same (check) as baseline_mask ^ binary_erosion(~baseline_mask, [1, 0, 1], border_value=1)
    temp = binary_erosion(baseline_mask, [1, 1, 0]) + binary_erosion(baseline_mask, [0, 1, 1])
    # convert lone False values to True
    return temp + binary_erosion(temp, [1, 0, 1])


def golotvin(data, half_window, x_data=None, n_sigma=2.0, sections=32, smooth_half_window=20):
    """
    [summary]

    Parameters
    ----------
    data : [type]
        [description]
    half_window : [type]
        [description]
    n_sigma : float, optional
        [description]. Default is 2.
    smooth_half_window : int, optional
        [description]. Default is 20.

    Returns
    -------
    [type]
        [description]

    References
    ----------
    Golotvin, S., et al. Improved Baseline Recognition and Modeling of
    FT NMR Spectra. Journal of Magnetic Resonance. 2000, 146, 122-125.

    """
    y, x = _yx_arrays(data, x_data, 0, len(data))
    y = np.asarray(data)
    n = y.shape[0]
    min_sigma = np.inf
    for i in range(sections):
        # use ddof=1 since sampling subsets of the data
        min_sigma = min(
            min_sigma,
            np.std(y[i * n // sections:(i + 1) * n // sections], ddof=1)
        )

    mask = (
        grey_dilation(y, 2 * half_window + 1) - grey_erosion(y, 2 * half_window + 1)
    ) < n_sigma * min_sigma
    mask = _remove_single_points(mask)

    if not mask.sum():
        warnings.warn(
            'no baseline points found; try increasing "n_sigma" or changing "half_window"',
            ParameterWarning
        )
        return np.zeros_like(y)  # TODO return zeros or throw an error?

    # plt.plot(x, y, '-', x[mask], y[mask], 'o-')
    # plt.show()

    # TODO maybe allow taking the mean of neighboring points when interpolating, like the other
    # methods
    rough_baseline = np.interp(x, x[mask], y[mask])
    # TODO pad the baseline before smoothing?
    baseline = uniform_filter1d(rough_baseline, 2 * smooth_half_window + 1)

    return baseline


def dietrich(data, x_data=None, smooth_half_window=None, n_sigma=3.0, mean_half_window=10,
             poly_order=5, max_iter=50, tol=1e-3, **pad_kwargs):
    """
    [summary]

    Parameters
    ----------
    data : [type]
        [description]
    smooth_half_window : int, optional
        [description]

    References
    ----------
    Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
    Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

    """
    y, x, *_ = _setup_polynomial(data, x_data)
    num_y = y.shape[0]

    if smooth_half_window is None:
        smooth_half_window = ceil(num_y / 256)
    smooth_y = uniform_filter1d(
        pad_edges(y, smooth_half_window, **pad_kwargs),
        2 * smooth_half_window + 1
    )[smooth_half_window:-smooth_half_window]

    power = np.diff(np.hstack((smooth_y[0], smooth_y)))**2
    mask = power < np.mean(power) + n_sigma * np.std(power, ddof=1)
    old_mask = np.ones_like(mask)
    while not np.array_equal(mask, old_mask):
        old_mask = mask
        masked_power = power[mask]
        if masked_power.size < 2:  # need at least 2 points for interpolation later
            warnings.warn(
                'not enough baseline points found; "n_sigma" is most likely too low',
                ParameterWarning
            )
            return np.zeros_like(y)  # TODO return zeros or throw an error?
        mask = power < np.mean(masked_power) + n_sigma * np.std(masked_power, ddof=1)

    mask = _remove_single_points(mask)
    if mask.sum() < 2:
        warnings.warn(
            'not enough baseline points found; try increasing "n_sigma"', ParameterWarning
        )
        return np.zeros_like(y)  # TODO return zeros or throw an error?

    rough_baseline = y.copy()
    indices = np.flatnonzero(mask)
    for i, point in enumerate(indices[:-1]):
        # TODO this is probably not right; want to only connect peak segments, not every other point
        point_2 = indices[i + 1]
        rough_baseline[point] = np.mean(
            y[max(0, point - mean_half_window):min(point + mean_half_window + 1, num_y)]
        )
        rough_baseline[point_2] = np.mean(
            y[max(0, point_2 - mean_half_window):min(point_2 + mean_half_window + 1, num_y)]
        )
        _interp_inplace(x[point:point_2 + 1], rough_baseline[point:point_2 + 1])

    baseline = rough_baseline
    old_coefs = np.zeros(poly_order + 1)
    vander, pseudo_inverse = _get_vander(x, poly_order)
    for i in range(max_iter):
        coefs = np.dot(pseudo_inverse, rough_baseline)
        baseline = np.dot(vander, coefs)
        if relative_difference(old_coefs, coefs) < tol:
            break
        old_coefs = coefs
        rough_baseline[mask] = baseline[mask]

    # plt.plot(x, y, '-', x[mask], smooth_y[mask], 'o-')
    # plt.show()

    return baseline


def _rolling_std(data, half_window, ddof=0):
    """
    Computes the rolling standard deviation of an array.

    Parameters
    ----------
    data : numpy.ndarray
        The array for the calculation.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int, optional
        The degrees of freedom for the calculation. Default is 0.

    Returns
    -------
    rolled_std : numpy.ndarray
        The array of the rolling standard deviation for each window.

    """
    # TODO this is a nieve approach; switch to Welford's method when possible;
    # cannot use numba with this approach since numba's std does not allow specifying
    # ddof
    num_y = data.shape[0]
    rolling_std = np.array([
        np.std(data[max(0, i - half_window):min(i + half_window + 1, num_y)], ddof=ddof)
        for i in range(num_y)
    ])

    return rolling_std


def _signal_start(mask):
    # TODO find a way to join this with golotvin's and dietrich's methods
    sig_start = [0] if mask[0] else []
    for i in range(1, mask.shape[0]):
        if mask[i] > mask[i - 1]:
            sig_start.append(i)

    return np.array(sig_start)


def noise_distribution(data, half_window, mean_half_window=5, smooth_half_window=None):
    """
    [summary]

    Parameters
    ----------
    data : [type]
        [description]
    half_window : [type]
        [description]

    References
    ----------
    Wang, K.C., et al. Distribution-Based Classification Method for Baseline
    Correction of Metabolomic 1D Proton Nuclear Magnetic Resonance Spectra.
    Analytical Chemistry. 2013, 85, 1231-1239.

    """
    y = np.asarray(data)
    num_y = y.shape[0]
    if smooth_half_window is None:
        smooth_half_window = half_window

    std = _rolling_std(y, half_window, 1)  # use dof=1 since sampling a subset of the data
    med = np.median(std)
    med2 = np.median(std[std < 2 * med])
    while med2 / med < 0.999:  # TODO make the 0.999 an input?
        med = med2
        med2 = np.median(std[std < 2 * med])
    noise_std = med2

    # TODO currently, this is peaks==1, baseline==0; switch to match golotvin & dietrich
    mask = np.zeros(num_y, int)
    half_win = 3  # TODO make the half_win an input
    for i in range(num_y):
        if std[i] > 1.1 * noise_std:  # TODO make the 1.1 an input; call threshold or cutoff or num_std
            mask[max(0, i - half_win):min(i + half_win + 1, num_y)] = 1

    sig_start = _signal_start(mask)
    sig_end = num_y - 1 - _signal_start(mask[::-1])[::-1]
    rough_z = y.copy()
    filler = np.arange(num_y, dtype=float)
    for i in range(sig_start.shape[0]):
        start = sig_start[i]
        end = sig_end[i]
        rough_z[start] = np.mean(
            y[max(0, start - mean_half_window):min(start + mean_half_window + 1, num_y)]
        )
        rough_z[end] = np.mean(
            y[max(0, end - mean_half_window):min(end + mean_half_window + 1, num_y)]
        )
        _interp_inplace(filler[start:end + 1], rough_z[start:end + 1])
    # TODO pad rough_z before smoothing? or pad y before doing rolling std?
    return uniform_filter1d(rough_z, 2 * smooth_half_window + 1)
