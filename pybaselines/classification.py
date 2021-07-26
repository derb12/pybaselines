# -*- coding: utf-8 -*-
"""Techniques that rely on classifying peak and/or baseline segments for fitting baselines.

Created on July 3, 2021
@author: Donald Erb

"""

from math import ceil
import warnings

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, grey_dilation, grey_erosion, uniform_filter1d
)

from ._algorithm_setup import _get_vander, _optimize_window, _setup_classification
from ._compat import jit
from .utils import ParameterWarning, _convert_coef, _interp_inplace, pad_edges, relative_difference


def _remove_single_points(mask):
    """
    Removes lone True or False values from a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The boolean array designating baseline points as True and peak points as False.

    Returns
    -------
    numpy.ndarray
        The input mask after removing lone True and False values.

    Notes
    -----
    Removes the lone True values first since True values designate the baseline.
    That way, the approach is more conservative with assigning baseline points.

    """
    # convert lone True values to False
    # same (check) as baseline_mask ^ binary_erosion(~baseline_mask, [1, 0, 1], border_value=1)
    temp = binary_erosion(mask, [1, 1, 0]) | binary_erosion(mask, [0, 1, 1])
    # convert lone False values to True
    return temp | binary_erosion(temp, [1, 0, 1])


def _find_peak_segments(mask):
    """
    Identifies the peak starts and ends from a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The boolean mask with peak points as 0 or False and baseline points
        as 1 or True.

    Returns
    -------
    peak_starts : numpy.ndarray
        The array identifying the indices where each peak begins.
    peak_ends : numpy.ndarray
        The array identifying the indices where each peak ends.

    """
    extended_mask = np.concatenate(([True], mask, [True]))
    peak_starts = extended_mask[1:-1] < extended_mask[:-2]
    peak_starts = np.flatnonzero(peak_starts)
    if peak_starts.size:
        peak_starts[1 if peak_starts[0] == 0 else 0:] -= 1

    peak_ends = extended_mask[1:-1] < extended_mask[2:]
    peak_ends = np.flatnonzero(peak_ends)
    if peak_ends.size:
        peak_ends[:-1 if peak_ends[-1] == mask.shape[0] - 1 else None] += 1

    return peak_starts, peak_ends


def _averaged_interp(x, y, mask, interp_half_window=0):
    """
    Averages each anchor point and then interpolates between segments.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values.
    y : numpy.ndarray
        The y-values.
    mask : numpy.ndarray
        A boolean array with 0 or False designating peak points and 1 or True
        designating baseline points.
    interp_half_window : int, optional
        The half-window to use for averaging around the anchor points before interpolating.
        Default is 0, which uses just the anchor point value.

    Returns
    -------
    output : numpy.ndarray
        A copy of the input `y` array with peak values in `mask` calulcated using linear
        interpolation.

    """
    output = y.copy()
    mask_sum = mask.sum()
    if not mask_sum:  # all points belong to peaks
        # will just interpolate between first and last points
        warnings.warn('there were no baseline points found', ParameterWarning)
    elif mask_sum == mask.shape[0]:  # all points belong to baseline
        warnings.warn('there were no peak points found', ParameterWarning)
        return output

    peak_starts, peak_ends = _find_peak_segments(mask)
    num_y = y.shape[0]
    for start, end in zip(peak_starts, peak_ends):
        left_mean = np.mean(
            y[max(0, start - interp_half_window):min(start + interp_half_window + 1, num_y)]
        )
        right_mean = np.mean(
            y[max(0, end - interp_half_window):min(end + interp_half_window + 1, num_y)]
        )
        _interp_inplace(x[start:end + 1], output[start:end + 1], left_mean, right_mean)

    return output


def golotvin(data, x_data=None, half_window=None, num_std=2.0, sections=32,
             smooth_half_window=None, interp_half_window=5, weights=None, **pad_kwargs):
    """
    Golotvin's method for identifying baseline regions.

    Divides the data into sections and takes the minimum standard deviation of all
    sections as the noise standard deviation for the entire data. Then classifies any point
    where the rolling max minus min is less than ``num_std * noise standard deviation``
    as belonging to the baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window to use for the rolling maximum and rolling minimum calculations.
        Should be approximately equal to the full-width-at-half-maximum of the peaks or
        features in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    sections : int, optional
        The number of sections to divide the input data into for finding the minimum
        standard deviation.
    smooth_half_window : int, optional
        The half window to use for smoothing the interpolated baseline with a moving average.
        Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from the moving average smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    References
    ----------
    Golotvin, S., et al. Improved Baseline Recognition and Modeling of
    FT NMR Spectra. Journal of Magnetic Resonance. 2000, 146, 122-125.

    """
    y, x, weight_array, *_ = _setup_classification(data, x_data, weights)
    if half_window is None:
        # _optimize_window(y) / 2 gives an "okay" estimate that at least scales
        # with data size
        half_window = ceil(_optimize_window(y) / 2)
    if smooth_half_window is None:
        smooth_half_window = half_window
    num_y = y.shape[0]
    min_sigma = np.inf
    for i in range(sections):
        # use ddof=1 since sampling subsets of the data
        min_sigma = min(
            min_sigma,
            np.std(y[i * num_y // sections:((i + 1) * num_y) // sections], ddof=1)
        )

    mask = (
        grey_dilation(y, 2 * half_window + 1) - grey_erosion(y, 2 * half_window + 1)
    ) < num_std * min_sigma
    mask = _remove_single_points(mask) & weight_array

    rough_baseline = _averaged_interp(x, y, mask, interp_half_window)
    baseline = uniform_filter1d(
        pad_edges(rough_baseline, smooth_half_window, **pad_kwargs),
        2 * smooth_half_window + 1
    )[smooth_half_window:num_y + smooth_half_window]

    return baseline, {'mask': mask}


def dietrich(data, x_data=None, smooth_half_window=None, num_std=3.0,
             interp_half_window=5, poly_order=5, max_iter=50, tol=1e-3, weights=None,
             return_coef=False, **pad_kwargs):
    """
    Dietrich's method for identifying baseline regions.

    Calculates the power spectrum of the data as the squared derivative of the data.
    Then baseline points are identified by iteratively removing points where the mean
    of the power spectrum is less than `num_std` times the standard deviation of the
    power spectrum.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    smooth_half_window : int, optional
        The half window to use for smoothing the input data with a moving average.
        Default is None, which will use N / 256. Set to 0 to not smooth the data.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    poly_order : int, optional
        The polynomial order for fitting the identified baseline. Default is 5.
    max_iter : int, optional
        The maximum number of iterations for fitting a polynomial to the identified
        baseline. If `max_iter` is 0, the returned baseline will be just the linear
        interpolation of the baseline segments. Default is 50.
    tol : float, optional
        The exit criteria for fitting a polynomial to the identified baseline points.
        Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input `x_data` and return them in the params dictionary.
        Default is False, since the conversion takes time.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.
        * 'coef': numpy.ndarray, shape (poly_order,)
            Only if `return_coef` is True and `max_iter` is greater than 0. The array
            of polynomial coefficients for the baseline, in increasing order. Can be
            used to create a polynomial using numpy.polynomial.polynomial.Polynomial().

    Notes
    -----
    When choosing parameters, first choose a `smooth_half_window` that appropriately
    smooths the data, and then reduce `num_std` until no peak regions are included in
    the baseline. If no value of `num_std` works, change `smooth_half_window` and repeat.

    If `max_iter` is 0, the baseline is simply a linear interpolation of the identified
    baseline points. Otherwise, a polynomial is iteratively fit through the baseline
    points, and the interpolated sections are replaced each iteration with the polynomial
    fit.

    References
    ----------
    Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
    Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

    """
    y, x, weight_array, original_domain = _setup_classification(data, x_data, weights)
    num_y = y.shape[0]

    if smooth_half_window is None:
        smooth_half_window = ceil(num_y / 256)
    smooth_y = uniform_filter1d(
        pad_edges(y, smooth_half_window, **pad_kwargs),
        2 * smooth_half_window + 1
    )[smooth_half_window:num_y + smooth_half_window]
    power = np.diff(np.concatenate((smooth_y[:1], smooth_y)))**2
    mask = power < np.mean(power) + num_std * np.std(power, ddof=1)
    old_mask = np.ones_like(mask)
    while not np.array_equal(mask, old_mask):
        old_mask = mask
        masked_power = power[mask]
        if masked_power.size < 2:  # need at least 2 points for std calculation
            warnings.warn(
                'not enough baseline points found; "num_std" is likely too low',
                ParameterWarning
            )
            break
        mask = power < np.mean(masked_power) + num_std * np.std(masked_power, ddof=1)

    mask = _remove_single_points(mask) & weight_array
    rough_baseline = _averaged_interp(x, y, mask, interp_half_window)

    params = {'mask': mask}
    baseline = rough_baseline
    if max_iter > 0:
        vander, pseudo_inverse = _get_vander(x, poly_order)
        old_coef = coef = np.dot(pseudo_inverse, rough_baseline)
        baseline = np.dot(vander, coef)
        for i in range(max_iter - 1):
            rough_baseline[mask] = baseline[mask]
            coef = np.dot(pseudo_inverse, rough_baseline)
            baseline = np.dot(vander, coef)
            if relative_difference(old_coef, coef) < tol:
                break
            old_coef = coef

        if return_coef:
            params['coef'] = _convert_coef(coef, original_domain)

    return baseline, params


@jit(nopython=True, cache=True)
def _rolling_std(data, half_window, ddof=0):
    """
    Computes the rolling standard deviation of an array.

    Parameters
    ----------
    data : numpy.ndarray
        The array for the calculation. Should be padded on the left and right
        edges by `half_window`.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int, optional
        The delta degrees of freedom for the calculation. Default is 0.

    Returns
    -------
    numpy.ndarray
        The array of the rolling standard deviation for each window.

    Notes
    -----
    This implementation is a version of Welford's method [1]_, modified for a
    fixed-length window [2]_. It is slightly modified from the version in [2]_
    in that it assumes the data is padded on the left and right. Other deviations
    from [2]_ are noted within the function.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    .. [2] Chmielowiec, A. Algorithm for error-free determination of the variance of all
           contiguous subsequences and fixed-length contiguous subsequences for a sequence
           of industrial measurement data. Computational Statistics. 2021, 1-28.

    """
    window_size = half_window * 2 + 1
    num_y = data.shape[0]
    squared_diff = np.zeros(num_y)
    mean = data[0]
    # fill the first window
    for i in range(1, window_size):
        val = data[i]
        size_factor = i / (i + 1)
        squared_diff[i] = squared_diff[i - 1] + 2 * size_factor * (val - mean)**2
        mean = mean * size_factor + val / (i + 1)
    # at this point, mean == np.mean(data[:window_size])

    # update squared_diff[half_window] with squared_diff[window_size - 1] / 2; if
    # this isn't done, all values within [half_window:-half_window] in the output are
    # off; no idea why... but it works
    squared_diff[half_window] = squared_diff[window_size - 1] / 2
    for j in range(half_window + 1, num_y):
        old_val = data[j - half_window - 1]
        new_val = data[j + half_window]
        val_diff = new_val - old_val  # reference divided by window_size here

        new_mean = mean + val_diff / window_size
        squared_diff[j] = squared_diff[j - 1] + val_diff * (old_val + new_val - mean - new_mean)
        mean = new_mean

    return np.sqrt(squared_diff / (window_size - ddof))


def std_distribution(data, x_data=None, half_window=None, interp_half_window=5,
                     fill_half_window=3, num_std=1.1, smooth_half_window=None,
                     weights=None, **pad_kwargs):
    """
    Identifies baseline segments by analyzing the rolling standard deviation distribution.

    The rolling standard deviations are split into two distributions, with the smaller
    distribution assigned to noise. Baseline points are then identified as any point
    where the rolled standard deviation is less than a multiple of the median of the
    noise's standard deviation distribution.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window to use for the rolling standard deviation calculation. Should
        be approximately equal to the full-width-at-half-maximum of the peaks or features
        in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    fill_half_window : int, optional
        When a point is identified as a peak point, all points +- `fill_half_window`
        are likewise set as peak points. Default is 3.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 1.1.
    smooth_half_window : int, optional
        The half window to use for smoothing the interpolated baseline with a moving average.
        Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from the moving average smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    References
    ----------
    Wang, K.C., et al. Distribution-Based Classification Method for Baseline
    Correction of Metabolomic 1D Proton Nuclear Magnetic Resonance Spectra.
    Analytical Chemistry. 2013, 85, 1231-1239.

    """
    y, x, weight_array, _ = _setup_classification(data, x_data, weights)
    if half_window is None:
        # _optimize_window(y) / 2 gives an "okay" estimate that at least scales
        # with data size
        half_window = ceil(_optimize_window(y) / 2)
    if smooth_half_window is None:
        smooth_half_window = half_window

    # use dof=1 since sampling a subset of the data; reflect the data since the
    # standard deviation calculation requires noisy data to work
    std = _rolling_std(np.pad(y, half_window, 'reflect'), half_window, 1)[half_window:-half_window]
    median = np.median(std)
    median_2 = np.median(std[std < 2 * median])  # TODO make the 2 an input?
    while median_2 / median < 0.999:  # TODO make the 0.999 an input?
        median = median_2
        median_2 = np.median(std[std < 2 * median])
    noise_std = median_2

    # use ~ to convert from peak==1, baseline==0 to peak==0, baseline==1; if done before,
    # would have to do ~binary_dilation(~mask) or binary_erosion(np.hstack((1, mask, 1))[1:-1]
    mask = np.logical_and(
        ~binary_dilation(std > num_std * noise_std, np.ones(2 * fill_half_window + 1)),
        weight_array
    )

    rough_baseline = _averaged_interp(x, y, mask, interp_half_window)

    baseline = uniform_filter1d(
        pad_edges(rough_baseline, smooth_half_window, **pad_kwargs),
        2 * smooth_half_window + 1
    )[smooth_half_window:y.shape[0] + smooth_half_window]

    return baseline, {'mask': mask}
