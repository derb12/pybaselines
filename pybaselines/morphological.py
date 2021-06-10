# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Created on March 5, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solveh_banded
from scipy.ndimage import grey_closing, grey_dilation, grey_erosion, grey_opening, uniform_filter1d

from . import utils
from ._algorithm_setup import _optimize_window, _setup_morphology, _setup_whittaker
from .utils import _HAS_PENTAPY, _pentapy_solve, pad_edges, padded_convolve, relative_difference


# make _optimize_window available to users and document it so that
# it is included in the api documentation
def optimize_window(*args, **kwargs):
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
    return _optimize_window(*args, **kwargs)


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


def _avg_opening(y, half_window, opening=None):
    """
    Averages the dilation and erosion of a morphological opening on data.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The array of the measured data.
    half_window : int, optional
        The half window size to use for the operations.
    opening : numpy.ndarray, optional
        The output of scipy.ndimage.grey_opening(y, window_size). Default is
        None, which will compute the value.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The average of the dilation and erosion of the opening.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64 595-600.

    """
    window_size = 2 * half_window + 1
    if opening is None:
        opening = grey_opening(y, [window_size])
    return 0.5 * (grey_dilation(opening, [window_size]) + grey_erosion(opening, [window_size]))


def mpls(data, half_window=None, lam=1e6, p=0.0, diff_order=2, tol=1e-3, max_iter=50,
         weights=None, **window_kwargs):
    """
    The Morphological penalized least squares (MPLS) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Residuals
        above the data will be given p weight, and residuals below the data
        will be given p-1 weight. Default is 0.0.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be
        calculated following the procedure in [1]_.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    .. [1] Li, Zhong, et al. Morphological weighted penalized least squares for
           background correction. Analyst, 2013, 138, 4483-4492.

    """
    y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
    if weights is not None:
        w = weights
    else:
        rough_baseline = grey_opening(y, [2 * half_wind + 1])
        diff = np.diff(np.hstack([rough_baseline[0], rough_baseline, rough_baseline[-1]]))
        # diff == 0 means the point is on a flat segment, and diff != 0 means the
        # adjacent point is not the same flat segment. The union of the two finds
        # the endpoints of each segment, and np.flatnonzero converts the mask to
        # indices; indices will always be even-sized.
        indices = np.flatnonzero(
            ((diff[1:] == 0) | (diff[:-1] == 0)) & ((diff[1:] != 0) | (diff[:-1] != 0))
        )
        w = np.full(y.shape[0], p)
        # find the index of min(y) in the region between flat regions
        for previous_segment, next_segment in zip(indices[1::2], indices[2::2]):
            index = np.argmin(y[previous_segment:next_segment + 1]) + previous_segment
            w[index] = 1 - p

    using_pentapy = _HAS_PENTAPY and diff_order == 2
    _, diagonals, weight_array = _setup_whittaker(
        y, lam, diff_order, w, False, not using_pentapy, using_pentapy
    )
    main_diag_idx = diff_order if using_pentapy else -1
    diagonals[main_diag_idx] = diagonals[main_diag_idx] + weight_array
    if using_pentapy:
        baseline = _pentapy_solve(diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER)
    else:
        baseline = solveh_banded(
            diagonals, weight_array * y, overwrite_ab=True, overwrite_b=True, check_finite=False
        )

    params = {'weights': weight_array, 'half_window': half_wind}
    return baseline, params


def mor(data, half_window=None, **window_kwargs):
    """
    A Morphological based (Mor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

    """
    y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
    opening = grey_opening(y, [2 * half_wind + 1])
    baseline = np.minimum(opening, _avg_opening(y, half_wind, opening))

    return baseline, {'half_window': half_wind}


def imor(data, half_window=None, tol=1e-3, max_iter=200, **window_kwargs):
    """
    An Improved Morphological based (IMor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Dai, L., et al. An Automated Baseline Correction Method Based on Iterative
    Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

    """
    y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
    baseline = np.minimum(y, _avg_opening(y, half_wind))
    for _ in range(max_iter - 1):
        baseline_new = np.minimum(y, _avg_opening(baseline, half_wind))
        if relative_difference(baseline, baseline_new) < tol:
            break
        baseline = baseline_new

    return baseline, {'half_window': half_wind}


def amormol(data, half_window=None, tol=1e-3, max_iter=200, pad_kwargs=None, **window_kwargs):
    """
    Iteratively averaging morphological and mollified (aMorMol) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from convolution.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

    """
    y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
    window_size = 2 * half_wind + 1
    kernel = _mollifier_kernel(window_size)
    data_bounds = slice(window_size, -window_size)

    pad_kws = pad_kwargs if pad_kwargs is not None else {}
    y = pad_edges(y, window_size, **pad_kws)
    baseline = y
    for _ in range(max_iter):
        baseline_new = padded_convolve(
            np.minimum(
                y,
                (grey_closing(baseline, [window_size]) + grey_opening(baseline, [window_size])) / 2
            ),
            kernel
        )
        if relative_difference(baseline[data_bounds], baseline_new[data_bounds]) < tol:
            break
        baseline = baseline_new

    return baseline[data_bounds], {'half_window': half_wind}


def mormol(data, half_window=None, tol=1e-3, max_iter=250, smooth_half_window=None,
           pad_kwargs=None, **window_kwargs):
    """
    Iterative morphological and mollified (MorMol) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    smooth_half_window : int, optional
        The half-window to use for smoothing the data before performing the
        morphological operation. Default is None, which will use a value of 1,
        which gives no smoothing.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from convolution.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Koch, M., et al. Iterative morphological and mollifier-based baseline
    correction for Raman spectra. J Raman Spectroscopy, 2017, 48(2), 336-342.

    """
    y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
    window_size = 2 * half_wind + 1
    kernel = _mollifier_kernel(window_size)
    if smooth_half_window is None:
        smooth_half_window = 1
    smooth_kernel = _mollifier_kernel(smooth_half_window)
    data_bounds = slice(window_size, -window_size)

    pad_kws = pad_kwargs if pad_kwargs is not None else {}
    y = pad_edges(y, window_size, **pad_kws)
    baseline = np.zeros(y.shape[0])
    for _ in range(max_iter):
        y_smooth = padded_convolve(y - baseline, smooth_kernel)
        baseline_new = baseline + padded_convolve(grey_erosion(y_smooth, window_size), kernel)
        if relative_difference(baseline[data_bounds], baseline_new[data_bounds]) < tol:
            break
        baseline = baseline_new

    return baseline[data_bounds], {'half_window': half_wind}


def _changing_rolling_ball(y, half_window):
    """
    Calculates the rolling ball algorithm with a changing window size.

    Parameters
    ----------
    y : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : array-like, shape (N,)
        The array of half-windows to use for the morphology functions.

    Returns
    -------
    rough_baseline : numpy.ndarray, shape (N,)
        The array of the baseline after the morphology functions.

    Raises
    ------
    ValueError
        Raised if y and half_window do not have the same length.

    References
    ----------
    Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
    backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
    109, 209-213.

    """
    num_y = len(y)
    if len(half_window) != num_y:
        raise ValueError('half_window array must be the same size as the data array')

    minimum_y = np.empty(num_y)
    for i, half_win in enumerate(half_window):
        minimum_y[i] = min(y[max(0, i - half_win):min(i + half_win + 1, num_y)])

    rough_baseline = np.empty(num_y)
    for i, half_win in enumerate(half_window):
        rough_baseline[i] = max(minimum_y[max(0, i - half_win):min(i + half_win + 1, num_y)])

    return rough_baseline


def _changing_smooth_window(rough_baseline, smooth_half_window):
    """
    Smooths an array with the rolling average and a changing window size.

    Parameters
    ----------
    rough_baseline : numpy.ndarray, shape (N,)
        The array of the baseline before smoothing, with N data points.
    smooth_half_window : array-like, shape (N,)
        The array of half-windows to use for smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The smoothed baseline array.

    Raises
    ------
    ValueError
        Raised if rough_baseline and smooth_half_window do not have the same length.

    References
    ----------
    Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
    backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
    109, 209-213.

    """
    num_y = rough_baseline.shape[0]
    if len(smooth_half_window) != num_y:
        raise ValueError('smooth_half_window array must be the same size as the data array')

    baseline = np.empty(num_y)
    for i, half_win in enumerate(smooth_half_window):
        baseline_slice = rough_baseline[max(0, i - half_win):min(i + half_win + 1, num_y)]
        baseline[i] = baseline_slice.sum() / baseline_slice.size

    return baseline


def rolling_ball(data, half_window=None, smooth_half_window=None, pad_kwargs=None, **window_kwargs):
    """
    The rolling ball baseline algorithm.

    Applies a minimum and then maximum moving window, and subsequently smooths the
    result, giving a baseline that resembles rolling a ball across the data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int or array-like(int), optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`. If
        an array of integers is input, its length must be equal to N.
    smooth_half_window : int or array-like(int), optional
        The half-window to use for smoothing the data after performing the
        morphological operation. Default is None, which will use the same
        value as used for the morphological operation. If an array of integers
        is input, its length must be equal to N.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from the moving average.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int or numpy.ndarray(int)
            The half window or array of half windows used for the
            morphological calculations.

    Notes
    -----
    To use a changing window size for either the morphological or smoothing
    operations, the half windows must be arrays. Otherwise, the size of the
    rolling ball is assumed to be constant.

    References
    ----------
    Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
    backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
    109, 209-213.

    Liland, K., et al. Optimal Choice of Baseline Correction for Multivariate
    Calibration of Spectra. Applied Spectroscopy, 2010, 64(9), 1007-1016.

    """
    if half_window is None or isinstance(half_window, int):
        y, half_wind = _setup_morphology(data, half_window, **window_kwargs)
        rough_baseline = grey_opening(y, 2 * half_wind + 1)
    else:
        # do not need to convert data to numpy array
        half_wind = half_window
        rough_baseline = _changing_rolling_ball(data, half_wind)

    if smooth_half_window is None:
        smooth_half_window = half_wind

    if isinstance(smooth_half_window, int):
        if smooth_half_window == 0:
            baseline = rough_baseline
        else:
            pad_kws = pad_kwargs if pad_kwargs is not None else {}
            baseline = uniform_filter1d(
                pad_edges(rough_baseline, smooth_half_window, **pad_kws),
                2 * smooth_half_window + 1
            )[smooth_half_window:-smooth_half_window]
    else:
        baseline = _changing_smooth_window(rough_baseline, smooth_half_window)

    return baseline, {'half_window': half_wind}
