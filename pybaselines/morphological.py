# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

 Morphological
    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) iamor (Iterative averaging morphological)

Created on March 5, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import grey_closing, grey_dilation, grey_erosion, grey_opening
from scipy.sparse.linalg import spsolve

from .utils import _setup_whittaker, mollify, relative_difference


def _smooth(data, x_data=None, use_whittaker=True, lam=1e6,
            order=2, weights=None):
    """
    Smooth a baseline using either Whittaker smoothing.

    Parameters
    ----------
    data : [type]
        [description]
    x_data : [type], optional
        [description], by default None
    use_whittaker : bool, optional
        [description], by default True
    lam : [type], optional
        [description], by default 1e6
    order : int, optional
        [description], by default 2

    Returns
    -------
    [type]
        [description]
    """
    y = np.asarray(data)
    if use_whittaker:
        _, D, W, w = _setup_whittaker(y, lam, order, weights)
        z = spsolve(W + D, w * y)
    else:
        raise ValueError  #TODO need any other smoothers? maybe add the mollify/convolution here
    return z


def optimize_window(data, increment=1, max_hits=1, window_tol=1e-6, max_half_window=None):
    """
    Optimizes the morphological window size.

    Parameters
    ----------
    data : array-like, shape (N,)
        The measured data values.
    increment : int, optional
        The step size for iterating half windows. Default is 1.
    max_hits : int, optional
        The number of consecutive half windows that must produce the same
        morphological opening before accepting the half window as the optimum
        value. Default is 1.
    window_tol : float, optional
        The tolerance value for considering two morphological openings as
        equivalent. Default is 1e-6.
    max_half_window : int, optional
        The maximum allowable window size. If None (default), will be set
        to (len(data) - 1) / 2.

    Returns
    -------
    half_window : int
        The optimized half window size.

    References
    ----------
    Dai, L., et al.. An Automated Baseline Correction Method Based on Iterative
    Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

    Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

    """
    y = np.asarray(data)
    if max_half_window is None:
        max_half_window = (y.shape[0] - 1) / 2

    opening = grey_opening(y, [3])  # half window = 1 to start
    hits = 0
    best_half_window = 1
    for half_window in range(2, max_half_window, increment):
        new_opening = grey_opening(y, [half_window * 2 + 1])
        if relative_difference(opening, new_opening) < window_tol:
            if hits == 0:
                # keep just the first window that fits tolerance
                best_half_window = half_window - increment
            hits += 1
            if hits >= max_hits:
                half_window = best_half_window
                break
        elif hits:
            hits = 0
        opening = new_opening

    return half_window


def _setup_morphology(data, half_window=None, **window_kwargs):
    """
    Sets the starting parameters for morphology-based methods.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using pybaselines.morphological.optimize_window if
        **window_kwargs has values, otherwise will use the number of data points
        divided by 5.
    **window_kwargs
        Keyword arguments to pass to pybaselines.morphological.optimize_window.
        Possible items are:

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

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    window : int
        The accepted half window size.

    Notes
    -----
    Ensures that window size is odd since morphological operations operate in
    the range [-window, ..., window].

    Half windows are dealt with rather than full window sizes to clarify their
    usage. SciPy morphology operations deal with absolute window sizes.

    """
    y = np.asarray(data)
    if half_window is not None:
        window = half_window
    elif window_kwargs:
        window = optimize_window(y, **window_kwargs)
    else:
        window = y.shape[0] // 5

    if window % 2 == 0:
        window += 1

    return y, window


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


def mpls(data, lam=1e6, p=0.0, order=2, tol=1e-3, max_iter=50, weights=None, **window_kwargs):
    """
    The Morphological penalized least squares (MPLS) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Residuals
        above the data will be given p weight, and residuals below the data
        will be given p-1 weight. Default is 0.0.
    order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
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

            * 'half_window': int
                The half-window used for the morphology functions. If a value is input,
                then that value will be used. Default is None, which will optimize the
                half-window size using pybaselines.morphological.optimize_window if
                `increment`, `max_hits`, `window_tol`, or `max_half_window` are
                specified, otherwise will use the number of data points
                divided by 5.
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

    Returns
    -------
    numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'half_window': int
            The half window used for the morphological calculations.

    Notes
    -----
    Although the MPLS is technically a penalized least squares algorithm,
    it is included in pybaselines.morphological rather than
    pybaselines.penalized_least_squares.

    References
    ----------
    .. [1] Li, Zhong, et al. Morphological weighted penalized least squares for
           background correction. Analyst, 2013, 138, 4483-4492.

    """
    y, half_window = _setup_morphology(data, **window_kwargs)
    if weights is not None:
        w = weights
    else:
        bkg = grey_opening(y, [2 * half_window + 1])
        diff = np.diff(bkg, prepend=bkg[0], append=bkg[-1])
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

    return (
        _smooth(y, use_whittaker=True, lam=lam, order=order, weights=w),
        {'weights': w, 'half_window': half_window}
    )


def mor(data, smooth=False, smooth_kwargs=None, **window_kwargs):
    """
    A Morphological based (Mor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    smooth : bool, optional
        If True, will smooth the obtained baseline with Whittaker smoothing.
        Default is False.
    smooth_kwargs : dict, optional
        A dictionary of arguments for smoothing if `smooth` is True. Default is None.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'half_window': int
                The half-window used for the morphology functions. If a value is input,
                then that value will be used. Default is None, which will optimize the
                half-window size using pybaselines.morphological.optimize_window if
                `increment`, `max_hits`, `window_tol`, or `max_half_window` are
                specified, otherwise will use the number of data points
                divided by 5.
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

    Returns
    -------
    z : numpy.ndarray, shape (N,)
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
    y, half_window = _setup_morphology(data, **window_kwargs)
    opening = grey_opening(y, [2 * half_window + 1])
    z = np.minimum(opening, _avg_opening(y, half_window, opening))
    if smooth:
        smooth_kws = smooth_kwargs if smooth_kwargs is not None else {}
        z = _smooth(z, **smooth_kws)

    return z, {'half_window': half_window}


def imor(data, tol=1e-3, max_iter=200, smooth=False, **window_kwargs):
    """
    An Improved Morphological based (IMor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    smooth : bool, optional
        If True, will smooth the obtained baseline with mollification.
        Default is False.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'half_window': int
                The half-window used for the morphology functions. If a value is input,
                then that value will be used. Default is None, which will optimize the
                half-window size using pybaselines.morphological.optimize_window if
                `increment`, `max_hits`, `window_tol`, or `max_half_window` are
                specified, otherwise will use the number of data points
                divided by 5.
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

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    Notes
    -----
    Can optionally smooth the obtained background with mollification, which
    makes the result more comparable to the iaMor algorith (Chen, H., et al.).

    References
    ----------
    Dai, L., et al.. An Automated Baseline Correction Method Based on Iterative
    Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

    """
    y, half_window = _setup_morphology(data, **window_kwargs)
    window_size = 2 * half_window + 1
    if smooth:
        kernel = np.array([
            np.exp(-1 / (1 - (i / window_size)**2)) for i in range(-half_window + 1, half_window)
        ])
        kernel = kernel / kernel.sum()
    z = np.minimum(y, _avg_opening(y, window_size))
    for _ in range(max_iter - 1):
        z_new = np.minimum(y, _avg_opening(z, half_window))
        if smooth:
            z_new = mollify(z_new, kernel)

        if relative_difference(z, z_new) < tol:
            break
        z = z_new

    return z, {'half_window': half_window}


def iamor(data, tol=1e-3, max_iter=200, **window_kwargs):
    """
    Iteratively averaging morphological (iaMor) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'half_window': int
                The half-window used for the morphology functions. If a value is input,
                then that value will be used. Default is None, which will optimize the
                half-window size using pybaselines.morphological.optimize_window if
                `increment`, `max_hits`, `window_tol`, or `max_half_window` are
                specified, otherwise will use the number of data points
                divided by 5.
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

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    Notes
    -----
    There is no established name for this method, so just used iaMor.

    References
    ----------
    Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

    """
    y, half_window = _setup_morphology(data, **window_kwargs)
    window_size = 2 * half_window + 1
    #TODO does window_size need to actually be the delta-x values rather
    # than the index-based window size, similar to loess? ie: x[2*half_window+1] - x[0]
    kernel = np.array([
        np.exp(-1 / (1 - (i / window_size)**2)) for i in range(-half_window + 1, half_window)
    ])
    # normalize the kernel before convolution so that can just do
    # np.convolve(y, normalized_kernel) rather than
    # np.convolve(y, kernel) / np.convolve(np.ones(y.shape[0]), kernel)
    kernel = kernel / kernel.sum()

    z = y
    for _ in range(max_iter):
        z_new = mollify(
            np.minimum(
                y,
                0.5 * (grey_closing(z, [window_size]) + grey_opening(z, [window_size]))
            ),
            kernel
        )
        if relative_difference(z, z_new) < tol:
            break
        z = z_new

    return z, {'half_window': half_window}
