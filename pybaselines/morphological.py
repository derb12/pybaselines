# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Baseline fitting techniques can be grouped accordingly (note: when a method
is labelled as 'improved', that is the method's name, not my editorialization):

a) Penalized least squares
    1) AsLS (Asymmetric Least Squares)
    2) IAsLS (Improved Asymmetric Least Squares)
    3) airPLS ()
    5) MPLS (Morphological Penalized Least Squares)
    4) arPLS
    5) drPLS
    6) IarPLS
    7) asPLS

b) Morphological
    1) MPLS (Morphological Penalized Least Squares)
    2) Mor (Morphological)
    3) IMor (Improved Morphological)

c) Polynomial
    1) ModPoly (Modified Polynomial)
    2) IModPoly (Improved Modified Polynomial)

Created on March 5, 2021

@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import grey_closing, grey_dilation, grey_erosion, grey_opening
from scipy.sparse.linalg import spsolve

from .utils import _setup_pls, mollify, relative_difference


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
        D, W, w = _setup_pls(y.shape[0], lam, order, weights)
        z = spsolve(W + D, w * y)
    return z


def _get_window(data, half_window=None, **window_kwargs):
    """
    [summary]

    Parameters
    ----------
    data : np.ndarray, shape (M,)
        [description]
    half_window : int, optional
        [description], by default None
    **window_kwargs
        Keyword arguments to pass to pybaselines.morphological.optimize_window.
        Keys include.

    Returns
    -------
    window : int
        The accepted half window size.

    Notes
    -----
    Ensures that window size is odd since morphological operations operate in
    the range [-window, ..., window].

    Half windows are dealt with rather than full window sizes to clarify their
    usage. SciPy morphology operations deal with absolute window sizes.

    """
    if half_window is not None:
        window = half_window
    elif window_kwargs:
        window = optimize_window(data, **window_kwargs)
    else:
        window = data.shape[0] // 5

    if window % 2 == 0:
        window += 1

    return window


def _avg_opening(y, half_window, opening=None):
    """
    Averages the dilation and erosion of an opening on data.

    Adapted from:
    Perez-Pueyo, R., et al., Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments, Applied Spectroscopy 64 (2010) 595-600.

    Parameters
    ----------
    y : np.ndarray
        The array of the measured data.
    half_window : int, optional
        The half window size to use for the operations.
    opening : np.ndarray, optional
        The output of scipy.ndimage.grey_opening(y, window_size). Default is
        None, which will compute the value.

    Returns
    -------
    np.ndarray
        The average of the dilation and erosion of the opening.

    """
    window_size = 2 * half_window + 1
    if opening is None:
        opening = grey_opening(y, [window_size])
    return 0.5 * (grey_dilation(opening, [window_size]) + grey_erosion(opening, [window_size]))


def optimize_window(data, increment=1, max_hits=1, window_tol=1e-6, max_half_window=None):
    """
    Optimizes the morphological window size.

    Parameters
    ----------
    data : array-like
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
    Dai, L., et al., An Automated Baseline Correction Method Based on Iterative
    Morphological Operations, Applied Spectroscopy 72(5) (2018) 731-739.

    Chen, H., et al., An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications, Applied Spectroscopy 73(3) (2019) 284-293.

    """
    if max_half_window is None:
        max_half_window = (data.shape[0] - 1) / 2

    opening = grey_opening(data, [3])  # half window = 1 to start
    hits = 0
    best_half_window = 1
    for half_window in range(2, max_half_window, increment):
        new_opening = grey_opening(data, [half_window * 2 + 1])
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


def mpls(data, lam=1e6, p=0.0, order=2, tol=0.001, max_iter=500, **window_kwargs):
    """
    The Morphological penalized least squares (MPLS) baseline algorithm.

    Adapted from:
    Li, Zhong, et al., Morphological weighted penalized least squares for
    background correction, Analyst 138 (2013) 4483-4492.

    Notes
    -----
    Although the MPLS is technically a penalized least squares algorithm,
    it is included in pybaselines.morphological rather than
    pybaselines.penalized_least_squares.

    """
    half_window = _get_window(data, **window_kwargs)
    bkg = grey_opening(data, [2 * half_window + 1])
    diff = np.diff(bkg, prepend=bkg[0], append=bkg[-1])
    # diff == 0 means the point is on a flat segment, and diff != 0 means the
    # adjacent point is not the same flat segment. The union of the two finds
    # the endpoints of each segment, and np.flatnonzero converts the mask to
    # indices; indices will always be even-sized.
    indices = np.flatnonzero(
        ((diff[1:] == 0) | (diff[:-1] == 0)) & ((diff[1:] != 0) | (diff[:-1] != 0))
    )
    w = np.full(data.shape[0], p)
    # find the index of min(y) in the region between flat regions
    for previous_segment, next_segment in zip(indices[1::2], indices[2::2]):
        index = np.argmin(data[previous_segment:next_segment + 1]) + previous_segment
        w[index] = 1 - p

    return (
        _smooth(data, use_whittaker=True, lam=lam, order=order, weights=w),
        {'weights': w, 'half_window': half_window}
    )


def mor(data, smooth=False, smooth_kwargs=None, **window_kwargs):
    """
    A Morphological based (Mor) baseline algorithm.

    Adapted from:
    Perez-Pueyo, R., et al., Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments, Applied Spectroscopy 64 (2010) 595-600.

    Can optionally smooth the obtained background with Whittaker smoothing.

    """
    half_window = _get_window(data, **window_kwargs)
    opening = grey_opening(data, [2 * half_window + 1])
    z = np.minimum(opening, _avg_opening(data, half_window, opening))
    if smooth:
        smooth_kws = smooth_kwargs if smooth_kwargs is not None else {}
        z = _smooth(z, **smooth_kws)

    return z, {'half_window': half_window}


def imor(data, tol=1e-3, max_iter=200, smooth=False, **window_kwargs):
    """
    An Improved Morphological based (IMor) baseline algorithm.

    Adapted from:
    Dai, L., et al., An Automated Baseline Correction Method Based on Iterative
    Morphological Operations, Applied Spectroscopy 72(5) (2018) 731-739.

    Can optionally smooth the obtained background with mollification, which
    makes the result more comparable to the iaMor algorith (Chen, H., et al.).

    """
    half_window = _get_window(data, **window_kwargs)
    window_size = 2 * half_window + 1
    if smooth:
        kernel = np.array([
            np.exp(-1 / (1 - (i / window_size)**2)) for i in range(-half_window + 1, half_window)
        ])
        kernel = kernel / kernel.sum()
    z = np.minimum(data, _avg_opening(data, window_size))
    for _ in range(max_iter - 1):
        z_new = np.minimum(data, _avg_opening(z, half_window))
        if smooth:
            z_new = mollify(z_new, kernel)

        if relative_difference(z, z_new) < tol:
            break
        z = z_new

    return z, {'half_window': half_window}


def iamor(data, tol=1e-3, max_iter=200, **window_kwargs):
    """
    Iteratively averaging morphological (iaMor) baseline.

    There is no established name for this method, so just used iaMor.

    Adapted from:
    Chen, H., et al., An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications, Applied Spectroscopy 73(3) (2019) 284-293.

    """
    half_window = _get_window(data, **window_kwargs)
    window_size = 2 * half_window + 1
    kernel = np.array([
        np.exp(-1 / (1 - (i / window_size)**2)) for i in range(-half_window + 1, half_window)
    ])
    # normalize the kernel before convolution so that can just do
    # np.convolve(y, normalized_kernel) rather than
    # np.convolve(y, kernel) / np.convolve(np.ones(y.shape[0]), kernel)
    kernel = kernel / kernel.sum()

    z = data
    for _ in range(max_iter):
        z_new = mollify(
            np.minimum(
                data,
                0.5 * (grey_closing(z, [window_size]) + grey_opening(z, [window_size]))
            ),
            kernel
        )
        if relative_difference(z, z_new) < tol:
            break
        z = z_new

    return z, {'half_window': half_window}
