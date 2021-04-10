# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Whittaker
    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)
    8) psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)

Created on Sept. 13, 2019
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

from ._algorithm_setup import (_setup_polynomial, _setup_whittaker,
                               difference_matrix)
from .utils import _MIN_FLOAT, relative_difference


def asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using asymmetric least squared (AsLS) fitting.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Default is 1e-2.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    Notes
    -----
    Algorithm initially developed in [1]_ and [2]_, and code was adapted from
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    References
    ----------
    .. [1] Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14),
           3631-3636.
    .. [2] Eilers, P., et al. Baseline correction with asymmetric least squares
           smoothing. Leiden University Medical Centre Report, 2005, 1(1).

    """
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        mask = (y > z)
        w_new = p * mask + (1 - p) * (~mask)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    residual = y - z
    return z, {'roughness': z.T * D * z, 'fidelity': residual.T * W * residual, 'weights': w}


def iasls(data, x_data=None, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm.

    The algorithm consideres both the first and second derivatives of the residual.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Default is 1e-2.
    lam_1 : float, optional
        The smoothing parameter for the first derivative. Default is 1e-4.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be set by fitting the data with a second order polynomial.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    References
    ----------
    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    """
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    y, x, *_ = _setup_polynomial(data, x_data)
    if weights is None:
        z = np.polynomial.Polynomial.fit(x, y, 2)(x)
        mask = (y > z)
        weights = p * mask + (1 - p) * (~mask)

    _, D, W, w = _setup_whittaker(y, lam, 2, weights)
    D_1 = difference_matrix(y.shape[0], 1)
    D_1 = lam_1 * D_1.T * D_1
    for _ in range(max_iter):
        weights_and_d1 = W.T * W + D_1
        z = spsolve(weights_and_d1 + D, weights_and_d1 * y)
        mask = (y > z)
        w_new = p * mask + (1 - p) * (~mask)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    residual = y - z
    params = {
        'roughness': z.T * D * z + residual.T * D_1.T * D_1 * residual,
        'fidelity': residual.T * W * residual, 'weights': w
    }
    return z, params


def airpls(data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

    Parameters
    ----------
    data : array-like
        The y-values of the measured data.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    """
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    y_l1_norm = np.linalg.norm(y, 1)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        residual = y - z
        neg_mask = (residual < 0)
        # same as abs(residual[neg_mask]).sum() since residual[neg_mask] are all negative
        residual_l1_norm = -1 * residual[neg_mask].sum()
        if residual_l1_norm / y_l1_norm < tol:
            break
        w = np.exp(i * abs(residual) / residual_l1_norm) * neg_mask
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': residual.T * W * residual, 'weights': w}


def arpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Asymmetrically reweighted penalized least squares smoothing (ArPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        residual = y - z
        neg_mask = residual < 0
        mean = np.mean(residual[neg_mask])
        std = max(abs(np.std(residual[neg_mask])), _MIN_FLOAT)
        w_new = 1 / (1 + np.exp(2 * (residual - (2 * std - mean)) / std))
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': residual.T * W * residual, 'weights': w}


def drpls(data, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None):
    """
    Doubly reweighted penalized least squares baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    eta : float
        A term for controlling the value of lam; should be between 0 and 1.
        Low values will produce smoother baselines, while higher values will
        more aggressively fit peaks. Default is 0.5.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    y, D, W, w = _setup_whittaker(data, lam, 2, weights)
    D_1 = difference_matrix(y.shape[0], 1)
    D_1 = D_1.T * D_1
    Identity = identity(y.shape[0])
    for i in range(1, max_iter + 1):
        z = spsolve(W + D_1 + (Identity - eta * W) * D, w * y)
        residual = y - z
        neg_mask = residual < 0
        std = max(abs(np.std(residual[neg_mask])), _MIN_FLOAT)
        inner = np.exp(i) * (residual - (2 * std - np.mean(residual[neg_mask]))) / std
        w_new = 0.5 * (1 - (inner / (1 + abs(inner))))
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return (
        z,
        {'roughness': (Identity - eta * W) * (z.T * D * z) + z.T * D_1 * z,
         'fidelity': residual.T * W * residual, 'weights': w}
    )


def iarpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    """
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        residual = y - z
        std = max(abs(np.std(residual[residual < 0])), _MIN_FLOAT)
        inner = np.exp(i) * (residual - 2 * std) / std
        w_new = 0.5 * (1 - (inner / np.sqrt(1 + (inner)**2)))
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': residual.T * W * residual, 'weights': w}


def aspls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None, alpha=None):
    """
    Adaptive smoothness penalized least squares smoothing (asPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.
    alpha : array-like, shape (N,), optional
        An array of values that control the local value of `lam` to better
        fit peak and non-peak regions. If None (default), then the initial values
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using
    adaptive smoothness parameter penalized least squares method.
    Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    if alpha is None:
        alpha_array = np.ones_like(w)
    else:
        alpha_array = np.asarray(alpha).copy()
    # Use a sparse matrix rather than an array for alpha in order to keep sparcity.
    alpha_matrix = diags(alpha_array, format='csr')
    for i in range(1, max_iter + 1):
        z = spsolve(W + alpha_matrix * D, w * y)
        residual = y - z
        std = max(abs(np.std(residual[residual < 0])), _MIN_FLOAT)
        w_new = 1 / (1 + np.exp(2 * (residual - std) / std))
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)
        abs_d = abs(residual)
        alpha_matrix.setdiag(abs_d / np.nanmax(abs_d))

    params = {
        'roughness': z.T * alpha_matrix * D * z, 'fidelity': residual.T * W * residual,
        'weights': w, 'alpha': alpha_matrix.data[0]
    }

    return z, params


def psalsa(data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Default is 0.5.
    k : float, optional
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak. Default is None, which sets `k` to
        one-tenth of the standard deviation of the input data. A large k value
        will produce similar results to :func:`.asls`.
    diff_order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the initial weights
        will be an array with size equal to N and all values set to 1.

    Returns
    -------
    z : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    Notes
    -----
    Similar to the asymmetric least squares (AsLS) algorithm, but applies an
    exponential decay weighting to values greater than the baseline to allow
    using a higher `p` value to better fit noisy data.

    The exit criteria for the original algorithm was to check whether the signs
    of the residuals do not change between two iterations, but the comparison of
    the l2 norms of the weight arrays between iterations is used instead to be
    more comparable to other Whittaker-smoothing-based algorithms.

    References
    ----------
    Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
    for analytical instruments. 2014 IEEE 11th International Multi-Conference on
    Systems, Signals, and Devices, 2014, 1-5.

    """
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    y, D, W, w = _setup_whittaker(data, lam, diff_order, weights)
    if k is None:
        k = np.std(y) / 10

    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        residual = y - z
        mask = residual > 0
        w_new = mask * p * np.exp(-residual / k) + (~mask) * (1 - p)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': residual.T * W * residual, 'weights': w}
