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

Created on Sept. 13, 2019
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

from . import utils


def asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using assymetric least squared (AsLS) fitting.

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
    y, D, W, w = utils._setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        mask = (y > z)
        w_new = p * mask + (1 - p) * (~mask)
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    diff = y - z
    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


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
        The penalizing weighting factor. Must be between 0 and 1. Residuals
        above the data will be given p weight, and residuals below the data
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
    y, x, *_ = utils._setup_polynomial(data, x_data)
    if weights is None:
        z = np.polynomial.Polynomial.fit(x, y, 2)(x)
        mask = (y > z)
        weights = p * mask + (1 - p) * (~mask)

    _, D, W, w = utils._setup_whittaker(y, lam, 2, weights)
    D_1 = utils.difference_matrix(y.shape[0], 1)
    D_1 = lam_1 * D_1.T * D_1
    for _ in range(max_iter):
        weights_and_d1 = W.T * W + D_1
        z = spsolve(weights_and_d1 + D, weights_and_d1 * y)
        mask = (y > z)
        w_new = p * mask + (1 - p) * (~mask)
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    diff = y - z
    return (
        z,
        {'roughness': z.T * D * z + diff.T * D_1.T * D_1 * diff,
         'fidelity': diff.T * W * diff, 'weights': w}
    )


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
    y, D, W, w = utils._setup_whittaker(data, lam, diff_order, weights)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        diff = y - z
        neg_mask = (diff < 0)
        diff_neg_sum = abs(diff[neg_mask].sum())
        if diff_neg_sum / (abs(y)).sum() < tol:
            break
        w = np.exp(i * abs(diff) / diff_neg_sum) * neg_mask
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


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
    y, D, W, w = utils._setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        diff = y - z
        neg_mask = diff < 0
        mean = np.mean(diff[neg_mask])
        std = max(abs(np.std(diff[neg_mask])), utils._MIN_FLOAT)
        w_new = 1 / (1 + np.exp(2 * (diff - (2 * std - mean)) / std))
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


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
    y, D, W, w = utils._setup_whittaker(data, lam, 2, weights)
    D_1 = utils.difference_matrix(y.shape[0], 1)
    D_1 = D_1.T * D_1
    Identity = identity(y.shape[0])
    for i in range(1, max_iter + 1):
        z = spsolve(W + D_1 + (Identity - eta * W) * D, w * y)
        diff = y - z
        neg_mask = diff < 0
        mean = np.mean(diff[neg_mask])
        std = max(abs(np.std(diff[neg_mask])), utils._MIN_FLOAT)
        w_new = 0.5 * (
            1 - ((np.exp(i) * (diff - (2 * std - mean)) / std) / (1 + abs(np.exp(i) * (diff - (2 * std - mean)) / std)))
        )
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return (
        z,
        {'roughness': (Identity - eta * W) * (z.T * D * z) + z.T * D_1 * z,
         'fidelity': diff.T * W * diff, 'weights': w}
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
    y, D, W, w = utils._setup_whittaker(data, lam, diff_order, weights)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        diff = y - z
        std = max(abs(np.std(diff[diff < 0])), utils._MIN_FLOAT)
        w_new = 0.5 * (
            1 - ((np.exp(i) * (diff - 2 * std) / std) / np.sqrt(1 + (np.exp(i) * (diff - 2 * std) / std)**2))
        )
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


def aspls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
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
    y, D, W, w = utils._setup_whittaker(data, lam, diff_order, weights)
    # Use a sparse diagonal matrix rather than an array for alpha in order to keep sparcity.
    alpha = diags(w)
    for i in range(1, max_iter + 1):
        z = spsolve(W + alpha * D, w * y)
        diff = y - z
        std = max(abs(np.std(diff[diff < 0])), utils._MIN_FLOAT) #TODO check whether dof should be 1 rather than 0
        w_new = 1 / (1 + np.exp(2 * (diff - std) / std))
        if utils.relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)
        alpha.setdiag(abs(diff) / np.nanmax(abs(diff)))

    return z, {'roughness': z.T * alpha * D * z, 'fidelity': diff.T * W * diff, 'weights': w}
