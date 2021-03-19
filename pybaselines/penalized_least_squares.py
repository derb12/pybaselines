# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Baseline fitting techniques can be grouped accordingly (note: when a method
is labelled as 'improved', that is the method's name, not editorialization):

Penalized least squares
    1) AsLS (Asymmetric Least Squares)
    2) IAsLS (Improved Asymmetric Least Squares)
    3) airPLS (Adaptive iteratively reweighted penalized least squares)
    4) arPLS (Asymmetrically reweighted penalized least squares)
    5) drPLS (Doubly reweighted penalized least squares)
    6) IarPLS (Improved Asymmetrically reweighted penalized least squares)
    7) asPLS (Adaptive smoothness penalized least squares)

Created on Sep 13, 2019

@author: Donald Erb

"""

import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve

from .utils import _setup_pls, difference_matrix, relative_difference


def asls(data, lam=1e5, p=1e-3, order=2, max_iter=250, tol=1e-3, weights=None):
    """
    Fits the baseline using assymetric least squared (AsLS) fitting.

    Code adapted from
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry. 75(14) (2003)
    3631-3636.

    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report 1(1) (2005).

    """
    y = np.asarray(data)
    D, W, w = _setup_pls(y.shape[0], lam, order, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        w_new = p * (y > z) + (1 - p) * (y < z)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    diff = y - z
    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


def iasls(data, x_data, lam=1e5, p=1e-3, lam_1=1e-4, max_iter=50, tol=1e-3, weights=None, full=False):
    """
    Improved asymmetric least squares (IAsLS).

    References
    ----------
    He, S., et al., Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods 6(12) (2014) 4402-4407.

    """
    y = np.asarray(data)
    x = np.asarray(x_data)
    if weights is None:
        z = np.polynomial.Polynomial.fit(x, y, 2)(x)
        weights = p * (y > z) + (1 - p) * (y < z)

    D, W, w = _setup_pls(y.shape[0], lam, 2, weights)
    D1 = difference_matrix(y.shape[0], 1)
    D1 = lam_1 * D1.T * D1
    for _ in range(max_iter):
        z = spsolve(W.T * W + D1 + D, (W.T * W + D1) * y)
        w_new = p * (y > z) + (1 - p) * (y < z)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    diff = y - z
    return (
        z,
        {'roughness': z.T * D * z + diff.T * D_1.T * D_1 * diff,
         'fidelity': diff.T * W * diff, 'weights': w}
    )


def airpls(data, lam=1e6, order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

    Parameters
    ----------
    data : array-like
        The y-values of the measured data.
    lam : float, optional
        The smoothing parameter. Default is 1e6.
    order : {2, 1, 3}, optional
        The order of the differential matrix. Default is 2 (second order
        differential matrix).
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 0.001.

    Returns
    -------
    z : np.ndarray
        The fitted background vector.

    References
    ----------
    Zhang, Z.M., et al., Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst 135(5) (2010) 1138-1146.

    """
    y = np.asarray(data)
    D, W, w = _setup_pls(y.shape[0], lam, order, weights)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        diff = y - z
        neg_mask = (diff < 0)
        diff_neg_sum = abs(diff[neg_mask].sum())
        if diff_neg_sum / (abs(y)).sum() < tol:
            break
        w[~neg_mask] = 0
        w[neg_mask] = np.exp(i * abs(diff[diff < 0]) / diff_neg_sum)
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


def arpls(data, lam=10**5, order=2, max_iter=500, tol=0.01, weights=None):
    """
    Asymmetrically reweighted penalized least squares smoothing (ArPLS).

    References
    ----------
    Baek, et. al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    y = np.asarray(data)
    D, W, w = _setup_pls(y.shape[0], lam, 2, weights)
    for _ in range(max_iter):
        z = spsolve(W + D, w * y)
        diff = y - z
        mean = np.mean(diff[diff < 0])
        std = max(abs(np.std(diff[diff < 0])), np.finfo(np.float).eps)
        w_new = 1 / (1 + np.exp(2 * (diff - (2 * std - mean)) / std))
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


def drpls(data, lam=1e5, eta=0.5, max_iter=100, tol=1e-3, weights=None):
    """
    Doubly reweighted penalized least squares baseline.

    References
    ----------
    Xu, D. et al., Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics 58 (2019) 3913-3920.

    """
    y = np.asarray(data)

    D, W, w = _setup_pls(y.shape[0], lam, 2, weights)
    D1 = difference_matrix(y.shape[0], 1)
    D1 = D1.T * D1
    Identity = identity(y.shape[0])
    for i in range(1, max_iter + 1):
        z = spsolve(W + D1 + (Identity - eta * W) * D, w * y)
        diff = y - z
        mean = np.mean(diff[diff < 0])
        std = max(abs(np.std(diff[diff < 0])), np.finfo(np.float).eps)
        w_new = 0.5 * (
            1 - ((np.exp(i) * (diff - (2 * std - mean)) / std) / (1 + abs(np.exp(i) * (diff - (2 * std - mean)) / std)))
        )
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return (
        z,
        {'roughness': (Identity - eta * W) * (z.T * D * z) + z.T * D_1 * z,
         'fidelity': diff.T * W * diff, 'weights': w}
    )


def iarpls(data, lam=10**5, order=2, max_iter=500, tol=0.01, weights=None):
    """
    Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

    References
    ----------
    Ye, J. et. al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    """
    y = np.asarray(data)
    D, W, w = _setup_pls(y.shape[0], lam, 2, weights)
    for i in range(1, max_iter + 1):
        z = spsolve(W + D, w * y)
        diff = y - z
        std = max(abs(np.std(diff[diff < 0])), np.finfo(np.float).eps)
        w_new = 0.5 * (
            1 - ((np.exp(i) * (diff - 2 * std) / std) / np.sqrt(1 + (np.exp(i) * (diff - 2 * std) / std)**2))
        )
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)

    return z, {'roughness': z.T * D * z, 'fidelity': diff.T * W * diff, 'weights': w}


def aspls(data, lam=10**5, order=2, max_iter=250, tol=1e-3, weights=None):
    """
    Adaptive smoothness penalized least squares smoothing (asPLS).

    References
    ----------
    Zhang, F. et. al. Baseline correction for infrared spectra using
    adaptive smoothness parameter penalized least squares method.
    Spectroscopy Letters, 53(3) (2020), 222-233.

    """
    y = np.asarray(data)
    D, W, w = _setup_pls(y.shape[0], lam, 2, weights)
    # Use a sparse diagonal matrix rather than an array for alpha in order to keep sparcity.
    alpha = diags(w)
    for i in range(1, max_iter + 1):
        z = spsolve(W + alpha * D, w * y)
        diff = y - z
        std = max(abs(np.std(diff[diff < 0])), np.finfo(np.float).eps) #TODO check whether dof should be 1 rather than 0
        w_new = 1 / (1 + np.exp(2 * (diff - std)) / std)
        if relative_difference(w, w_new) < tol:
            break
        w = w_new
        W.setdiag(w)
        alpha.setdiag(abs(diff) / np.nanmax(abs(diff)))

    return z, {'roughness': z.T * alpha * D * z, 'fidelity': diff.T * W * diff, 'weights': w}
