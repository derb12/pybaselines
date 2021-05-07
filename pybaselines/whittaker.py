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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from ._algorithm_setup import _setup_polynomial, _setup_whittaker, difference_matrix
from .utils import _MIN_FLOAT, PERMC_SPEC, relative_difference


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
    baseline : numpy.ndarray, shape (N,)
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
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, 1(1).

    """
    if p < 0 or p > 1:
        raise ValueError('p must be between 0 and 1')
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        baseline = spsolve(weight_matrix + diff_matrix, weight_array * y, permc_spec=PERMC_SPEC)
        mask = (y > baseline)
        new_weights = p * mask + (1 - p) * (~mask)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    residual = y - baseline
    params = {
        'roughness': baseline.T * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
        baseline = np.polynomial.Polynomial.fit(x, y, 2)(x)
        mask = (y > baseline)
        weights = p * mask + (1 - p) * (~mask)

    _, diff_matrix, weight_matrix, weight_array = _setup_whittaker(y, lam, 2, weights)
    diff_matrix_1 = difference_matrix(y.shape[0], 1)
    diff_matrix_1 = lam_1 * diff_matrix_1.T * diff_matrix_1
    for _ in range(max_iter):
        weights_and_d1 = weight_matrix.T * weight_matrix + diff_matrix_1
        baseline = spsolve(weights_and_d1 + diff_matrix, weights_and_d1 * y, permc_spec=PERMC_SPEC)
        mask = (y > baseline)
        new_weights = p * mask + (1 - p) * (~mask)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    residual = y - baseline
    params = {
        'roughness': baseline.T * diff_matrix * baseline + residual.T * diff_matrix_1 * residual,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }
    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    y_l1_norm = np.linalg.norm(y, 1)
    for i in range(1, max_iter + 1):
        baseline = spsolve(weight_matrix + diff_matrix, weight_array * y, permc_spec=PERMC_SPEC)
        residual = y - baseline
        neg_mask = (residual < 0)
        # same as abs(residual[neg_mask]).sum() since residual[neg_mask] are all negative
        residual_l1_norm = -1 * residual[neg_mask].sum()
        if residual_l1_norm / y_l1_norm < tol:
            break
        weight_array = np.exp(i * residual / residual_l1_norm) * neg_mask
        weight_matrix.setdiag(weight_array)

    params = {
        'roughness': baseline.T * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    for _ in range(max_iter):
        baseline = spsolve(weight_matrix + diff_matrix, weight_array * y, permc_spec=PERMC_SPEC)
        residual = y - baseline
        neg_mask = residual < 0
        mean = np.mean(residual[neg_mask])
        std = max(abs(np.std(residual[neg_mask])), _MIN_FLOAT)
        new_weights = 1 / (1 + np.exp(2 * (residual - (2 * std - mean)) / std))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    params = {
        'roughness': baseline.T * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, 2, weights)
    diff_matrix_1 = difference_matrix(y.shape[0], 1)
    diff_matrix_1 = diff_matrix_1.T * diff_matrix_1
    identity_matrix = difference_matrix(y.shape[0], 0)
    for i in range(1, max_iter + 1):
        baseline = spsolve(
            weight_matrix + diff_matrix_1 + (identity_matrix - eta * weight_matrix) * diff_matrix,
            weight_array * y, permc_spec=PERMC_SPEC
        )
        residual = y - baseline
        neg_mask = residual < 0
        std = max(abs(np.std(residual[neg_mask])), _MIN_FLOAT)
        inner = np.exp(i) * (residual - (2 * std - np.mean(residual[neg_mask]))) / std
        new_weights = 0.5 * (1 - (inner / (1 + abs(inner))))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    params = {
        'roughness': (
            (identity_matrix - eta * weight_matrix) * (baseline.T * diff_matrix * baseline)
            + baseline.T * diff_matrix_1 * baseline
        ),
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    for i in range(1, max_iter + 1):
        baseline = spsolve(weight_matrix + diff_matrix, weight_array * y, permc_spec=PERMC_SPEC)
        residual = y - baseline
        std = max(abs(np.std(residual[residual < 0])), _MIN_FLOAT)
        inner = np.exp(i) * (residual - 2 * std) / std
        new_weights = 0.5 * (1 - (inner / np.sqrt(1 + (inner)**2)))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    params = {
        'roughness': baseline.T * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    if alpha is None:
        alpha_array = np.ones_like(weight_array)
    else:
        alpha_array = np.asarray(alpha).copy()
    # Use a sparse matrix rather than an array for alpha in order to keep sparcity.
    alpha_matrix = diags(alpha_array, format='csr')
    for i in range(1, max_iter + 1):
        baseline = spsolve(
            weight_matrix + alpha_matrix * diff_matrix, weight_array * y, permc_spec=PERMC_SPEC
        )
        residual = y - baseline
        std = max(abs(np.std(residual[residual < 0])), _MIN_FLOAT)
        new_weights = 1 / (1 + np.exp(2 * (residual - std) / std))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)
        abs_d = abs(residual)
        alpha_matrix.setdiag(abs_d / np.nanmax(abs_d))

    params = {
        'roughness': baseline.T * alpha_matrix * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array,
        'alpha': alpha_matrix.data[0]
    }

    return baseline, params


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
    baseline : numpy.ndarray, shape (N,)
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
    y, diff_matrix, weight_matrix, weight_array = _setup_whittaker(data, lam, diff_order, weights)
    if k is None:
        k = np.std(y) / 10

    for _ in range(max_iter):
        baseline = spsolve(weight_matrix + diff_matrix, weight_array * y, permc_spec=PERMC_SPEC)
        residual = y - baseline
        mask = residual > 0
        new_weights = mask * p * np.exp(-residual / k) + (~mask) * (1 - p)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    params = {
        'roughness': baseline.T * diff_matrix * baseline,
        'fidelity': residual.T * weight_matrix * residual, 'weights': weight_array
    }

    return baseline, params
