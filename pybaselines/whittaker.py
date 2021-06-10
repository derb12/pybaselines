# -*- coding: utf-8 -*-
"""Whittaker-smoothing-based techniques for fitting baselines to experimental data.

Created on Sept. 13, 2019
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded

from . import utils
from ._algorithm_setup import _diff_1_diags, _setup_whittaker, _yx_arrays
from .utils import _HAS_PENTAPY, _pentapy_solve, _safe_std, relative_difference


def _shift_rows(matrix, diagonals=2):
    """
    Shifts the rows of a matrix with equal number of upper and lower off-diagonals.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to be shifted. Note that all modifications are done in-place.
    diagonals : int
        The number of upper or lower (same for symmetric matrix) diagonals, not
        including the main diagonal. For example, a matrix with five diagonal rows
        would use a `diagonals` of 2.

    Returns
    -------
    matrix : numpy.ndarray
        The shifted matrix.

    Notes
    -----
    Necessary to match the diagonal matrix format required by SciPy's solve_banded
    function.

    Performs the following transformation (left is input, right is output):

        [[a b c ... d 0 0]        [[0 0 a ... b c d]
         [e f g ... h i 0]         [0 e f ... g h i]
         [j k l ... m n o]   -->   [j k l ... m n o]
         [0 p q ... r s t]         [p q r ... s t 0]
         [0 0 u ... v w x]]        [u v w ... x 0 0]]

    The right matrix would be directly obtained when using SciPy's sparse diagonal
    matrices, but when using multiplication with NumPy arrays, the result is the
    left matrix, which has to be shifted to match the desired format.

    """
    for row, shift in enumerate(range(-diagonals, 0)):
        matrix[row, -shift:] = matrix[row, :shift]
        matrix[row, :-shift] = 0

    for row, shift in enumerate(range(1, diagonals + 1), diagonals + 1):
        matrix[row, :-shift] = matrix[row, shift:]
        matrix[row, -shift:] = 0

    return matrix


def asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using asymmetric least squared (AsLS) fitting.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given p weight, and values less than the baseline
        will be given p-1 weight. Default is 1e-2.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
    if not 0 < p < 1:
        raise ValueError('p must be between 0 and 1')
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, not using_pentapy, using_pentapy
    )
    main_diag_idx = diff_order if using_pentapy else -1
    main_diagonal = diagonals[main_diag_idx].copy()
    for i in range(max_iter):
        diagonals[main_diag_idx] = main_diagonal + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER)
        else:
            baseline = solveh_banded(
                diagonals, weight_array * y, overwrite_b=True, check_finite=False
            )
        mask = y > baseline
        new_weights = p * mask + (1 - p) * (~mask)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params


def iasls(data, x_data=None, lam=1e6, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3, weights=None):
    """
    Fits the baseline using the improved asymmetric least squares (IAsLS) algorithm.

    The algorithm consideres both the first and second derivatives of the residual.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
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
    if not 0 < p < 1:
        raise ValueError('p must be between 0 and 1')
    if weights is None:
        y, x = _yx_arrays(data, x_data)
        baseline = np.polynomial.Polynomial.fit(x, y, 2)(x)
        mask = y > baseline
        weights = p * mask + (1 - p) * (~mask)

        _, d2_diags, weight_array = _setup_whittaker(
            y, lam, 2, weights, False, not _HAS_PENTAPY, _HAS_PENTAPY
        )
    else:
        y, d2_diags, weight_array = _setup_whittaker(
            data, lam, 2, weights, False, not _HAS_PENTAPY, _HAS_PENTAPY
        )

    diagonals = (
        d2_diags
        + _diff_1_diags(y.shape[0], not _HAS_PENTAPY, True)[::-1 if _HAS_PENTAPY else 1]
    )
    main_diag_idx = 2 if _HAS_PENTAPY else -1
    main_diagonal = diagonals[main_diag_idx].copy()

    # lam_1 * D_1.T * D_1 * y
    d1_y = y.copy()
    d1_y[0] = y[0] - y[1]
    d1_y[-1] = y[-1] - y[-2]
    d1_y[1:-1] = 2 * y[1:-1] - y[:-2] - y[2:]
    d1_y = lam_1 * d1_y
    for i in range(max_iter):
        weight_squared = weight_array * weight_array
        diagonals[main_diag_idx] = main_diagonal + weight_squared
        if _HAS_PENTAPY:
            baseline = _pentapy_solve(
                diagonals, weight_squared * y + d1_y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solveh_banded(
                diagonals, weight_squared * y + d1_y, overwrite_b=True, check_finite=False
            )
        mask = y > baseline
        new_weights = p * mask + (1 - p) * (~mask)
        calc_diff = relative_difference(weight_array, new_weights)
        if calc_diff < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params


def airpls(data, lam=1e6, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Adaptive iteratively reweighted penalized least squares (airPLS) baseline.

    Parameters
    ----------
    data : array-like
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, True, not using_pentapy, using_pentapy
    )
    y_l1_norm = np.abs(y).sum()
    main_diag_idx = diff_order if using_pentapy else -1
    main_diagonal = diagonals[main_diag_idx].copy()
    for i in range(1, max_iter + 1):
        diagonals[main_diag_idx] = main_diagonal + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(
                diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solveh_banded(
                diagonals, weight_array * y, overwrite_b=True, check_finite=False
            )
        residual = y - baseline
        neg_mask = residual < 0
        neg_residual = residual[neg_mask]
        # same as abs(neg_residual).sum() since neg_residual are all negative
        residual_l1_norm = -1 * neg_residual.sum()
        calc_diff = residual_l1_norm / y_l1_norm
        if calc_diff < tol:
            break
        # only use negative residual in exp to avoid exponential overflow warnings
        # and accidently creating a weight of nan (inf * 0 = nan)
        weight_array[neg_mask] = np.exp(i * neg_residual / residual_l1_norm)
        weight_array[~neg_mask] = 0

    params = {'weights': weight_array}

    return baseline, params


def arpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Asymmetrically reweighted penalized least squares smoothing (arPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, not using_pentapy, using_pentapy
    )
    main_diag_idx = diff_order if using_pentapy else -1
    main_diagonal = diagonals[main_diag_idx].copy()
    for i in range(max_iter):
        diagonals[main_diag_idx] = main_diagonal + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(
                diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solveh_banded(
                diagonals, weight_array * y, overwrite_b=True, check_finite=False
            )
        residual = y - baseline
        neg_residual = residual[residual < 0]
        std = _safe_std(neg_residual)
        new_weights = 1 / (1 + np.exp(2 * (residual - (2 * std - np.mean(neg_residual))) / std))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params


def drpls(data, lam=1e5, eta=0.5, max_iter=50, tol=1e-3, weights=None):
    """
    Doubly reweighted penalized least squares (drPLS) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
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
    y, d2_diagonals, weight_array = _setup_whittaker(data, lam, 2, weights, False, False)
    d1_d2_diagonals = d2_diagonals + _diff_1_diags(y.shape[0], False, True)
    if _HAS_PENTAPY:
        d1_d2_diagonals = d1_d2_diagonals[::-1]
    # identity - eta * D_2.T * D_2; overwrite d2_diagonals since it's no longer needed;
    # reversed to match the original diagonal structure of the D_2.T * D_2 sparse matrix
    d2_diagonals = -eta * d2_diagonals[::-1]
    d2_diagonals[2] += 1.
    for i in range(1, max_iter + 1):
        d2_w_diagonals = d2_diagonals * weight_array
        if _HAS_PENTAPY:
            baseline = _pentapy_solve(
                d1_d2_diagonals + d2_w_diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solve_banded(
                (2, 2), d1_d2_diagonals + _shift_rows(d2_w_diagonals), weight_array * y,
                overwrite_b=True, overwrite_ab=True, check_finite=False
            )
        residual = y - baseline
        neg_residual = residual[residual < 0]
        std = _safe_std(neg_residual)
        inner = np.exp(i) * (residual - (2 * std - np.mean(neg_residual))) / std
        new_weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params


def iarpls(data, lam=1e5, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Improved asymmetrically reweighted penalized least squares smoothing (IarPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, not using_pentapy, using_pentapy
    )
    main_diag_idx = diff_order if using_pentapy else -1
    main_diagonal = diagonals[main_diag_idx].copy()
    for i in range(1, max_iter + 1):
        diagonals[main_diag_idx] = main_diagonal + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(
                diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solveh_banded(
                diagonals, weight_array * y, overwrite_b=True, check_finite=False
            )
        residual = y - baseline
        std = _safe_std(residual[residual < 0])
        inner = np.exp(i) * (residual - 2 * std) / std
        new_weights = 0.5 * (1 - (inner / np.sqrt(1 + (inner)**2)))
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params


def aspls(data, lam=1e5, diff_order=2, max_iter=100, tol=1e-3, weights=None, alpha=None):
    """
    Adaptive smoothness penalized least squares smoothing (asPLS).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e5.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
        * 'alpha': numpy.ndarray, shape (N,)
            The array of alpha values used for fitting the data in the final iteration.

    Raises
    ------
    ValueError
        Raised if `alpha` and `data` do not have the same shape.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using
    adaptive smoothness parameter penalized least squares method.
    Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, False, True
    )
    if alpha is None:
        alpha_array = np.ones_like(weight_array)
    else:
        alpha_array = np.asarray(alpha)
        if alpha_array.shape != y.shape:
            raise ValueError('alpha must have the same shape as the input data')

    lower_upper = (diff_order, diff_order)
    for i in range(1, max_iter + 1):
        alpha_diagonals = diagonals * alpha_array
        alpha_diagonals[diff_order] = alpha_diagonals[diff_order] + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(
                alpha_diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solve_banded(
                lower_upper, _shift_rows(alpha_diagonals, diff_order), weight_array * y,
                overwrite_ab=True, overwrite_b=True, check_finite=False
            )
        residual = y - baseline
        std = _safe_std(residual[residual < 0])
        new_weights = 1 / (1 + np.exp(2 * (residual - std) / std))
        calc_diff = relative_difference(weight_array, new_weights)
        if calc_diff < tol:
            break
        weight_array = new_weights
        abs_d = np.abs(residual)
        alpha_array = abs_d / abs_d.max()

    params = {'weights': weight_array, 'alpha': alpha_array}

    return baseline, params


def psalsa(data, lam=1e5, p=0.5, k=None, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    Peaked Signal's Asymmetric Least Squares Algorithm (psalsa).

    Similar to the asymmetric least squares (AsLS) algorithm, but applies an
    exponential decay weighting to values greater than the baseline to allow
    using a higher `p` value to better fit noisy data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
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
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
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
    if not 0 < p < 1:
        raise ValueError('p must be between 0 and 1')
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, not using_pentapy, using_pentapy
    )
    if k is None:
        k = np.std(y) / 10

    num_y = y.shape[0]
    main_diag_idx = diff_order if using_pentapy else -1
    main_diagonal = diagonals[main_diag_idx].copy()
    for i in range(max_iter):
        diagonals[main_diag_idx] = main_diagonal + weight_array
        if using_pentapy:
            baseline = _pentapy_solve(
                diagonals, weight_array * y, True, True, utils.PENTAPY_SOLVER
            )
        else:
            baseline = solveh_banded(
                diagonals, weight_array * y, overwrite_b=True, check_finite=False
            )
        residual = y - baseline
        # only use positive residual in exp to avoid exponential overflow warnings
        # and accidently creating a weight of nan (inf * 0 = nan)
        new_weights = np.full(num_y, 1 - p, dtype=float)
        mask = residual > 0
        new_weights[mask] = p * np.exp(-residual[mask] / k)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array}

    return baseline, params
