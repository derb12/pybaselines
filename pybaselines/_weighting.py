# -*- coding: utf-8 -*-
"""Contains various weighting schemes used in pybaselines."""

import numpy as np
from scipy.special import expit

from .utils import _MIN_FLOAT


def _asls(y, baseline, p):
    """
    The weighting for the asymmetric least squares algorithm (asls).

    Also used by the improved asymmetric least squares algorithm (iasls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `p - 1` weight.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, 1(1).

    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    """
    mask = y > baseline
    weights = p * mask + (1 - p) * (~mask)
    return weights


def _safe_std(array, **kwargs):
    """
    Calculates the standard deviation and protects against nan and 0.

    Used to prevent propogating nan or dividing by 0.

    Parameters
    ----------
    array : numpy.ndarray
        The array of values for calculating the standard deviation.
    **kwargs
        Additional keyword arguments to pass to :func:`numpy.std`.

    Returns
    -------
    std : float
        The standard deviation of the array, or `_MIN_FLOAT` if the
        calculated standard deviation was 0 or if `array` was empty.

    Notes
    -----
    Does not protect against the calculated standard deviation of a non-empty
    array being nan because that would indicate that nan or inf was within the
    array, which should not be protected.

    """
    # std would be 0 for an array with size of 1 and inf if size <= ddof; only
    # internally use ddof=1, so the second condition is already covered
    if array.size < 2:
        std = _MIN_FLOAT
    else:
        std = array.std(**kwargs)
        if std == 0:
            std = _MIN_FLOAT

    return std


def _arpls(y, baseline):
    """
    The weighting for asymmetrically reweighted penalized least squares smoothing (arpls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(2 / std) * (residual - (2 * std - np.mean(neg_residual))))
    return weights


def _drpls(y, baseline, iteration):
    """
    The weighting for the doubly reweighted penalized least squares algorithm (drpls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    inner = (np.exp(iteration) / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights


def _iarpls(y, baseline, iteration):
    """
    Weighting for improved asymmetrically reweighted penalized least squares smoothing (iarpls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    """
    residual = y - baseline
    std = _safe_std(residual[residual < 0], ddof=1)  # dof=1 since sampling a subset
    inner = (np.exp(iteration) / std) * (residual - 2 * std)
    weights = 0.5 * (1 - (inner / np.sqrt(1 + inner**2)))
    return weights


def _aspls(y, baseline):
    """
    Weighting for the adaptive smoothness penalized least squares smoothing (aspls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.
    residual : numpy.ndarray, shape (N,)
        The residual, ``y - baseline``.

    Notes
    -----
    The weighting uses an asymmetric coefficient (`k` in the asPLS paper) of 0.5 instead
    of the 2 listed in the asPLS paper. pybaselines uses the factor of 0.5 since it
    matches the results in Table 2 and Figure 5 of the asPLS paper closer than the
    factor of 2 and fits noisy data much better.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using adaptive smoothness
    parameter penalized least squares method. Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    residual = y - baseline
    std = _safe_std(residual[residual < 0], ddof=1)  # use dof=1 since sampling a subset
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(0.5 / std) * (residual - std))
    return weights, residual


def _psalsa(y, baseline, p, k, len_y):
    """
    Weighting for the peaked signal's asymmetric least squares algorithm (psalsa).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `p - 1` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.
    len_y : int
        The length of `y`, `N`. Precomputed to avoid repeated calculations.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
    for analytical instruments. 2014 IEEE 11th International Multi-Conference on
    Systems, Signals, and Devices, 2014, 1-5.

    """
    residual = y - baseline
    # only use positive residual in exp to avoid exponential overflow warnings
    # and accidently creating a weight of nan (inf * 0 = nan)
    weights = np.full(len_y, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-residual[mask] / k)
    return weights


def _derpsalsa(y, baseline, p, k, len_y, partial_weights):
    """
    Weights for derivative peak-screening asymmetric least squares algorithm (derpsalsa).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `p - 1` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.
    len_y : int
        The length of `y`, `N`. Precomputed to avoid repeated calculations.
    partial_weights : numpy.ndarray, shape (N,)
        The weights associated with the first and second derivatives of the data.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    Notes
    -----
    The reference is not clear as to how `p` and `1-p` are applied. An alternative could
    be that `partial_weights` are multiplied only where the residual is greater than
    0 and that all other weights are `1-p`, but based on Figure 1c in the reference, the
    total weights are never greater than `partial_weights`, so that must mean the non-peak
    regions have a weight of `1-p` times `partial_weights` rather than just `1-p`;
    both weighting systems give near identical results, so it is not a big deal.

    References
    ----------
    Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
    automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
    51(10), 2061-2065.

    """
    residual = y - baseline
    # no need for caution since inner exponential is always negative, but still mask
    # since it's faster than performing the square and exp on the full residual
    weights = np.full(len_y, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-((residual[mask] / k)**2) / 2)
    weights *= partial_weights
    return weights


def _quantile(y, fit, quantile, eps=None):
    r"""
    An approximation of quantile loss.

    The loss is defined as :math:`\rho(r) / |r|`, where r is the residual, `y - fit`,
    and the function :math:`\rho(r)` is `quantile` for `r` > 0 and 1 - `quantile`
    for `r` < 0. Rather than using `|r|` as the denominator, which is non-differentiable
    and causes issues when `r` = 0, the denominator is approximated as
    :math:`\sqrt{r^2 + eps}` where `eps` is a small number.

    Parameters
    ----------
    y : numpy.ndarray
        The values of the raw data.
    fit : numpy.ndarray
        The fit values.
    quantile : float
        The quantile value.
    eps : float, optional
        A small value added to the square of `residual` to prevent dividing by 0.
        Default is None, which uses `(1e-6 * max(abs(fit)))**2`.

    Returns
    -------
    numpy.ndarray
        The calculated loss, which can be used as weighting when performing iteratively
        reweighted least squares (IRLS)

    References
    ----------
    Schnabel, S., et al. Simultaneous estimation of quantile curves using quantile
    sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

    """
    if eps is None:
        # 1e-6 seems to work better than the 1e-4 in Schnabel, et al
        eps = (np.abs(fit).max() * 1e-6)**2
    residual = y - fit
    numerator = np.where(residual > 0, quantile, 1 - quantile)
    # use max(eps, _MIN_FLOAT) to ensure that eps + 0 > 0
    denominator = np.sqrt(residual**2 + max(eps, _MIN_FLOAT))  # approximates abs(residual)

    return numerator / denominator
