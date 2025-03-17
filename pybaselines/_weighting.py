# -*- coding: utf-8 -*-
"""Contains various weighting schemes used in pybaselines."""

import warnings

import numpy as np
from scipy.special import erf, expit

from .utils import _MIN_FLOAT, ParameterWarning


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
        will be given `1 - p` weight.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.

    References
    ----------
    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, [unpublished].

    Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    """
    weights = np.where(y > baseline, p, 1 - p)
    return weights


def _airpls(y, baseline, iteration, normalize_weights):
    """
    The weighting for adaptive iteratively reweighted penalized least squares (airPLS).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.
    normalize_weights : bool
        If True, will normalize the computed weights between 0 and 1 to improve
        the numerical stabilty. Set to False to use the original implementation, which
        sets weights for all negative residuals to be greater than 1.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.
    residual_l1_norm : float
        The L1 norm of the negative residuals, used to calculate the exit criteria
        for the airPLS algorithm.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    Notes
    -----
    Equation 9 in the original algorithm was misprinted according to the author
    (https://github.com/zmzhang/airPLS/issues/8), so the correct weighting is used here.

    """
    residual = y - baseline
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), 0.0, exit_early
    else:
        exit_early = False

    residual_l1_norm = neg_residual.sum()

    # the exponential of the iteration term is used to make weights more binary at higher
    # iterations (ie. the largest residuals control the weighting); setting the maximum
    # iteration to 50 still acheives this purpose while avoiding unnecessarily high
    # weights at high iterations which causes numerical instability
    # TODO a better way to address the high weighting would be to normalize the weights by
    # dividing by the max weight since the original airpls weighting sets all weights for
    # negative residuals to be 1 or higher while all other weighting schemes keep the weights
    # within the range [0, 1]; doing so would deviate from the paper, however, which takes
    # priority -> in reality, as long as a reasonable tolerance value is used, numerical
    # instability should never actually be an issue

    # clip from [0, log(max dtype)] since the positive residuals (negative values) do not matter
    log_max = np.log(np.finfo(y.dtype).max)
    inner = np.clip(
        (min(iteration, 50) / residual_l1_norm) * neg_residual,
        a_min=0,
        a_max=log_max - np.spacing(log_max)
    )
    weights = np.zeros_like(y)
    weights[neg_mask] = np.exp(inner)
    if normalize_weights:
        weights[neg_mask] /= weights[neg_mask].max()

    return weights, abs(residual_l1_norm), exit_early


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
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), exit_early
    else:
        exit_early = False
    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(2 / std) * (residual - (2 * std - np.mean(neg_residual))))
    return weights, exit_early


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
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), exit_early
    else:
        exit_early = False

    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still acheives this purpose while avoiding unnecesarry
    # overflow for high iterations
    inner = (np.exp(min(iteration, 100)) / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights, exit_early


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
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Ye, J., et al. Baseline correction method based on improved asymmetrically
    reweighted penalized least squares for Raman spectrum. Applied Optics, 2020,
    59, 10933-10943.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), exit_early
    else:
        exit_early = False

    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still acheives this purpose while avoiding unnecesarry
    # overflow for high iterations
    inner = (np.exp(min(iteration, 100)) / std) * (residual - 2 * std)
    weights = 0.5 * (1 - (inner / np.sqrt(1 + inner**2)))
    return weights, exit_early


def _aspls(y, baseline, asymmetric_coef):
    """
    Weighting for the adaptive smoothness penalized least squares smoothing (aspls).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    asymmetric_coef : float
        The asymmetric coefficient for the weighting. Higher values leads to a steeper
        weighting curve (ie. more step-like).

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.
    residual : numpy.ndarray, shape (N,)
        The residual, ``y - baseline``.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    Notes
    -----
    The default asymmetric coefficient (`k` in the asPLS paper) is 0.5 instead
    of the 2 listed in the asPLS paper. pybaselines uses the factor of 0.5 since it
    matches the results in Table 2 and Figure 5 of the asPLS paper closer than the
    factor of 2 and fits noisy data much better.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using adaptive smoothness
    parameter penalized least squares method. Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), residual, exit_early
    else:
        exit_early = False
    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since sampling subset

    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(asymmetric_coef / std) * (residual - std))
    return weights, residual, exit_early


def _psalsa(y, baseline, p, k, shape_y):
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
        will be given `1 - p` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.
    shape_y : int or (int,) or (int, int)
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
    weights = np.full(shape_y, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-residual[mask] / k)
    return weights


def _derpsalsa(y, baseline, p, k, shape_y, partial_weights):
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
        will be given `1 - p` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.
    shape_y : int or (int,) or (int, int)
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
    weights = np.full(shape_y, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-0.5 * ((residual[mask] / k)**2))
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


def _brpls(y, baseline, beta):
    """
    The weighting for Bayesian Reweighted Penalized Least Squares (BrPLS).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The measured data.
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    beta : float
        A value between 0 and 1 designating the probability of signal within the data.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The calculated weights.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Wang, Q., et al. Spectral baseline estimation using penalized least squares
    with weights derived from the Bayesian method. Nuclear Science and Techniques,
    2022, 140, 250-257.

    """
    residual = y - baseline
    # exclude residual == 0 to ensure mean and sigma are both nonzero since both
    # are used within the demoninator
    neg_residual = residual[residual < 0].ravel()  # ravel so x.dot(x) == sum(x**2) for 2D too
    pos_residual = residual[residual > 0]
    if neg_residual.size < 2 or pos_residual.size < 2:
        exit_early = True
        if neg_residual.size < 2:
            position = 'below'
        else:
            position = 'above'
        warnings.warn(
            (f'almost all baseline points are {position} the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), exit_early
    else:
        exit_early = False

    # note: both mean and sigma are calculated following expectation-maximization for exponential
    # and gaussian distributions, respectively
    mean = np.mean(pos_residual)
    # sigma is the quadratic mean, ie. the root mean square
    sigma = np.sqrt(neg_residual.dot(neg_residual) / neg_residual.size)

    inner = (residual / (sigma * np.sqrt(2))) - (sigma / (mean * np.sqrt(2)))
    multiplier = ((beta * np.sqrt(0.5 * np.pi)) / max(1 - beta, _MIN_FLOAT)) * (sigma / mean)
    # overflow occurs at 2 * multiplier * exp(max_val**2), where the 2 is from 1 + max(erf(x));
    # clip to ignore overflow warning since 1 / (1 + inf) == 0, which is fine, but can
    # also cause nan if erf(x) = -1 and exp(x**2) = inf since 0 * inf = nan
    max_val = np.sqrt(np.log(np.finfo(y.dtype).max))
    max_val -= np.spacing(max_val)  # ensure limit is below max value

    partial = np.exp(np.clip(inner, -max_val, max_val)**2)
    if multiplier < 0.5:  # no need to worry about multiplication overflow
        weights = 1 / (1 + multiplier * (1 + erf(inner)) * partial)
    else:
        max_val_mult = np.finfo(y.dtype).max / (2 * multiplier)
        max_val_mult -= np.spacing(max_val_mult)  # ensure limit is below max value

        weights = 1 / (1 + multiplier * (1 + erf(inner)) * np.clip(partial, None, max_val_mult))
    return weights, exit_early


def _lsrpls(y, baseline, iteration):
    """
    The weighting for the locally symmetric reweighted penalized least squares (lsrpls).

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
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
    Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
             stacklevel=2
        )
        return np.zeros_like(y), exit_early
    else:
        exit_early = False

    std = _safe_std(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still acheives this purpose while avoiding unnecesarry
    # overflow for high iterations
    inner = (10**(min(iteration, 100)) / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights, exit_early
