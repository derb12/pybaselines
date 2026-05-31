# -*- coding: utf-8 -*-
"""Contains various weighting schemes used in pybaselines."""

from functools import wraps
import warnings

import numpy as np
from scipy.special import erf, expit

from .utils import _MIN_FLOAT, ParameterWarning
from ._compat import _np_ge_2


def masked_weighting(weighting_func):
    """
    A decorator that adds mask support for weighting functions.

    Only valid indices, indicated with `False` values within `mask`, will be passed
    to the wrapped weighting function. Remaining indices in the output weights will
    have values set to 0.

    Parameters
    ----------
    weighting_func : Callable
        The weighting function to wrap. The function must not use a `mask`
        keyword argument since that is internally hooked into within this decorator.

    Returns
    -------
    Callable
        The wrapped function, now with masking support. Note that this wrapper makes
        the wrapped function keyword only after for the first argument.

    """
    @wraps(weighting_func)
    def inner(residual, *, mask=None, **kwargs):
        no_mask = mask is None
        if no_mask:
            input_residual = residual
        else:
            fit_mask = np.logical_not(mask)
            input_residual = residual[fit_mask]

        output = weighting_func(input_residual, **kwargs)
        if no_mask:
            full_output = output
        else:
            if isinstance(output, tuple):
                output_weights, *other = output
            else:
                output_weights = output
                other = ()
            full_weights = np.zeros(residual.shape)
            full_weights[fit_mask] = output_weights
            if other:
                full_output = (full_weights, *other)
            else:
                full_output = full_weights

        return full_output

    return inner


@masked_weighting
def _asls(residual, p):
    """
    The weighting for the asymmetric least squares algorithm (asls).

    Also used by the improved asymmetric least squares algorithm (iasls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Eilers, P., et al. Baseline correction with asymmetric least squares smoothing.
    Leiden University Medical Centre Report, 2005, [unpublished].

    Eilers, P. Parametric Time Warping. Analytical Chemistry, 2004, 76(2), 404-411.

    """
    weights = np.where(residual > 0, p, 1 - p)
    return weights


@masked_weighting
def _iasls(residual, p):
    """
    The weighting for the improved asymmetric least squares algorithm (iasls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    Notes
    -----
    Equivalent to ``_asls(y, baseline, p)**2``, but faster since the square is only
    applied to two scalars rather than the entire array.

    References
    ----------
    He, S., et al. Baseline correction for raman spectra using an improved
    asymmetric least squares method, Analytical Methods, 2014, 6(12), 4402-4407.

    """
    weights = np.where(residual > 0, p**2, (1 - p)**2)
    return weights


@masked_weighting
def _airpls(residual, iteration, normalize_weights=False):
    """
    The weighting for adaptive iteratively reweighted penalized least squares (airPLS).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.
    normalize_weights : bool, optional
        If True, will normalize the computed weights between 0 and 1 to improve
        the numerical stability. Set to False (default) to use the original implementation, which
        sets weights for all negative residuals to be greater than 1.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    residual_l1_norm : float
        The L1 norm of the negative residuals, used to calculate the exit criteria
        for the airPLS algorithm.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    Notes
    -----
    Equation 9 in the original algorithm was misprinted according to the author
    (https://github.com/zmzhang/airPLS/issues/8), so the correct weighting is used here.

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    """
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), 0.0, exit_early
    else:
        exit_early = False

    residual_l1_norm = neg_residual.sum()

    # the exponential of the iteration term is used to make weights more binary at higher
    # iterations (ie. the largest residuals control the weighting); setting the maximum
    # iteration to 50 still achieves this purpose while avoiding unnecessarily high
    # weights at high iterations which causes numerical instability
    # TODO a better way to address the high weighting would be to normalize the weights by
    # dividing by the max weight since the original airpls weighting sets all weights for
    # negative residuals to be 1 or higher while all other weighting schemes keep the weights
    # within the range [0, 1]; doing so would deviate from the paper, however, which takes
    # priority -> in reality, as long as a reasonable tolerance value is used, numerical
    # instability should never actually be an issue

    # clip from [0, log(max dtype)] since the positive residuals (negative values) do not matter
    log_max = np.log(np.finfo(residual.dtype).max)
    inner = np.clip(
        (min(iteration, 50) / residual_l1_norm) * neg_residual,
        a_min=0,
        a_max=log_max - np.spacing(log_max)
    )
    weights = np.zeros(residual.shape)
    weights[neg_mask] = np.exp(inner)
    if normalize_weights:
        weights[neg_mask] /= weights[neg_mask].max()

    return weights, abs(residual_l1_norm), exit_early


def _safe_std_mean(array, **kwargs):
    """
    Calculates the standard deviation and mean and protects against nan and 0.

    Used to prevent propagating nan or dividing by 0 when using the standard deviation.

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
    float
        The calculated mean of the array.

    Notes
    -----
    Does not protect against the calculated standard deviation of a non-empty
    array being nan because that would indicate that nan or inf was within the
    array, which should not be protected.

    Using the mean to compute the standard deviation with NumPy >= v2.0 reduces time by
    ~10% compared to making two separate calculations for the mean and standard deviation.

    """
    mean = np.mean(array, keepdims=True)  # must use keepdims=True if using to calc the std
    # std would be 0 for an array with size of 1 and inf if size <= ddof; only
    # internally use ddof=1, so the second condition is already covered
    if array.size < 2:
        std = _MIN_FLOAT
    else:
        if _np_ge_2():
            kwargs['mean'] = mean
        std = np.std(array, **kwargs)
        if std == 0:
            std = _MIN_FLOAT

    # since mean is computed with keepdims=True for use with np.std, need to get
    # the scalar value back out using flattening and indexing to work for 1D and 2D
    return std, mean.ravel()[0]


@masked_weighting
def _arpls(residual):
    """
    The weighting for asymmetrically reweighted penalized least squares smoothing (arpls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False
    std, mean = _safe_std_mean(neg_residual, ddof=1)  # use dof=1 since sampling subset
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(2 / std) * (residual - (2 * std - mean)))
    return weights, exit_early


@masked_weighting
def _drpls(residual, iteration):
    """
    The weighting for the doubly reweighted penalized least squares algorithm (drpls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False

    std, mean = _safe_std_mean(neg_residual, ddof=1)  # use dof=1 since sampling subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still achieves this purpose while avoiding unnecessary
    # overflow for high iterations
    inner = (np.exp(min(iteration, 100)) / std) * (residual - (2 * std - mean))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights, exit_early


@masked_weighting
def _iarpls(residual, iteration):
    """
    Weighting for improved asymmetrically reweighted penalized least squares smoothing (iarpls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
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
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False

    std = _safe_std_mean(neg_residual, ddof=1)[0]  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still achieves this purpose while avoiding unnecessary
    # overflow for high iterations
    inner = (np.exp(min(iteration, 100)) / std) * (residual - 2 * std)
    weights = 0.5 * (1 - (inner / np.sqrt(1 + inner**2)))
    return weights, exit_early


@masked_weighting
def _aspls(residual, asymmetric_coef=2., alternate_weighting=True):
    """
    Weighting for the adaptive smoothness penalized least squares smoothing (aspls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    asymmetric_coef : float, optional
        The asymmetric coefficient for the weighting. Higher values leads to a steeper
        weighting curve (ie. more step-like). Default is 2.
    alternate_weighting : bool, optional
        If True (default), subtracts the mean of the negative residuals within the weighting
        equation. If False, uses the weighting equation as stated within the aspls paper.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using adaptive smoothness
    parameter penalized least squares method. Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False
    std, mean = _safe_std_mean(neg_residual, ddof=1)  # use dof=1 since sampling subset
    offset = std - mean if alternate_weighting else std
    # add a negative sign since expit performs 1/(1+exp(-input))
    weights = expit(-(asymmetric_coef / std) * (residual - offset))
    return weights, exit_early


@masked_weighting
def _psalsa(residual, p, k):
    """
    Weighting for the peaked signal's asymmetric least squares algorithm (psalsa).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Positive residuals
        will be given ``p * exp(-(residual) / k)`` weight, and negative residuals
        will be given ``1 - p`` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Oller-Moreno, S., et al. Adaptive Asymmetric Least Squares baseline estimation
    for analytical instruments. 2014 IEEE 11th International Multi-Conference on
    Systems, Signals, and Devices, 2014, 1-5.

    """
    # only use positive residual in exp to avoid exponential overflow warnings
    # and accidentally creating a weight of nan (inf * 0 = nan)
    weights = np.full(residual.shape, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-residual[mask] / k)
    return weights


@masked_weighting
def _derpsalsa_inner(residual, p, k):
    """
    Weights for derivative peak-screening asymmetric least squares algorithm (derpsalsa).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Positive residuals
        will be given ``p * exp(-[(residual) / k)]**2 / 2)`` weight, and negative residuals
        will be given ``1 - p`` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Korepanov, V. Asymmetric least-squares baseline algorithm with peak screening for
    automatic processing of the Raman spectra. Journal of Raman Spectroscopy. 2020,
    51(10), 2061-2065.

    """
    # no need for caution since inner exponential is always negative, but still mask
    # since it's faster than performing the square and exp on the full residual
    weights = np.full(residual.shape, 1 - p, dtype=float)
    mask = residual > 0
    weights[mask] = p * np.exp(-0.5 * ((residual[mask] / k)**2))
    return weights


def _derpsalsa(residual, p, k, partial_weights, mask=None):
    """
    Weights for derivative peak-screening asymmetric least squares algorithm (derpsalsa).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Positive residuals
        will be given ``p * exp(-[(residual) / k)]**2 / 2)`` weight, and negative residuals
        will be given ``1 - p`` weight.
    k : float
        A factor that controls the exponential decay of the weights for baseline
        values greater than the data. Should be approximately the height at which
        a value could be considered a peak.
    partial_weights : numpy.ndarray, shape (N,) or (M, N)
        The weights associated with the first and second derivatives of the data.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
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
    weights = _derpsalsa_inner(residual, p=p, k=k, mask=mask)
    weights *= partial_weights
    return weights


@masked_weighting
def _quantile(residual, quantile, eps=None):
    r"""
    An approximation of quantile loss.

    The loss is defined as :math:`\rho(r) / |r|`, where r is the residual, `data - baseline`,
    and the function :math:`\rho(r)` is `quantile` for `r` > 0 and 1 - `quantile`
    for `r` < 0. Rather than using `|r|` as the denominator, which is non-differentiable
    and causes issues when `r` = 0, the denominator is approximated as
    :math:`\sqrt{r^2 + eps}` where `eps` is a small number.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    quantile : float
        The quantile value.
    eps : float, optional
        A small value added to the square of `residual` to prevent dividing by 0.
        Default is None, which uses `(1e-4 * max(abs(residual)))**2`.

    Returns
    -------
    numpy.ndarray, shape (N,) or (M, N)
        The calculated loss, which can be used as weighting when performing iteratively
        reweighted least squares (IRLS)

    Notes
    -----
    The denominator of the weights approximates the absolute loss from the least squares
    result following [1]_, while the numerator gives the quantile weighting following [2]_.
    Note that the same weighting is also used for `Baseline.irsqr` from [3]_ even though they
    recommend using `|r|` in the denominator, for the non-differentiable reason listed in the
    top of this function.

    References
    ----------
    .. [1] Schlossmacher, E. An iterative technique for absolute deviations curve fitting.
        Journal of the American Statistical Association, 1973, 68, 857-859.
    .. [2] Schnabel, S., et al. Simultaneous estimation of quantile curves using quantile
        sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.
    .. [3] Han, Q., et al. Iterative Reweighted Quantile Regression Using Augmented Lagrangian
        Optimization for Baseline Correction. 2018 5th International Conference on Information
        Science and Control Engineering (ICISCE), 2018, 280-284.

    """
    if eps is None:
        eps = (abs(residual).max() * 1e-4)**2
    numerator = np.where(residual > 0, quantile, 1 - quantile)
    # use max(eps, _MIN_FLOAT) to ensure that eps + 0 > 0
    denominator = np.sqrt(residual**2 + max(eps, _MIN_FLOAT))  # approximates abs(residual)

    return numerator / denominator


@masked_weighting
def _brpls(residual, beta):
    """
    The weighting for Bayesian Reweighted Penalized Least Squares (BrPLS).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    beta : float
        A value between 0 and 1 designating the probability of signal within the data.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
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
    # exclude residual == 0 to ensure mean and sigma are both nonzero since both
    # are used within the denominator
    neg_residual = residual[residual < 0].ravel()  # ravel so x @ x == sum(x**2) for 2D too
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
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False

    # note: both mean and sigma are calculated following expectation-maximization for exponential
    # and gaussian distributions, respectively
    mean = np.mean(pos_residual)
    # sigma is the quadratic mean, ie. the root mean square
    sigma = np.sqrt((neg_residual @ neg_residual) / neg_residual.size)

    inner = (residual / (sigma * np.sqrt(2))) - (sigma / (mean * np.sqrt(2)))
    multiplier = ((beta * np.sqrt(0.5 * np.pi)) / max(1 - beta, _MIN_FLOAT)) * (sigma / mean)
    # overflow occurs at 2 * multiplier * exp(max_val**2), where the 2 is from 1 + max(erf(x));
    # clip to ignore overflow warning since 1 / (1 + inf) == 0, which is fine, but can
    # also cause nan if erf(x) = -1 and exp(x**2) = inf since 0 * inf = nan
    max_val = np.sqrt(np.log(np.finfo(residual.dtype).max))
    max_val -= np.spacing(max_val)  # ensure limit is below max value

    partial = np.exp(np.clip(inner, -max_val, max_val)**2)
    if multiplier < 0.5:  # no need to worry about multiplication overflow
        weights = 1 / (1 + multiplier * (1 + erf(inner)) * partial)
    else:
        max_val_mult = np.finfo(residual.dtype).max / (2 * multiplier)
        max_val_mult -= np.spacing(max_val_mult)  # ensure limit is below max value

        weights = 1 / (1 + multiplier * (1 + erf(inner)) * np.clip(partial, None, max_val_mult))
    return weights, exit_early


@masked_weighting
def _lsrpls(residual, iteration, alternate_weighting=False):
    """
    The weighting for the locally symmetric reweighted penalized least squares (lsrpls).

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,) or (M, N)
        The current residual, ``data - baseline``.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.
    alternate_weighting : bool, optional
        If False (default), the weighting uses a prefactor term of ``10^t``, where ``t`` is
        the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
        a prefactor term of ``exp(t)``. See the Notes section below for more details.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    exit_early : bool
        Designates if there is a potential error with the calculation such that no further
        iterations should be performed.

    Notes
    -----
    In the LSRPLS paper [1]_, the weighting equation is written with a prefactor term
    of ``10^t``, where ``t`` is the iteration number, but the plotted weighting curve in
    Figure 1 of the paper shows a prefactor term of ``exp(t)`` instead. Since it is ambiguous
    which prefactor term is actually used for the algorithm, both are permitted by setting
    `alternate_weighting` to True to use ``10^t`` and False to use ``exp(t)``. In practice,
    the prefactor determines how quickly the weighting curve converts from a sigmoidal curve
    to a step curve, and does not heavily influence the result.

    If ``alternate_weighting`` is False, the weighting is the same as the drPLS algorithm [2]_.

    References
    ----------
    .. [1] Heng, Z., et al. Baseline correction for Raman Spectra Based on Locally Symmetric
        Reweighted Penalized Least Squares. Chinese Journal of Lasers, 2018, 45(12), 1211001.
    .. [2] Xu, D. et al. Baseline correction method based on doubly reweighted
        penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    if alternate_weighting:
        return _drpls(residual, iteration=iteration)

    neg_residual = residual[residual < 0]
    if neg_residual.size < 2:
        exit_early = True
        warnings.warn(
            ('almost all baseline points are below the data, indicating that "tol"'
             ' is too low and/or "max_iter" is too high'), ParameterWarning,
            stacklevel=3
        )
        return np.zeros(residual.shape), exit_early
    else:
        exit_early = False

    std, mean = _safe_std_mean(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still achieves this purpose while avoiding unnecessary
    # overflow for high iterations
    inner = (10**(min(iteration, 100)) / std) * (residual - (2 * std - mean))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights, exit_early
