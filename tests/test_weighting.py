# -*- coding: utf-8 -*-
"""Tests for pybaselines._weighting."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import _weighting, utils, Baseline2D

from .conftest import get_data, get_data2d


def baseline_1d_normal():
    """
    Values for testing weights for a normally fit baseline.

    Approximates the baseline as a first order polynomial fit, which is the
    approximate fit for the first iteration of all Whittaker-smoothing-based
    algorithms with a `diff_order` of 2 and a sufficiently high `lam` value.

    """
    x_data, y_data = get_data()
    baseline = np.polynomial.Polynomial.fit(x_data, y_data, deg=1)(x_data)
    return y_data, baseline


def baseline_1d_all_above():
    """Values for testing weights when all baseline points are above the data."""
    x_data, y_data = get_data()
    baseline = np.full_like(y_data, y_data.max() + 10)
    return y_data, baseline


def baseline_1d_all_below():
    """Values for testing weights when all baseline points are below the data."""
    x_data, y_data = get_data()
    baseline = np.full_like(y_data, y_data.min() - 10)
    return y_data, baseline


def baseline_2d_normal():
    """
    Values for testing weights for a normally fit baseline.

    Approximates the baseline as a first order polynomial fit, which is the
    approximate fit for the first iteration of all Whittaker-smoothing-based
    algorithms with a `diff_order` of 2 and a sufficiently high `lam` value.

    """
    x_data, z_data, y_data = get_data2d()
    baseline = Baseline2D(x_data, z_data, check_finite=False, assume_sorted=True).poly(
        y_data, poly_order=1
    )[0]
    return y_data, baseline


def baseline_2d_all_above():
    """Values for testing weights when all baseline points are above the data."""
    x_data, z_data, y_data = get_data2d()
    baseline = np.full_like(y_data, y_data.max() + 10)
    return y_data, baseline


def baseline_2d_all_below():
    """Values for testing weights when all baseline points are below the data."""
    x_data, z_data, y_data = get_data2d()
    baseline = np.full_like(y_data, y_data.min() - 10)
    return y_data, baseline


def test_safe_std():
    """Checks that the calculated standard deviation is correct."""
    array = np.array((1, 2, 3))
    calc_std = _weighting._safe_std(array)

    assert_allclose(calc_std, np.std(array))


def test_safe_std_kwargs():
    """Checks that kwargs given to _safe_std are passed to numpy.std."""
    array = np.array((1, 2, 3))
    calc_std = _weighting._safe_std(array, ddof=1)

    assert_allclose(calc_std, np.std(array, ddof=1))


def test_safe_std_empty():
    """Checks that the returned standard deviation of an empty array is not nan."""
    calc_std = _weighting._safe_std(np.array(()))
    assert_allclose(calc_std, utils._MIN_FLOAT)


def test_safe_std_single():
    """Checks that the returned standard deviation of an array with a single value is not 0."""
    calc_std = _weighting._safe_std(np.array((1,)))
    assert_allclose(calc_std, utils._MIN_FLOAT)


def test_safe_std_zero():
    """Checks that the returned standard deviation is not 0."""
    calc_std = _weighting._safe_std(np.array((1, 1, 1)))
    assert_allclose(calc_std, utils._MIN_FLOAT)


# ignore the RuntimeWarning when using inf
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('run_enum', (0, 1))
def test_safe_std_allow_nan(run_enum):
    """
    Ensures that the standard deviation is allowed to be nan under certain conditions.

    _safe_std should allow the calculated standard deviation to be nan if there is
    more than one item in the array, since that would indicate that nan or inf is
    in the array and nan propogation would not want to be stopped in those cases.

    """
    if run_enum:
        array = np.array((1, 2, np.nan))
    else:
        array = np.array((1, 2, np.inf))

    assert np.isnan(_weighting._safe_std(array))


@pytest.mark.parametrize('quantile', np.linspace(0, 1, 21))
@pytest.mark.parametrize('one_d', (True, False))
def test_quantile_weighting(quantile, one_d):
    """Ensures the quantile weighting calculation is correct."""
    if one_d:
        y, fit = baseline_1d_normal()
    else:
        y, fit = baseline_2d_normal()

    residual = y - fit
    eps = 1e-10
    calc_loss = _weighting._quantile(y, fit, quantile, eps)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    denominator = np.sqrt(residual**2 + eps)

    expected_loss = numerator / denominator

    assert_allclose(calc_loss, expected_loss)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_normal(p, one_d):
    """Ensures asls weighting works as intented for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights = _weighting._asls(y_data, baseline, p)
    expected_weights = np.where(y_data > baseline, p, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_all_above(p, one_d):
    """Ensures asls weighting works as intented for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights = _weighting._asls(y_data, baseline, p)
    expected_weights = np.full_like(y_data, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_all_below(p, one_d):
    """Ensures asls weighting works as intented for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()
    weights = _weighting._asls(y_data, baseline, p)
    expected_weights = np.full_like(y_data, p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def expected_airpls(y, baseline, iteration):
    """
    The weighting for adaptive iteratively reweighted penalized least squares (airPLS).

    Does not perform error checking since this is just used for simple weighting cases.

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

    The pybaselines weighting differs from the original airPLS algorithm by dividing the
    weights by their maximum value to ensure that weights are within [0, 1]; the original
    algorithm allowed weights to be greater than 1 which does not make sense mathematically
    and does not match any other Whittaker-smoothing based algorithms.

    """
    residual = y - baseline
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    residual_l1_norm = abs(neg_residual).sum()

    new_weights = np.exp((-iteration / residual_l1_norm) * neg_residual)
    # Not stated in the paper, but without dividing by the maximum weight, the
    # calculated weights can be greater than 1 which does not make sense mathematically
    weights = np.zeros_like(y)
    weights[neg_mask] = new_weights / new_weights.max()

    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_airpls_normal(iteration, one_d):
    """Ensures airpls weighting works as intented for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, residual_l1_norm, exit_early = _weighting._airpls(y_data, baseline, iteration)
    expected_weights = expected_airpls(y_data, baseline, iteration)

    residual = y_data - baseline
    expected_residual_l1_norm = abs(residual[residual < 0].sum())

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_airpls_all_above(iteration, one_d):
    """Ensures airpls weighting works as intented for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, residual_l1_norm, exit_early = _weighting._airpls(y_data, baseline, iteration)
    expected_weights = expected_airpls(y_data, baseline, iteration)
    expected_residual_l1_norm = abs(y_data - baseline).sum()  # all residuals should be negative

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_airpls_all_below(iteration, one_d):
    """Ensures airpls weighting works as intented for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, residual_l1_norm, exit_early = _weighting._airpls(y_data, baseline, iteration)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, 0.0)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('dtype', (float, np.float16, np.float32))
def test_airpls_overflow(one_d, dtype):
    """Ensures exponential overflow does not occur from airpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    y_data = y_data.astype(dtype)
    baseline = baseline.astype(dtype)

    residual = y_data - baseline
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    expected_residual_l1_norm = abs(neg_residual).sum()

    # lowest iteration needed to cause exponential overflow for airpls weighting, plus 1
    iteration = (
        1
        + np.log(np.finfo(y_data.dtype).max) * expected_residual_l1_norm / abs(neg_residual).max()
    )

    # sanity check to ensure overflow actually should occur and that it produces nan weights
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_airpls(y_data, baseline, iteration)
    assert (~np.isfinite(expected_weights)).any()

    with np.errstate(over='raise'):
        weights, residual_l1_norm, exit_early = _weighting._airpls(y_data, baseline, iteration)

    assert np.isfinite(weights).all()
    assert not exit_early
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)


def expected_arpls(y, baseline):
    """
    The weighting for asymmetrically reweighted penalized least squares smoothing (arpls).

    Does not perform error checking since this is just used for simple weighting cases.

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
    std = _weighting._safe_std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    weights = 1 / (1 + np.exp((2 / std) * (residual - (2 * std - np.mean(neg_residual)))))
    return weights


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_normal(one_d):
    """Ensures arpls weighting works as intented for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._arpls(y_data, baseline)
    expected_weights = expected_arpls(y_data, baseline)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_all_above(one_d):
    """Ensures arpls weighting works as intented for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._arpls(y_data, baseline)
    expected_weights = expected_arpls(y_data, baseline)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_all_below(one_d):
    """Ensures arpls weighting works as intented for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._arpls(y_data, baseline)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_overflow(one_d):
    """Ensures exponential overflow does not occur from arpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    log_max = np.log(np.finfo(y_data.dtype).max)
    residual = y_data - baseline
    neg_residual = residual[residual < 0]
    std = np.std(neg_residual, ddof=1)
    mean = np.mean(neg_residual)
    # for exponential overlow, (residual + mean) / std > 0.5 * log_max + 2
    # changing one value in the baseline to cause overflow will not cause the mean and
    # standard deviation to change since the residual value will be positive at that index
    overflow_index = 10
    overflow_value = (0.5 * log_max + 2) * std - mean + 10  # add 10 for good measure
    if one_d:
        baseline[overflow_index] = overflow_value
    else:
        baseline[overflow_index, overflow_index] = overflow_value

    # sanity check to ensure overflow actually should occur
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_arpls(y_data, baseline)
    # the resulting weights should still be finite since 1 / (1 + inf) == 0
    assert np.isfinite(expected_weights).all()

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._arpls(y_data, baseline)

    assert np.isfinite(weights).all()
    assert not exit_early

    # the actual weight where overflow should have occurred should be 0
    if one_d:
        assert_allclose(weights[overflow_index], 0.0, atol=1e-14)
    else:
        assert_allclose(weights[overflow_index, overflow_index], 0.0, atol=1e-14)

    # weights should still be the same as the nieve calculation regardless of exponential overflow
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def expected_drpls(y, baseline, iteration):
    """
    The weighting for doubly reweighted penalized least squares (drpls).

    Does not perform error checking since this is just used for simple weighting cases.

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
    std = _weighting._safe_std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    inner = (np.exp(iteration) / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_normal(iteration, one_d):
    """Ensures drpls weighting works as intented for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._drpls(y_data, baseline, iteration)
    expected_weights = expected_drpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_all_above(iteration, one_d):
    """Ensures drpls weighting works as intented for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._drpls(y_data, baseline, iteration)
    expected_weights = expected_drpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_all_below(iteration, one_d):
    """Ensures drpls weighting works as intented for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._drpls(y_data, baseline, iteration)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_overflow(one_d):
    """Ensures exponential overflow does not occur from drpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    iteration = 1000
    # sanity check to ensure overflow actually should occur and that it produces nan weights
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_drpls(y_data, baseline, iteration)
    assert (~np.isfinite(expected_weights)).any()

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._drpls(y_data, baseline, iteration)

    assert np.isfinite(weights).all()
    assert not exit_early


def expected_iarpls(y, baseline, iteration):
    """
    The weighting for doubly reweighted penalized least squares (drpls).

    Does not perform error checking since this is just used for simple weighting cases.

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
    neg_residual = residual[residual < 0]

    std = np.std(neg_residual, ddof=1)  # use dof=1 since only sampling a subset
    # the exponential term is used to change the shape of the weighting from a logistic curve
    # at low iterations to a step curve at higher iterations (figure 1 in the paper); setting
    # the maximum iteration to 100 still acheives this purpose while avoiding unnecesarry
    # overflow for high iterations
    inner = (np.exp(iteration) / std) * (residual - 2 * std)
    weights = 0.5 * (1 - (inner / np.sqrt(1 + inner**2)))
    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_normal(iteration, one_d):
    """Ensures iarpls weighting works as intented for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._iarpls(y_data, baseline, iteration)
    expected_weights = expected_iarpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_all_above(iteration, one_d):
    """Ensures iarpls weighting works as intented for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._iarpls(y_data, baseline, iteration)
    expected_weights = expected_iarpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_all_below(iteration, one_d):
    """Ensures iarpls weighting works as intented for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._iarpls(y_data, baseline, iteration)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_overflow(one_d):
    """Ensures exponential overflow does not occur from iarpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    iteration = 1000
    # sanity check to ensure overflow actually should occur and that it produces nan weights
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_iarpls(y_data, baseline, iteration)
    assert (~np.isfinite(expected_weights)).any()

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._iarpls(y_data, baseline, iteration)

    assert np.isfinite(weights).all()
    assert not exit_early
