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
def test_quantile_weighting(quantile):
    """Ensures the quantile weighting calculation is correct."""
    y = np.linspace(-1, 1)
    fit = np.zeros(y.shape[0])
    residual = y - fit
    eps = 1e-10
    calc_loss = _weighting._quantile(y, fit, quantile, eps)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    denominator = np.sqrt(residual**2 + eps)

    expected_loss = numerator / denominator

    assert_allclose(calc_loss, expected_loss)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('two_d', (True, False))
def test_asls_normal(p, two_d):
    """Ensures asls weighting works as intented for a normal baseline."""
    if two_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights = _weighting._asls(y_data, baseline, p)
    expected_weights = np.where(y_data > baseline, p, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('two_d', (True, False))
def test_asls_all_above(p, two_d):
    """Ensures asls weighting works as intented for a baseline with all points above the data."""
    if two_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights = _weighting._asls(y_data, baseline, p)
    expected_weights = np.full_like(y_data, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('two_d', (True, False))
def test_asls_all_below(p, two_d):
    """Ensures asls weighting works as intented for a baseline with all points below the data."""
    if two_d:
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
@pytest.mark.parametrize('two_d', (True, False))
def test_airpls_normal(iteration, two_d):
    """Ensures airpls weighting works as intented for a normal baseline."""
    if two_d:
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
@pytest.mark.parametrize('two_d', (True, False))
def test_airpls_all_above(iteration, two_d):
    """Ensures airpls weighting works as intented for a baseline with all points above the data."""
    if two_d:
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
@pytest.mark.parametrize('two_d', (True, False))
def test_airpls_all_below(iteration, two_d):
    """Ensures airpls weighting works as intented for a baseline with all points below the data."""
    if two_d:
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


@pytest.mark.parametrize('two_d', (True, False))
@pytest.mark.parametrize('dtype', (float, np.float16, np.float32))
def test_airpls_overflow(two_d, dtype):
    """Ensures exponential overflow does not occur from airpls weighting."""
    if two_d:
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

