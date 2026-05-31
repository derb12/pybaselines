# -*- coding: utf-8 -*-
"""Tests for pybaselines._weighting."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.special import erf

from pybaselines import _weighting, utils

from .base_tests import get_data, get_data2d, _Poly2D


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
    baseline = _Poly2D(x_data, z_data).poly(y_data, poly_order=1)[0]
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


def data_and_mask(one_d):
    """Generates data and mask for testing."""
    if one_d:
        y, baseline = baseline_1d_normal()
    else:
        y, baseline = baseline_2d_normal()

    return y, baseline, np.random.default_rng(0).choice([True, False], y.shape, p=(0.3, 0.7))


def test_safe_std_mean():
    """Checks that the calculated mean and standard deviation is correct."""
    array = np.arange(60, dtype=float)
    expected_std = np.std(array)
    expected_mean = np.mean(array)

    calc_std, calc_mean = _weighting._safe_std_mean(array)

    assert_allclose(calc_std, expected_std, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, expected_mean, rtol=1e-12, atol=1e-12)

    # also ensure it works for 2D and 3D inputs; mean and std should remain the same
    calc_std, calc_mean = _weighting._safe_std_mean(array.reshape(10, -1))

    assert_allclose(calc_std, expected_std, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, expected_mean, rtol=1e-12, atol=1e-12)

    calc_std, calc_mean = _weighting._safe_std_mean(array.reshape(5, 3, -1))

    assert_allclose(calc_std, expected_std, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, expected_mean, rtol=1e-12, atol=1e-12)


def test_safe_std_mean_kwargs():
    """Checks that kwargs given to _safe_std_mean are passed to numpy.std."""
    array = np.array((1, 2, 3))
    expected_std = np.std(array, ddof=1)
    expected_mean = np.mean(array)

    calc_std, calc_mean = _weighting._safe_std_mean(array, ddof=1)

    assert_allclose(calc_std, expected_std, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, expected_mean, rtol=1e-12, atol=1e-12)


def test_safe_std_mean_empty():
    """Checks that the returned standard deviation of an empty array is not nan.

    # TODO is this needed any longer? It should not be called if the array is empty.
    """
    with pytest.warns(RuntimeWarning):
        calc_std, calc_mean = _weighting._safe_std_mean(np.array(()))
    assert_allclose(calc_std, utils._MIN_FLOAT, rtol=1e-12, atol=1e-12)
    assert np.isnan(calc_mean)


def test_safe_std_mean_single():
    """Checks that the returned standard deviation of an array with a single value is not 0."""
    array = np.array([1])

    calc_std, calc_mean = _weighting._safe_std_mean(array)

    assert_allclose(calc_std, utils._MIN_FLOAT, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, np.mean(array), rtol=1e-12, atol=1e-12)


def test_safe_std_mean_zero():
    """Checks that the returned standard deviation is not 0."""
    array = np.array((1, 1, 1))

    calc_std, calc_mean = _weighting._safe_std_mean(array)

    assert_allclose(calc_std, utils._MIN_FLOAT, rtol=1e-12, atol=1e-12)
    assert_allclose(calc_mean, np.mean(array), rtol=1e-12, atol=1e-12)


# ignore the RuntimeWarning when using inf
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('run_enum', (0, 1))
def test_safe_std_mean_allow_nan(run_enum):
    """
    Ensures that the standard deviation is allowed to be nan under certain conditions.

    _safe_std should allow the calculated standard deviation to be nan if there is
    more than one item in the array, since that would indicate that nan or inf is
    in the array and nan propagation would not want to be stopped in those cases.

    """
    if run_enum:
        array = np.array((1, 2, np.nan))
    else:
        array = np.array((1, 2, np.inf))

    calc_std, calc_mean = _weighting._safe_std_mean(array)

    assert_allclose(calc_mean, np.mean(array), rtol=1e-12, atol=1e-12)
    assert np.isnan(calc_std)


@pytest.mark.parametrize('eps', (1e-10, None))
@pytest.mark.parametrize('quantile', np.linspace(0, 1, 5))
@pytest.mark.parametrize('one_d', (True, False))
def test_quantile_weighting(quantile, one_d, eps):
    """Ensures the quantile weighting calculation is correct."""
    if one_d:
        y, fit = baseline_1d_normal()
    else:
        y, fit = baseline_2d_normal()

    residual = y - fit
    calc_loss = _weighting._quantile(y - fit, quantile=quantile, eps=eps)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    expected_eps = (abs(residual).max() * 1e-4)**2 if eps is None else eps
    denominator = np.sqrt(residual**2 + expected_eps)

    expected_loss = numerator / denominator

    assert_allclose(calc_loss, expected_loss, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('eps', (1e-10, None))
@pytest.mark.parametrize('quantile', np.linspace(0, 1, 5))
@pytest.mark.parametrize('one_d', (True, False))
def test_quantile_masked(quantile, one_d, eps):
    """Ensures quantile weighting works with masks."""
    y_data, baseline, mask = data_and_mask(one_d)
    mask_inv = np.logical_not(mask)

    residual = y_data - baseline
    eps = 1e-10
    calc_loss = _weighting._quantile(residual, quantile=quantile, eps=eps, mask=mask)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    expected_eps = (abs(residual[mask_inv]).max() * 1e-4)**2 if eps is None else eps
    denominator = np.sqrt(residual**2 + expected_eps)

    expected_loss = numerator / denominator
    expected_loss[mask] = 0

    assert_allclose(calc_loss, expected_loss, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_normal(p, one_d):
    """Ensures asls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights = _weighting._asls(y_data - baseline, p=p)
    expected_weights = np.where(y_data > baseline, p, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_all_above(p, one_d):
    """Ensures asls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights = _weighting._asls(y_data - baseline, p=p)
    expected_weights = np.full_like(y_data, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_asls_all_below(p, one_d):
    """Ensures asls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()
    weights = _weighting._asls(y_data - baseline, p=p)
    expected_weights = np.full_like(y_data, p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
def test_asls_masked(one_d):
    """Ensures asls weighting works with masks."""
    y_data, baseline, mask = data_and_mask(one_d)
    p = 0.1

    weights = _weighting._asls(y_data - baseline, p=p, mask=mask)
    expected_weights = np.where(y_data > baseline, p, 1 - p)
    expected_weights[mask] = 0

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_iasls_normal(p, one_d):
    """Ensures iasls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights = _weighting._iasls(y_data - baseline, p=p)
    expected_weights = np.where(y_data > baseline, p**2, (1 - p)**2)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert ((weights >= 0) & (weights <= 1)).all()

    # also test against _asls, which should just be the sqrt of _iasls
    expected_weights_2 = _weighting._asls(y_data - baseline, p=p)**2
    assert_allclose(weights, expected_weights_2, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_iasls_all_above(p, one_d):
    """Ensures iasls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights = _weighting._iasls(y_data - baseline, p=p)
    expected_weights = np.full_like(y_data, (1 - p)**2)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('p', (0.01, 0.99))
@pytest.mark.parametrize('one_d', (True, False))
def test_iasls_all_below(p, one_d):
    """Ensures iasls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()
    weights = _weighting._iasls(y_data - baseline, p=p)
    expected_weights = np.full_like(y_data, p**2)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
def test_iasls_masked(one_d):
    """Ensures iasls weighting works with masks."""
    y_data, baseline, mask = data_and_mask(one_d)
    p = 0.1

    weights = _weighting._iasls(y_data - baseline, p=p, mask=mask)
    expected_weights = np.where(y_data > baseline, p**2, (1 - p)**2)
    expected_weights[mask] = 0

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def expected_airpls(y, baseline, iteration, normalize_weights):
    """
    The weighting for adaptive iteratively reweighted penalized least squares (airPLS).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.
    normalize_weights : bool
        If True, will normalize the computed weights between 0 and 1 to improve
        the numerical stability. Set to False to use the original implementation, which
        sets weights for all negative residuals to be greater than 1.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    Notes
    -----
    Equation 9 in the original algorithm was misprinted according to the author
    (https://github.com/zmzhang/airPLS/issues/8), so the correct weighting is used here.

    References
    ----------
    Zhang, Z.M., et al. Baseline correction using adaptive iteratively
    reweighted penalized least squares. Analyst, 2010, 135(5), 1138-1146.

    """
    residual = y - baseline
    neg_mask = residual < 0
    neg_residual = residual[neg_mask]
    residual_l1_norm = abs(neg_residual).sum()

    weights = np.zeros_like(y)
    weights[neg_mask] = np.exp((-iteration / residual_l1_norm) * neg_residual)
    if normalize_weights:
        weights /= weights.max()

    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('normalize', (True, False))
def test_airpls_normal(iteration, one_d, normalize):
    """Ensures airpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, residual_l1_norm, exit_early = _weighting._airpls(
        y_data - baseline, iteration=iteration, normalize_weights=normalize
    )
    expected_weights = expected_airpls(y_data, baseline, iteration, normalize)

    residual = y_data - baseline
    expected_residual_l1_norm = abs(residual[residual < 0].sum())

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)
    assert not exit_early

    # airpls differs from other weighting schemes in that all negative residuals
    # should have a weight of 1 or more rather than being in the range [0, 1]
    assert (weights >= 0).all()
    if normalize:
        assert (weights <= 1).all()
    else:
        assert (weights[residual < 0] >= 1).all()


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('normalize', (True, False))
def test_airpls_all_above(iteration, one_d, normalize):
    """Ensures airpls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, residual_l1_norm, exit_early = _weighting._airpls(
        y_data - baseline, iteration=iteration, normalize_weights=normalize
    )
    expected_weights = expected_airpls(y_data, baseline, iteration, normalize)
    expected_residual_l1_norm = abs(y_data - baseline).sum()  # all residuals should be negative

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('normalize', (True, False))
def test_airpls_all_below(iteration, one_d, normalize):
    """Ensures airpls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, residual_l1_norm, exit_early = _weighting._airpls(
            y_data - baseline, iteration=iteration, normalize_weights=normalize
        )
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, 0.0)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('dtype', (float, np.float16, np.float32))
@pytest.mark.parametrize('normalize', (True, False))
def test_airpls_overflow(one_d, dtype, normalize):
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
        expected_weights = expected_airpls(y_data, baseline, iteration, normalize)
    assert (~np.isfinite(expected_weights)).any()

    with np.errstate(over='raise'):
        weights, residual_l1_norm, exit_early = _weighting._airpls(
            y_data - baseline, iteration=iteration, normalize_weights=normalize
        )

    assert np.isfinite(weights).all()
    assert not exit_early
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
def test_airpls_masked(one_d):
    """Ensures iasls weighting works with masks."""
    y_data, baseline, mask = data_and_mask(one_d)
    iteration = 5
    mask_inv = np.logical_not(mask)

    weights, residual_l1_norm, exit_early = _weighting._airpls(
        y_data - baseline, iteration=iteration, mask=mask
    )
    partial_weights = expected_airpls(y_data[mask_inv], baseline[mask_inv], iteration, False)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    residual = (y_data - baseline)[mask_inv]
    expected_residual_l1_norm = abs(residual[residual < 0].sum())

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(residual_l1_norm, expected_residual_l1_norm, rtol=1e-12, atol=1e-12)
    assert not exit_early


def expected_arpls(y, baseline):
    """
    The weighting for asymmetrically reweighted penalized least squares smoothing (arpls).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Baek, S.J., et al. Baseline correction using asymmetrically reweighted
    penalized least squares smoothing. Analyst, 2015, 140, 250-257.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    std = np.std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    weights = 1 / (1 + np.exp((2 / std) * (residual - (2 * std - np.mean(neg_residual)))))
    return weights


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_normal(one_d):
    """Ensures arpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._arpls(y_data - baseline)
    expected_weights = expected_arpls(y_data, baseline)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_all_above(one_d):
    """Ensures arpls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._arpls(y_data - baseline)
    expected_weights = expected_arpls(y_data, baseline)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_all_below(one_d):
    """Ensures arpls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._arpls(y_data - baseline)
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
    # for exponential overflow, (residual + mean) / std > 0.5 * log_max + 2
    # changing one value in the baseline to cause overflow will not cause the mean and
    # standard deviation to change since the residual value will be positive at that index
    overflow_index = 10
    overflow_value = (0.5 * log_max + 2) * std - mean + 10  # add 10 for good measure
    if one_d:
        baseline[overflow_index] = y_data[overflow_index] - overflow_value
    else:
        baseline[overflow_index, overflow_index] = (
            y_data[overflow_index, overflow_index] - overflow_value
        )

    # sanity check to ensure overflow actually should occur
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_arpls(y_data, baseline)
    # the resulting weights should still be finite since 1 / (1 + inf) == 0
    assert np.isfinite(expected_weights).all()

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._arpls(y_data - baseline)

    assert np.isfinite(weights).all()
    assert not exit_early

    # the actual weight where overflow should have occurred should be 0
    if one_d:
        assert_allclose(weights[overflow_index], 0.0, atol=1e-14)
    else:
        assert_allclose(weights[overflow_index, overflow_index], 0.0, atol=1e-14)

    # weights should still be the same as the naive calculation regardless of exponential overflow
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
def test_arpls_masked(one_d):
    """Ensures arpls weighting works with masks."""
    y_data, baseline, mask = data_and_mask(one_d)
    mask_inv = np.logical_not(mask)

    weights, exit_early = _weighting._arpls(y_data - baseline, mask=mask)
    partial_weights = expected_arpls(y_data[mask_inv], baseline[mask_inv])
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


def expected_drpls(y, baseline, iteration):
    """
    The weighting for doubly reweighted penalized least squares (drpls).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Xu, D. et al. Baseline correction method based on doubly reweighted
    penalized least squares, Applied Optics, 2019, 58, 3913-3920.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    std = np.std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    inner = (np.exp(iteration) / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_normal(iteration, one_d):
    """Ensures drpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._drpls(y_data - baseline, iteration=iteration)
    expected_weights = expected_drpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_all_above(iteration, one_d):
    """Ensures drpls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._drpls(y_data - baseline, iteration=iteration)
    expected_weights = expected_drpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_all_below(iteration, one_d):
    """Ensures drpls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._drpls(y_data - baseline, iteration=iteration)
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
        weights, exit_early = _weighting._drpls(y_data - baseline, iteration=iteration)

    assert np.isfinite(weights).all()
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_drpls_masked(one_d):
    """Ensures drpls weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    iteration = 5
    mask_inv = np.logical_not(mask)

    weights, exit_early = _weighting._drpls(y_data - baseline, iteration=iteration, mask=mask)
    partial_weights = expected_drpls(y_data[mask_inv], baseline[mask_inv], iteration)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


def expected_iarpls(y, baseline, iteration):
    """
    The weighting for doubly reweighted penalized least squares (drpls).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
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
    # the maximum iteration to 100 still achieves this purpose while avoiding unnecessary
    # overflow for high iterations
    inner = (np.exp(iteration) / std) * (residual - 2 * std)
    weights = 0.5 * (1 - (inner / np.sqrt(1 + inner**2)))
    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_normal(iteration, one_d):
    """Ensures iarpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._iarpls(y_data - baseline, iteration=iteration)
    expected_weights = expected_iarpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_all_above(iteration, one_d):
    """Ensures iarpls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._iarpls(y_data - baseline, iteration=iteration)
    expected_weights = expected_iarpls(y_data, baseline, iteration)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_all_below(iteration, one_d):
    """Ensures iarpls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._iarpls(y_data - baseline, iteration=iteration)
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
        weights, exit_early = _weighting._iarpls(y_data - baseline, iteration=iteration)

    assert np.isfinite(weights).all()
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_iarpls_masked(one_d):
    """Ensures iarpls weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    iteration = 5
    mask_inv = np.logical_not(mask)

    weights, exit_early = _weighting._iarpls(y_data - baseline, iteration=iteration, mask=mask)
    partial_weights = expected_iarpls(y_data[mask_inv], baseline[mask_inv], iteration)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


def expected_aspls(y, baseline, asymmetric_coef, alternate_weighting):
    """
    The weighting for adaptive smoothness penalized least squares smoothing (aspls).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    asymmetric_coef : float
        The asymmetric coefficient for the weighting. Higher values leads to a steeper
        weighting curve (ie. more step-like).
    alternate_weighting : bool
        If True (default), subtracts the mean of the negative residuals within the weighting
        equation. If False, uses the weighting equation as stated within the aspls paper.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.
    alpha_array : numpy.ndarray, shape (N,) or (M, N)
        The updated alpha values.

    References
    ----------
    Zhang, F., et al. Baseline correction for infrared spectra using adaptive smoothness
    parameter penalized least squares method. Spectroscopy Letters, 2020, 53(3), 222-233.

    """
    residual = y - baseline
    neg_residual = residual[residual < 0]
    std = np.std(neg_residual, ddof=1)  # use dof=1 since sampling subset

    if alternate_weighting:
        shifted_residual = residual + neg_residual.mean()
    else:
        shifted_residual = residual
    weights = 1 / (1 + np.exp(asymmetric_coef * (shifted_residual - std) / std))

    abs_d = np.abs(residual)
    alpha_array = abs_d / abs_d.max()

    return weights, alpha_array


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_aspls_normal(one_d, asymmetric_coef, alternate_weighting):
    """Ensures aspls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early, alpha = _weighting._aspls(
        y_data - baseline, asymmetric_coef=asymmetric_coef, alternate_weighting=alternate_weighting
    )
    expected_weights, expected_alpha = expected_aspls(
        y_data, baseline, asymmetric_coef, alternate_weighting
    )

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == y_data.shape
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)
    assert ((alpha >= 0) & (alpha <= 1)).all()


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_aspls_all_above(one_d, asymmetric_coef, alternate_weighting):
    """Ensures aspls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()

    weights, exit_early, alpha = _weighting._aspls(
        y_data - baseline, asymmetric_coef=asymmetric_coef, alternate_weighting=alternate_weighting
    )
    expected_weights, expected_alpha = expected_aspls(
        y_data, baseline, asymmetric_coef, alternate_weighting
    )

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == y_data.shape
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)
    assert ((alpha >= 0) & (alpha <= 1)).all()


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_aspls_all_below(one_d, asymmetric_coef, alternate_weighting):
    """Ensures aspls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early, alpha = _weighting._aspls(
            y_data - baseline, asymmetric_coef=asymmetric_coef,
            alternate_weighting=alternate_weighting
        )
    expected_weights = np.zeros_like(y_data)
    expected_alpha = np.ones_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == y_data.shape
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_aspls_overflow(one_d, asymmetric_coef, alternate_weighting):
    """Ensures exponential overflow does not occur from aspls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    log_max = np.log(np.finfo(y_data.dtype).max)
    residual = y_data - baseline
    std = np.std(residual[residual < 0], ddof=1)
    # for exponential overflow, (residual / std) > (log_max / asymmetric_coef) + 1
    # changing one value in the baseline to cause overflow will not cause the
    # standard deviation to change since the residual value will be positive at that index
    overflow_index = 10
    overflow_value = ((log_max / asymmetric_coef) + 1) * std + 10  # add 10 for good measure
    if one_d:
        baseline[overflow_index] = y_data[overflow_index] - overflow_value
    else:
        baseline[overflow_index, overflow_index] = (
            y_data[overflow_index, overflow_index] - overflow_value
        )

    # sanity check to ensure overflow actually should occur
    with pytest.warns(RuntimeWarning):
        expected_weights, expected_alpha = expected_aspls(
            y_data, baseline, asymmetric_coef, alternate_weighting
        )
    # the resulting weights should still be finite since 1 / (1 + inf) == 0
    assert np.isfinite(expected_weights).all()

    with np.errstate(over='raise'):
        weights, exit_early, alpha = _weighting._aspls(
            y_data - baseline, asymmetric_coef=asymmetric_coef,
            alternate_weighting=alternate_weighting
        )

    assert np.isfinite(weights).all()
    assert not exit_early

    # the actual weight where overflow should have occurred should be 0
    if one_d:
        assert_allclose(weights[overflow_index], 0.0, atol=1e-14)
    else:
        assert_allclose(weights[overflow_index, overflow_index], 0.0, atol=1e-14)

    # weights should still be the same as the naive calculation regardless of exponential overflow
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_aspls_masked(one_d, asymmetric_coef, alternate_weighting):
    """Ensures aspls weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    mask_inv = np.logical_not(mask)

    weights, exit_early, alpha = _weighting._aspls(
        y_data - baseline, asymmetric_coef=asymmetric_coef,
        alternate_weighting=alternate_weighting, mask=mask
    )
    partial_weights, partial_alpha = expected_aspls(
       y_data[mask_inv], baseline[mask_inv], asymmetric_coef, alternate_weighting
    )
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights
    expected_alpha = np.ones_like(y_data)
    expected_alpha[mask_inv] = partial_alpha

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early

    assert isinstance(alpha, np.ndarray)
    assert alpha.shape == y_data.shape
    assert_allclose(alpha, expected_alpha, rtol=1e-12, atol=1e-12)


def expected_psalsa(y, baseline, p, k):
    """
    Weighting for the peaked signal's asymmetric least squares algorithm (psalsa).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight.
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
    residual = y - baseline
    weights = np.where(residual > 0, p * np.exp(-residual / k), 1 - p)
    return weights


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_psalsa_normal(one_d, k, p):
    """Ensures psalsa weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights = _weighting._psalsa(y_data - baseline, p=p, k=k)
    expected_weights = expected_psalsa(y_data, baseline, p, k)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_psalsa_all_above(one_d, k, p):
    """Ensures psalsa weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()

    weights = _weighting._psalsa(y_data - baseline, p=p, k=k)
    expected_weights = np.full_like(y_data, 1 - p)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_psalsa_all_below(one_d, k, p):
    """Ensures psalsa weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    weights = _weighting._psalsa(y_data - baseline, p=p, k=k)
    expected_weights = p * np.exp(-(y_data - baseline) / k)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_psalsa_overflow(one_d, k, p):
    """Ensures exponential overflow does not occur from psalsa weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    log_max = np.log(np.finfo(y_data.dtype).max)
    # for exponential overflow, -(residual / k) > log_max
    overflow_index = 10
    overflow_value = k * log_max + 10  # add 10 for good measure
    if one_d:
        baseline[overflow_index] = y_data[overflow_index] + overflow_value
    else:
        baseline[overflow_index, overflow_index] = (
            y_data[overflow_index, overflow_index] + overflow_value
        )

    # sanity check to ensure overflow actually should occur
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_psalsa(y_data, baseline, p, k)
    # weights in naive approach should still be finite since overflow only occurs in regions
    # where the exponential value is not actually used
    assert np.isfinite(expected_weights).all()

    with np.errstate(over='raise'):
        weights = _weighting._psalsa(y_data - baseline, p=p, k=k)

    assert np.isfinite(weights).all()

    # the actual weight where overflow should have occurred should be 1 - p
    if one_d:
        assert_allclose(weights[overflow_index], 1 - p, atol=1e-14)
    else:
        assert_allclose(weights[overflow_index, overflow_index], 1 - p, atol=1e-14)
    # weights should still be the same as the naive calculation regardless of exponential overflow
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('one_d', (True, False))
def test_psalsa_masked(one_d):
    """Ensures psalsa weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    mask_inv = np.logical_not(mask)
    p = 0.5
    k = 2

    weights = _weighting._psalsa(y_data - baseline, p=p, k=k, mask=mask)
    partial_weights = expected_psalsa(y_data[mask_inv], baseline[mask_inv], p, k)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def expected_derpsalsa(y, baseline, p, k, partial_weights):
    """
    Weights for derivative peak-screening asymmetric least squares algorithm (derpsalsa).

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    p : float
        The penalizing weighting factor. Must be between 0 and 1. Values greater
        than the baseline will be given `p` weight, and values less than the baseline
        will be given `1 - p` weight.
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
    residual = y - baseline
    weights = partial_weights * np.where(residual > 0, p * np.exp(-((residual / k)**2) / 2), 1 - p)
    return weights


@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_derpsalsa_normal(k, p):
    """Ensures derpsalsa weighting works as intended for a normal baseline."""
    y_data, baseline = baseline_1d_normal()

    diff_y_1 = np.gradient(y_data)
    diff_y_2 = np.gradient(diff_y_1)
    rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / y_data.size)
    rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / y_data.size)

    diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
    diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
    partial_weights = diff_1_weights * diff_2_weights

    weights = _weighting._derpsalsa(y_data - baseline, p=p, k=k, partial_weights=partial_weights)
    expected_weights = expected_derpsalsa(y_data, baseline, p, k, partial_weights)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_derpsalsa_all_above(k, p):
    """Ensures derpsalsa weighting works as intended for a baseline completely above the data."""
    y_data, baseline = baseline_1d_all_above()

    diff_y_1 = np.gradient(y_data)
    diff_y_2 = np.gradient(diff_y_1)
    rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / y_data.size)
    rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / y_data.size)

    diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
    diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
    partial_weights = diff_1_weights * diff_2_weights

    weights = _weighting._derpsalsa(y_data - baseline, p=p, k=k, partial_weights=partial_weights)
    expected_weights = np.full_like(y_data, partial_weights * (1 - p))

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('k', (0.5, 2))
@pytest.mark.parametrize('p', (0.01, 0.99))
def test_derpsalsa_all_below(k, p):
    """Ensures derpsalsa weighting works as intended for a baseline completely below the data."""
    y_data, baseline = baseline_1d_all_below()

    diff_y_1 = np.gradient(y_data)
    diff_y_2 = np.gradient(diff_y_1)
    rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / y_data.size)
    rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / y_data.size)

    diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
    diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
    partial_weights = diff_1_weights * diff_2_weights

    weights = _weighting._derpsalsa(y_data - baseline, p=p, k=k, partial_weights=partial_weights)
    expected_weights = partial_weights * p * np.exp(-0.5 * ((y_data - baseline) / k)**2)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def test_derpsalsa_masked():
    """Ensures derpsalsa weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d=True)
    mask_inv = np.logical_not(mask)
    p = 0.1
    k = 2

    diff_y_1 = np.gradient(y_data)
    diff_y_2 = np.gradient(diff_y_1)
    rms_diff_1 = np.sqrt(diff_y_1.dot(diff_y_1) / y_data.size)
    rms_diff_2 = np.sqrt(diff_y_2.dot(diff_y_2) / y_data.size)

    diff_1_weights = np.exp(-((diff_y_1 / rms_diff_1)**2) / 2)
    diff_2_weights = np.exp(-((diff_y_2 / rms_diff_2)**2) / 2)
    partial_weights = diff_1_weights * diff_2_weights

    weights = _weighting._derpsalsa(
        y_data - baseline, p=p, k=k, partial_weights=partial_weights, mask=mask
    )
    partial_weights = expected_derpsalsa(
        y_data[mask_inv], baseline[mask_inv], p, k, partial_weights[mask_inv]
    )
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)


def expected_brpls(y, baseline, beta):
    """
    The weighting for Bayesian Reweighted Penalized Least Squares (BrPLS).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    beta : float
        A value between 0 and 1 designating the probability of signal within the data.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

    References
    ----------
    Wang, Q., et al. Spectral baseline estimation using penalized least squares
    with weights derived from the Bayesian method. Nuclear Science and Techniques,
    2022, 140, 250-257.

    """
    residual = y - baseline
    # exclude residual == 0 to ensure mean and sigma are both nonzero since both
    # are used within the denominator
    neg_residual = residual[residual < 0]
    pos_residual = residual[residual > 0]

    mean = np.mean(pos_residual)
    sigma = np.sqrt((neg_residual**2).sum() / neg_residual.size)

    multiplier = ((beta * np.sqrt(0.5 * np.pi)) / (1 - beta)) * (sigma / mean)
    inner = (residual / (sigma * np.sqrt(2))) - (sigma / (mean * np.sqrt(2)))

    calc = (1 + erf(inner)) * np.exp(inner**2)
    # nan can apper where erf(x) = -1 and exp(x**2) = inf since 0 * inf = nan; in that
    # case, overflow should have been avoided so it would be 0 * large number -> set to 0
    calc[np.isnan(calc)] = 0

    weights = 1 / (1 + multiplier * calc)
    return weights


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('beta', (0.1, 0.5, 0.9))
def test_brpls_normal(one_d, beta):
    """Ensures brpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta)
    expected_weights = expected_brpls(y_data, baseline, beta)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('one_d', (True, False))
def test_brpls_all_above(one_d):
    """Ensures brpls weighting works as intended for a baseline with all points above the data."""
    beta = 0.5
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_brpls_all_below(one_d):
    """Ensures brpls weighting works as intended for a baseline with all points below the data."""
    beta = 0.5
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta)
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('beta', (0.1, 0.5, 0.9))
@pytest.mark.parametrize('positive', (True, False))
@pytest.mark.parametrize('dtype', (float, np.float32))
def test_brpls_overflow(one_d, beta, positive, dtype):
    """Ensures overflow does not occur from brpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()
    y_data = y_data.astype(dtype)
    baseline = baseline.astype(dtype)

    residual = y_data - baseline
    neg_residual = residual[residual < 0]
    pos_residual = residual[residual > 0]
    mean = np.mean(pos_residual)
    sigma = np.sqrt((neg_residual**2).sum() / neg_residual.size)

    sqrt_log_max = np.sqrt(np.log(np.finfo(y_data.dtype).max))
    # for exponential overflow, either
    # residual > sigma * sqrt(2) * (sqrt_log_max + sigma / (mean * sqrt(2)))
    # or residual < sigma * sqrt(2) * (-sqrt_log_max + sigma / (mean * sqrt(2)))
    # changing a positive or negative residual will change mean and sigma, so just
    # add 10000 to ensure overflow in either case
    max_val_pos = 10000 + np.sqrt(2) * sigma * (sqrt_log_max + sigma / (np.sqrt(2) * mean))
    max_val_neg = -10000 + np.sqrt(2) * sigma * (-sqrt_log_max + sigma / (np.sqrt(2) * mean))
    assert np.isfinite(max_val_neg)
    if positive:
        overflow_value = max_val_pos
        expected_value = 0.
    else:
        overflow_value = max_val_neg
        expected_value = 1.
    overflow_index = 10
    if one_d:
        baseline[overflow_index] = y_data[overflow_index] - overflow_value
    else:
        baseline[overflow_index, overflow_index] = (
            y_data[overflow_index, overflow_index] - overflow_value
        )

    # sanity check to ensure overflow should actually occur; note that both exponential and
    # multiplication overflow can occur
    with pytest.warns(RuntimeWarning):
        expected_weights = expected_brpls(y_data, baseline, beta)
    # the resulting weights should still be finite since 1 / (1 + inf) == 0
    assert np.isfinite(expected_weights).all()

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta)

    assert np.isfinite(weights).all()
    assert not exit_early

    # the actual weight where overflow should have occurred should be 0 or 1
    if one_d:
        assert_allclose(weights[overflow_index], expected_value, atol=1e-14)
        assert_allclose(expected_weights[overflow_index], expected_value, atol=1e-14)
    else:
        assert_allclose(weights[overflow_index, overflow_index], expected_value, atol=1e-14)
        assert_allclose(
            expected_weights[overflow_index, overflow_index], expected_value, atol=1e-14
        )

    # weights should still be the same as the naive calculation regardless
    # of overflow; have to use relatively high rtol to cover float32 cases
    assert_allclose(weights, expected_weights, rtol=5e-6, atol=1e-10)


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('beta', (0, 1))
def test_brpls_beta_extremes(one_d, beta):
    """Ensures beta values of 0 and 1 are handled correctly."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    if beta == 1:
        # the resulting weights should still be finite since 1 / (1 + inf) == 0
        with pytest.warns(RuntimeWarning):
            expected_weights = expected_brpls(y_data, baseline, beta)
        fill_value = 0
    else:
        expected_weights = expected_brpls(y_data, baseline, beta)
        fill_value = 1
    assert np.isfinite(expected_weights).all()
    assert_allclose(np.full(y_data.shape, fill_value), expected_weights, rtol=1e-10, atol=1e-10)

    with np.errstate(divide='raise'):
        weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta)

    assert np.isfinite(weights).all()
    assert not exit_early

    assert_allclose(weights, expected_weights, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('one_d', (True, False))
def test_brpls_masked(one_d):
    """Ensures brpls weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    mask_inv = np.logical_not(mask)
    beta = 0.5

    weights, exit_early = _weighting._brpls(y_data - baseline, beta=beta, mask=mask)
    partial_weights = expected_brpls(y_data[mask_inv], baseline[mask_inv], beta)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


def expected_lsrpls(y, baseline, iteration, alternate_weighting):
    """
    The weighting for the locally symmetric reweighted penalized least squares (lsrpls).

    Does not perform error checking since this is just used for simple weighting cases.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,) or (M, N)
        The measured data.
    baseline : numpy.ndarray, shape (N,) or (M, N)
        The calculated baseline.
    iteration : int
        The iteration number. Should be 1-based, such that the first iteration is 1
        instead of 0.
    alternate_weighting : bool
        If False, the weighting uses a prefactor term of ``10^t``, where ``t`` is
        the iteration number, which is equation 8 within the LSRPLS paper [1]_. If True, uses
        a prefactor term of ``exp(t)``. See the Notes section below for more details.

    Returns
    -------
    weights : numpy.ndarray, shape (N,) or (M, N)
        The calculated weights.

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
    residual = y - baseline
    neg_residual = residual[residual < 0]

    std = np.std(neg_residual, ddof=1)  # use dof=1 since sampling subset
    if alternate_weighting:
        prefactor = np.exp(iteration)
    else:
        prefactor = 10**iteration
    inner = (prefactor / std) * (residual - (2 * std - np.mean(neg_residual)))
    weights = 0.5 * (1 - (inner / (1 + np.abs(inner))))
    return weights


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_lsrpls_normal(iteration, one_d, alternate_weighting):
    """Ensures lsrpls weighting works as intended for a normal baseline."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    weights, exit_early = _weighting._lsrpls(
        y_data - baseline, iteration=iteration, alternate_weighting=alternate_weighting
    )
    expected_weights = expected_lsrpls(
        y_data, baseline, iteration, alternate_weighting
    )

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early
    assert ((weights >= 0) & (weights <= 1)).all()


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_lsrpls_all_above(iteration, one_d, alternate_weighting):
    """Ensures lsrpls weighting works as intended for a baseline with all points above the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_above()
    else:
        y_data, baseline = baseline_2d_all_above()
    weights, exit_early = _weighting._lsrpls(
        y_data - baseline, iteration=iteration, alternate_weighting=alternate_weighting
    )
    expected_weights = expected_lsrpls(y_data, baseline, iteration, alternate_weighting)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('iteration', (1, 10))
@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_lsrpls_all_below(iteration, one_d, alternate_weighting):
    """Ensures lsrpls weighting works as intended for a baseline with all points below the data."""
    if one_d:
        y_data, baseline = baseline_1d_all_below()
    else:
        y_data, baseline = baseline_2d_all_below()

    with pytest.warns(utils.ParameterWarning):
        weights, exit_early = _weighting._lsrpls(
            y_data - baseline, iteration=iteration, alternate_weighting=alternate_weighting
        )
    expected_weights = np.zeros_like(y_data)

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert exit_early


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_lsrpls_overflow(one_d, alternate_weighting):
    """Ensures exponential overflow does not occur from lsrpls weighting."""
    if one_d:
        y_data, baseline = baseline_1d_normal()
    else:
        y_data, baseline = baseline_2d_normal()

    iteration = 10000
    # sanity check to ensure overflow actually should occur
    if alternate_weighting:
        # exponential overflow only warns, so ensure that it produces nan weights
        with pytest.warns(RuntimeWarning):
            expected_weights = expected_drpls(y_data, baseline, iteration)
        assert (~np.isfinite(expected_weights)).any()
    else:
        with pytest.raises(OverflowError):
            expected_lsrpls(y_data, baseline, iteration, alternate_weighting)

    with np.errstate(over='raise'):
        weights, exit_early = _weighting._lsrpls(
            y_data - baseline, iteration=iteration, alternate_weighting=alternate_weighting
        )

    assert np.isfinite(weights).all()
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
@pytest.mark.parametrize('alternate_weighting', (True, False))
def test_lsrpls_masked(one_d, alternate_weighting):
    """Ensures lsrpls weighting works as intended for a normal baseline."""
    y_data, baseline, mask = data_and_mask(one_d)
    iteration = 5
    mask_inv = np.logical_not(mask)

    partial_weights = expected_drpls(y_data[mask_inv], baseline[mask_inv], iteration)
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    weights, exit_early = _weighting._lsrpls(
        y_data - baseline, iteration=iteration, alternate_weighting=alternate_weighting,
        mask=mask
    )
    partial_weights = expected_lsrpls(
        y_data[mask_inv], baseline[mask_inv], iteration, alternate_weighting
    )
    expected_weights = np.zeros_like(y_data)
    expected_weights[mask_inv] = partial_weights

    assert isinstance(weights, np.ndarray)
    assert weights.shape == y_data.shape
    assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
    assert not exit_early


@pytest.mark.parametrize('one_d', (True, False))
def test_masked_weighting(one_d):
    """Tests basic functionality of `masked_weighting`."""

    def func(residual, a, b=3):
        weights = np.where(residual > 0, 1., 0.)
        return weights, a, np.full_like(residual, b)

    wrapped_func = _weighting.masked_weighting(func)
    residual = np.linspace(-1, 1, 100)
    if not one_d:
        residual = residual.reshape(5, -1)
    a = 5.
    b = 10.
    expected_weights = np.where(residual > 0, 1., 0.)

    # sanity check for non wrapped func output
    normal_output = func(residual, a, b)
    assert len(normal_output) == 3
    assert_allclose(normal_output[0], expected_weights, rtol=1e-14, atol=1e-14)
    assert_allclose(normal_output[1], a, rtol=1e-14, atol=1e-14)
    assert_allclose(normal_output[2], b, rtol=1e-14, atol=1e-14)
    assert normal_output[2].shape == residual.shape

    # ensure keyword only after wrapping
    with pytest.raises(TypeError):
        wrapped_func(residual, a, b)
    with pytest.raises(TypeError):
        wrapped_func(residual, a, b=b)

    # call without a mask, should be same as non-wrapped
    wrapped_output = wrapped_func(residual, a=a, b=b)
    assert len(wrapped_output) == 3
    assert_allclose(wrapped_output[0], expected_weights, rtol=1e-14, atol=1e-14)
    assert_allclose(wrapped_output[1], a, rtol=1e-14, atol=1e-14)
    assert_allclose(wrapped_output[2], b, rtol=1e-14, atol=1e-14)
    assert wrapped_output[2].shape == residual.shape

    # masked call
    mask = np.zeros(residual.shape, dtype=bool)
    mask[..., 3:6] = True
    mask_inverse = np.logical_not(mask)
    masked_output = wrapped_func(residual, a=a, b=b, mask=mask)
    assert len(masked_output) == 3
    assert_allclose(
        masked_output[0], np.where(mask_inverse, expected_weights, 0),
        rtol=1e-14, atol=1e-14
    )
    assert_allclose(masked_output[1], a, rtol=1e-14, atol=1e-14)
    assert_allclose(masked_output[2], b, rtol=1e-14, atol=1e-14)
    assert masked_output[2].shape == residual[mask_inverse].shape
