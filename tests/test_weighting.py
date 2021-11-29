# -*- coding: utf-8 -*-
"""Tests for pybaselines._weighting."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import _weighting, utils


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
