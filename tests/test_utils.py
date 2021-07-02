# -*- coding: utf-8 -*-
"""Tests for pybaselines.utils.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import pytest

from pybaselines import utils

from .conftest import gaussian


@pytest.fixture(scope='module')
def _x_data():
    """x-values for testing."""
    return np.linspace(-20, 20)


@pytest.mark.parametrize('sigma', [0.1, 1, 10])
@pytest.mark.parametrize('center', [-10, 0, 10])
@pytest.mark.parametrize('height', [0.1, 1, 10])
def test_gaussian(_x_data, height, center, sigma):
    """Ensures that gaussian function in pybaselines.utils is correct."""
    assert_array_almost_equal(
        utils.gaussian(_x_data, height, center, sigma),
        gaussian(_x_data, height, center, sigma)
    )


@pytest.mark.parametrize('window_size', (1, 20, 100))
@pytest.mark.parametrize('sigma', (1, 2, 5))
def test_gaussian_kernel(window_size, sigma):
    """
    Tests gaussian_kernel for various window_sizes and sigma values.

    Ensures area is always 1, so that kernel is normalized.
    """
    kernel = utils.gaussian_kernel(window_size, sigma)

    assert kernel.size == window_size
    assert kernel.shape == (window_size,)
    assert_almost_equal(np.sum(kernel), 1)


@pytest.mark.parametrize('sign', (1, -1))
def test_relative_difference_scalar(sign):
    """Tests relative_difference to ensure it uses abs for scalars."""
    old = 3.0 * sign
    new = 4
    assert_almost_equal(utils.relative_difference(old, new), abs((old - new) / old))


def test_relative_difference_array():
    """Tests relative_difference to ensure it uses l2 norm for arrays."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    norm_ab = np.sqrt(((a - b)**2).sum())
    norm_a = np.sqrt(((a)**2).sum())

    assert_almost_equal(utils.relative_difference(a, b), norm_ab / norm_a)


def test_relative_difference_array_l1():
    """Tests `norm_order` keyword for relative_difference."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    norm_ab = np.abs(a - b).sum()
    norm_a = np.abs(a).sum()

    assert_almost_equal(utils.relative_difference(a, b, 1), norm_ab / norm_a)


def test_relative_difference_zero():
    """Ensures relative difference works when 0 is the denominator."""
    a = np.array([0, 0, 0])
    b = np.array([4, 5, 6])
    norm_ab = np.sqrt(((a - b)**2).sum())

    assert_almost_equal(utils.relative_difference(a, b), norm_ab / np.finfo(float).eps)


def test_safe_std():
    """Checks that the calculated standard deviation is correct."""
    array = np.array((1, 2, 3))
    calc_std = utils._safe_std(array)

    assert_almost_equal(calc_std, np.std(array))


def test_safe_std_kwargs():
    """Checks that kwargs given to _safe_std are passed to numpy.std."""
    array = np.array((1, 2, 3))
    calc_std = utils._safe_std(array, ddof=1)

    assert_almost_equal(calc_std, np.std(array, ddof=1))


def test_safe_std_empty():
    """Checks that the returned standard deviation of an empty array is not nan."""
    calc_std = utils._safe_std(np.array(()))
    assert_almost_equal(calc_std, utils._MIN_FLOAT)


def test_safe_std_single():
    """Checks that the returned standard deviation of an array with a single value is not 0."""
    calc_std = utils._safe_std(np.array((1,)))
    assert_almost_equal(calc_std, utils._MIN_FLOAT)


def test_safe_std_zero():
    """Checks that the returned standard deviation is not 0."""
    calc_std = utils._safe_std(np.array((1, 1, 1)))
    assert_almost_equal(calc_std, utils._MIN_FLOAT)


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

    assert np.isnan(utils._safe_std(array))


def test_interp_inplace():
    """Tests that _interp_inplace modified the input array inplace."""
    x = np.arange(10)
    y_actual = 2 + 5 * x

    y_calc = np.empty_like(y_actual)
    y_calc[0] = y_actual[0]
    y_calc[-1] = y_actual[-1]

    output = utils._interp_inplace(x, y_calc)

    # should not output anything from the function
    assert output is None

    assert_array_almost_equal(y_calc, y_actual)
