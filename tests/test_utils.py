# -*- coding: utf-8 -*-
"""Tests for pybaselines.utils.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal, assert_array_equal
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


def test_relative_difference_scalar():
    """Tests relative_difference to ensure it uses abs for scalars."""
    assert_almost_equal(utils.relative_difference(3.0, 4), 1 / 3)


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


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, 5))
def test_difference_matrix(diff_order):
    """Tests all covered differential matrices."""
    diff_matrix = utils.difference_matrix(10, diff_order).toarray()
    numpy_diff = np.diff(np.eye(10), diff_order).T

    assert_array_equal(diff_matrix, numpy_diff)


def test_difference_matrix_order2():
    """
    Tests the 2nd order differential matrix against the actual representation.

    The 2nd order differential matrix is most commonly used,
    so double-check that it is correct.
    """
    diff_matrix = utils.difference_matrix(8, 2).toarray()
    actual_matrix = np.array([
        [1, -2, 1, 0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0, 0, 0, 0],
        [0, 0, 1, -2, 1, 0, 0, 0],
        [0, 0, 0, 1, -2, 1, 0, 0],
        [0, 0, 0, 0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0, 1, -2, 1]
    ])

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_fail():
    """Ensures differential matrix fails for non-covered order."""
    with pytest.raises(ValueError):
        utils.difference_matrix(10, diff_order=10)
