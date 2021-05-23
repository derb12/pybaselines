# -*- coding: utf-8 -*-
"""Tests for pybaselines._algorithm_setup.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from scipy.sparse import identity

from pybaselines import _algorithm_setup


@pytest.mark.parametrize('diff_order', (0, 1, 2, 3, 4, 5))
def test_difference_matrix(diff_order):
    """Tests common differential matrices."""
    diff_matrix = _algorithm_setup.difference_matrix(10, diff_order).toarray()
    numpy_diff = np.diff(np.eye(10), diff_order).T

    assert_array_equal(diff_matrix, numpy_diff)


def test_difference_matrix_order_2():
    """
    Tests the 2nd order differential matrix against the actual representation.

    The 2nd order differential matrix is most commonly used,
    so double-check that it is correct.
    """
    diff_matrix = _algorithm_setup.difference_matrix(8, 2).toarray()
    actual_matrix = np.array([
        [1, -2, 1, 0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0, 0, 0, 0],
        [0, 0, 1, -2, 1, 0, 0, 0],
        [0, 0, 0, 1, -2, 1, 0, 0],
        [0, 0, 0, 0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0, 1, -2, 1]
    ])

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_order_0():
    """
    Tests the 0th order differential matrix against the actual representation.

    The 0th order differential matrix should be the same as the identity matrix,
    so double-check that it is correct.
    """
    diff_matrix = _algorithm_setup.difference_matrix(10, 0).toarray()
    actual_matrix = identity(10).toarray()

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_order_neg():
    """Ensures differential matrix fails for non-positive order."""
    with pytest.raises(ValueError):
        _algorithm_setup.difference_matrix(10, diff_order=-2)


@pytest.fixture
def small_data():
    """A small array of data for testing."""
    return np.arange(10)


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_whittacker_y_array(small_data, array_enum):
    """Ensures output y is always a numpy array."""
    if array_enum == 1:
        small_data = small_data.tolist()
    y, *_ = _algorithm_setup._setup_whittaker(small_data, 1)

    assert isinstance(y, np.ndarray)


@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('lam', (1, 20))
def test_setup_whittacker_diff_matrix(small_data, lam, diff_order):
    """Ensures output difference matrix is lam * diff_matrix.T * diff_matrix."""
    _, diff_matrix, *_ = _algorithm_setup._setup_whittaker(small_data, lam, diff_order)

    # numpy gives transpose of the desired differential matrix
    numpy_diff = np.diff(np.eye(small_data.shape[0]), diff_order).T
    desired_diff_matrix = lam * np.dot(numpy_diff.T, numpy_diff)

    assert_array_almost_equal(diff_matrix.toarray(), desired_diff_matrix)


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_whittacker_weights(small_data, weight_enum):
    """Ensures output weight matrix and array are correct."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0])
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        desired_weights = np.arange(small_data.shape[0])

    _, _, weight_matrix, weight_array = _algorithm_setup._setup_whittaker(small_data, 1, 2, weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert_array_equal(weight_matrix.toarray(), np.diag(desired_weights))


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_polynomial_output_array(small_data, array_enum):
    """Ensures output y and x are always numpy arrays and that x is scaled to [-1, 1]."""
    if array_enum == 1:
        small_data = small_data.tolist()
    x_data = small_data.copy()
    y, x, *_ = _algorithm_setup._setup_polynomial(small_data, x_data)

    assert isinstance(y, np.ndarray)
    assert_array_equal(y, np.asarray(small_data))
    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1, 1, y.shape[0]))


def test_setup_polynomial_no_x(small_data):
    """Ensures an x array is created if None is input."""
    y, x, *_ = _algorithm_setup._setup_polynomial(small_data)

    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1, 1, y.shape[0]))


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_polynomial_weights(small_data, weight_enum):
    """Ensures output weight array is correctly handled."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0])
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        desired_weights = np.arange(small_data.shape[0])

    _, _, weight_array, _ = _algorithm_setup._setup_polynomial(small_data, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


def test_setup_polynomial_domain(small_data):
    """Ensures output domain array is correct."""
    x = np.linspace(-5, 20, len(small_data))

    *_, original_domain = _algorithm_setup._setup_polynomial(small_data, x)

    assert_array_equal(original_domain, np.array([-5, 20]))


@pytest.mark.parametrize('vander_enum', (0, 1, 2, 3))
def test_setup_polynomial_vandermonde(small_data, vander_enum):
    """Ensures that the Vandermonde matrix and the pseudo-inverse matrix are correct."""
    if vander_enum == 0:
        # no weights specified
        weights = None
        poly_order = 2
    elif vander_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        poly_order = 4
    elif vander_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        poly_order = 2
    elif vander_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        poly_order = 4

    _, x, weight_array, _, vander_matrix, pinv_matrix = _algorithm_setup._setup_polynomial(
        small_data, small_data, weights, poly_order, True, True
    )

    desired_vander = np.polynomial.polynomial.polyvander(x, poly_order)
    assert_array_almost_equal(desired_vander, vander_matrix)

    desired_pinv = np.linalg.pinv(np.sqrt(weight_array)[:, np.newaxis] * desired_vander)
    assert_array_almost_equal(desired_pinv, pinv_matrix)


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_window_y_array(small_data, array_enum):
    """Ensures output y is always a numpy array."""
    if array_enum == 1:
        small_data = small_data.tolist()
    y = _algorithm_setup._setup_window(small_data, 1)

    assert isinstance(y, np.ndarray)


def test_setup_window_shape(small_data):
    """Ensures output y is correctly padded."""
    pad_length = 4
    y = _algorithm_setup._setup_window(small_data, pad_length, mode='edge')
    assert y.shape[0] == small_data.shape[0] + 2 * pad_length
