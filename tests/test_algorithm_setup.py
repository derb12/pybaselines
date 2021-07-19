# -*- coding: utf-8 -*-
"""Tests for pybaselines._algorithm_setup.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from scipy.sparse import dia_matrix, identity

from pybaselines import _algorithm_setup
from pybaselines.utils import ParameterWarning


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
    """Ensures differential matrix fails for negative order."""
    with pytest.raises(ValueError):
        _algorithm_setup.difference_matrix(10, diff_order=-2)


def test_difference_matrix_order_over():
    """
    Tests the (n + 1)th order differential matrix against the actual representation.

    If n is the number of data points and the difference order is greater than n,
    then differential matrix should have a shape of (0, n) with 0 stored elements,
    following a similar logic as np.diff.

    """
    diff_matrix = _algorithm_setup.difference_matrix(10, 11).toarray()
    actual_matrix = np.empty(shape=(0, 10))

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_size_neg():
    """Ensures differential matrix fails for negative data size."""
    with pytest.raises(ValueError):
        _algorithm_setup.difference_matrix(-1)


@pytest.mark.parametrize('form', ('dia', 'csc', 'csr'))
def test_difference_matrix_formats(form):
    """
    Ensures that the sparse format is correctly passed to the constructor.

    Tests both 0-order and 2-order, since 0-order uses a different constructor.
    """
    assert _algorithm_setup.difference_matrix(10, 2, form).format == form
    assert _algorithm_setup.difference_matrix(10, 0, form).format == form


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('upper_only', (True, False))
def test_diff_2_diags(data_size, upper_only):
    """Ensures the output of _diff_2_diags is the correct shape and values."""
    diagonal_data = _algorithm_setup._diff_2_diags(data_size, upper_only)

    diff_matrix = _algorithm_setup.difference_matrix(data_size, 2)
    diag_matrix = (diff_matrix.T * diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data
    if upper_only:
        actual_diagonal_data = diag_matrix.data[2:]

    assert_array_equal(diagonal_data, actual_diagonal_data[::-1])


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('upper_only', (True, False))
def test_diff_1_diags(data_size, upper_only):
    """Ensures the output of _diff_1_diags is the correct shape and values."""
    diagonal_data = _algorithm_setup._diff_1_diags(data_size, upper_only)

    diff_matrix = _algorithm_setup.difference_matrix(data_size, 1)
    diag_matrix = (diff_matrix.T * diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data
    if upper_only:
        actual_diagonal_data = diag_matrix.data[1:]

    assert_array_equal(diagonal_data, actual_diagonal_data[::-1])


@pytest.fixture
def small_data():
    """A small array of data for testing."""
    return np.arange(10, dtype=float)


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_whittaker_y_array(small_data, array_enum):
    """Ensures output y is always a numpy array."""
    if array_enum == 1:
        small_data = small_data.tolist()
    y, *_ = _algorithm_setup._setup_whittaker(small_data, 1)

    assert isinstance(y, np.ndarray)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('lam', (1, 20))
@pytest.mark.parametrize('upper_only', (True, False))
@pytest.mark.parametrize('reverse_diags', (True, False))
def test_setup_whittaker_diff_matrix(small_data, lam, diff_order, upper_only, reverse_diags):
    """Ensures output difference matrix diagonal data is in desired format."""
    _, diagonal_data, _ = _algorithm_setup._setup_whittaker(
        small_data, lam, diff_order, upper_only=upper_only, reverse_diags=reverse_diags
    )

    # numpy gives transpose of the desired differential matrix
    numpy_diff = np.diff(np.eye(small_data.shape[0]), diff_order).T
    desired_diagonals = dia_matrix(lam * np.dot(numpy_diff.T, numpy_diff)).data
    if upper_only:  # only include the upper diagonals
        desired_diagonals = desired_diagonals[diff_order:]

    # the diagonals should be in the opposite order as the diagonal matrix's data
    # if reverse_diags is False
    if not reverse_diags:
        desired_diagonals = desired_diagonals[::-1]

    assert_array_almost_equal(diagonal_data, desired_diagonals)


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_whittaker_weights(small_data, weight_enum):
    """Ensures output weight array is correct."""
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

    _, _, weight_array = _algorithm_setup._setup_whittaker(small_data, 1, 2, weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


@pytest.mark.parametrize('filler', (np.nan, np.inf, -np.inf))
def test_setup_whittaker_nan_inf_data_fails(small_data, filler):
    """Ensures NaN and Inf values within data will raise an exception."""
    small_data[0] = filler
    with pytest.raises(ValueError):
        _algorithm_setup._setup_whittaker(small_data, 1)


def test_setup_whittaker_wrong_weight_shape(small_data):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        _algorithm_setup._setup_whittaker(small_data, 1, 2, weights)


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_whittaker_diff_matrix_fails(small_data, diff_order):
    """Ensures using a diff_order < 1 with _setup_whittaker raises an exception."""
    with pytest.raises(ValueError):
        _algorithm_setup._setup_whittaker(small_data, 1, diff_order)


@pytest.mark.parametrize('diff_order', (4, 5))
def test_setup_whittaker_diff_matrix_warns(small_data, diff_order):
    """Ensures using a diff_order > 3 with _setup_whittaker raises a warning."""
    with pytest.warns(ParameterWarning):
        _algorithm_setup._setup_whittaker(small_data, 1, diff_order)


@pytest.mark.parametrize('array_enum', (0, 1))
def test_yx_arrays_output_array(small_data, array_enum):
    """Ensures output y and x are always numpy arrays and that x is not scaled."""
    if array_enum == 1:
        small_data = small_data.tolist()
    x_data = small_data.copy()
    y, x = _algorithm_setup._yx_arrays(small_data, x_data)

    actual_array = np.asarray(small_data)

    assert isinstance(y, np.ndarray)
    assert_array_equal(y, actual_array)
    assert isinstance(x, np.ndarray)
    assert_array_equal(x, actual_array)


def test_yx_arrays_no_x(small_data):
    """Ensures an x array is created if None is input."""
    y, x = _algorithm_setup._yx_arrays(small_data)

    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_polynomial_output_array(small_data, array_enum):
    """
    Ensures output y and x are always numpy arrays and that x is scaled to [-1., 1.].

    Similar test as for _yx_arrays, but want to double-check that
    output is correct.
    """
    if array_enum == 1:
        small_data = small_data.tolist()
    x_data = small_data.copy()
    y, x, *_ = _algorithm_setup._setup_polynomial(small_data, x_data)

    assert isinstance(y, np.ndarray)
    assert_array_equal(y, np.asarray(small_data))
    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


def test_setup_polynomial_no_x(small_data):
    """
    Ensures an x array is created if None is input.

    Same test as for _yx_arrays, but want to double-check that
    output is correct.
    """
    y, x, *_ = _algorithm_setup._setup_polynomial(small_data)

    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


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
@pytest.mark.parametrize('include_pinv', (True, False))
def test_setup_polynomial_vandermonde(small_data, vander_enum, include_pinv):
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

    output = _algorithm_setup._setup_polynomial(
        small_data, small_data, weights, poly_order, True, include_pinv
    )
    if include_pinv:
        _, x, weight_array, _, vander_matrix, pinv_matrix = output
    else:
        _, x, weight_array, _, vander_matrix = output

    desired_vander = np.polynomial.polynomial.polyvander(x, poly_order)
    assert_array_almost_equal(desired_vander, vander_matrix)

    if include_pinv:
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


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_classification_output_array(small_data, array_enum):
    """Ensures output y and x are always numpy arrays and that x is scaled to [-1., 1.]."""
    if array_enum == 1:
        small_data = small_data.tolist()
    x_data = small_data.copy()
    y, x, *_ = _algorithm_setup._setup_polynomial(small_data, x_data)

    assert isinstance(y, np.ndarray)
    assert_array_equal(y, np.asarray(small_data))
    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


def test_setup_classification_no_x(small_data):
    """
    Ensures an x array is created if None is input.

    Same test as for _yx_arrays, but want to double-check that
    output is correct.
    """
    y, x, *_ = _algorithm_setup._setup_classification(small_data)

    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_classification_weights(small_data, weight_enum):
    """Ensures output weight array is correctly handled in classification setup."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones(small_data.shape[0], bool)
    elif weight_enum == 1:
        # uniform 1 weighting, input as boolean dtype
        weights = np.ones(small_data.shape[0], bool)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points, input as ints
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0]).astype(bool)
    elif weight_enum == 3:
        # different weights for all points, weights input as a list of floats
        weights = np.arange(small_data.shape[0], dtype=float).tolist()
        desired_weights = np.arange(small_data.shape[0]).astype(bool)

    _, _, weight_array, _ = _algorithm_setup._setup_classification(small_data, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


def test_setup_classification_domain(small_data):
    """Ensures output domain array is correct."""
    x = np.linspace(-5, 20, len(small_data))

    *_, original_domain = _algorithm_setup._setup_classification(small_data, x)

    assert_array_equal(original_domain, np.array([-5, 20]))
