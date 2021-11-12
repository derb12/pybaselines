# -*- coding: utf-8 -*-
"""Tests for pybaselines._algorithm_setup.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import dia_matrix, identity
from scipy.sparse.linalg import spsolve

from pybaselines import _algorithm_setup, utils
from pybaselines.utils import ParameterWarning


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('lower_only', (True, False))
def test_diff_2_diags(data_size, lower_only):
    """Ensures the output of _diff_2_diags is the correct shape and values."""
    diagonal_data = _algorithm_setup._diff_2_diags(data_size, lower_only)

    diff_matrix = utils.difference_matrix(data_size, 2)
    diag_matrix = (diff_matrix.T @ diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data[::-1]
    if lower_only:
        actual_diagonal_data = actual_diagonal_data[2:]

    assert_array_equal(diagonal_data, actual_diagonal_data)


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('lower_only', (True, False))
def test_diff_1_diags(data_size, lower_only):
    """Ensures the output of _diff_1_diags is the correct shape and values."""
    diagonal_data = _algorithm_setup._diff_1_diags(data_size, lower_only)

    diff_matrix = utils.difference_matrix(data_size, 1)
    diag_matrix = (diff_matrix.T @ diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data[::-1]
    if lower_only:
        actual_diagonal_data = actual_diagonal_data[1:]

    assert_array_equal(diagonal_data, actual_diagonal_data)


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('lower_only', (True, False))
def test_diff_3_diags(data_size, lower_only):
    """Ensures the output of _diff_3_diags is the correct shape and values."""
    diagonal_data = _algorithm_setup._diff_3_diags(data_size, lower_only)

    diff_matrix = utils.difference_matrix(data_size, 3)
    diag_matrix = (diff_matrix.T @ diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data[::-1]
    if lower_only:
        actual_diagonal_data = actual_diagonal_data[3:]

    assert_array_equal(diagonal_data, actual_diagonal_data)


@pytest.mark.parametrize('data_size', (10, 1001))
@pytest.mark.parametrize('diff_order', (0, 1, 2, 3, 4, 5, 6, 7, 8))
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('padding', (-1, 0, 1, 2))
def test_diff_penalty_diagonals(data_size, diff_order, lower_only, padding):
    """
    Ensures the penalty matrix (squared finite difference matrix) diagonals are correct.

    Also tests the condition for when `data_size` < 2 * `diff_order` + 1 to ensure
    the slower, sparse route is taken.

    """
    diagonal_data = _algorithm_setup.diff_penalty_diagonals(
        data_size, diff_order, lower_only, padding
    )

    diff_matrix = utils.difference_matrix(data_size, diff_order)
    diag_matrix = (diff_matrix.T @ diff_matrix).todia()
    actual_diagonal_data = diag_matrix.data[::-1]
    if lower_only:
        actual_diagonal_data = actual_diagonal_data[diff_order:]
    if padding > 0:
        pad_layers = np.repeat(np.zeros((1, data_size)), padding, axis=0)
        if lower_only:
            actual_diagonal_data = np.concatenate((actual_diagonal_data, pad_layers))
        else:
            actual_diagonal_data = np.concatenate((pad_layers, actual_diagonal_data, pad_layers))

    assert_array_equal(diagonal_data, actual_diagonal_data)


def test_diff_penalty_diagonals_order_neg():
    """Ensures penalty matrix fails for negative order."""
    with pytest.raises(ValueError):
        _algorithm_setup.diff_penalty_diagonals(10, -1)


def test_diff_penalty_diagonals_datasize_too_small():
    """Ensures penalty matrix fails for data size <= 0."""
    with pytest.raises(ValueError):
        _algorithm_setup.diff_penalty_diagonals(0)
    with pytest.raises(ValueError):
        _algorithm_setup.diff_penalty_diagonals(-1)


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
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('reverse_diags', (True, False))
def test_setup_whittaker_diff_matrix(small_data, lam, diff_order, lower_only, reverse_diags):
    """Ensures output difference matrix diagonal data is in desired format."""
    if reverse_diags and lower_only:
        # this configuration is never used
        return

    _, diagonal_data, _ = _algorithm_setup._setup_whittaker(
        small_data, lam, diff_order, lower_only=lower_only, reverse_diags=reverse_diags
    )

    numpy_diff = np.diff(np.eye(small_data.shape[0]), diff_order, 0)
    desired_diagonals = dia_matrix(lam * (numpy_diff.T @ numpy_diff)).data[::-1]
    if lower_only:  # only include the lower diagonals
        desired_diagonals = desired_diagonals[diff_order:]

    # the diagonals should be in the opposite order as the diagonal matrix's data
    # if reverse_diags is False
    if reverse_diags:
        desired_diagonals = desired_diagonals[::-1]

    assert_allclose(diagonal_data, desired_diagonals, 1e-10)


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


def test_setup_whittaker_negative_lam_fails(small_data):
    """Ensures a negative lam value fails."""
    with pytest.raises(ValueError):
        _algorithm_setup._setup_whittaker(small_data, -1)


def test_setup_whittaker_array_lam(small_data):
    """Ensures a lam that is a single array passes while larger arrays fail."""
    _algorithm_setup._setup_whittaker(small_data, [1])
    with pytest.raises(ValueError):
        _algorithm_setup._setup_whittaker(small_data, [1, 2])


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


def test_setup_polynomial_wrong_weight_shape(small_data):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        _algorithm_setup._setup_polynomial(small_data, weights=weights)


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
    assert_allclose(desired_vander, vander_matrix, 1e-12)

    if include_pinv:
        desired_pinv = np.linalg.pinv(np.sqrt(weight_array)[:, np.newaxis] * desired_vander)
        assert_allclose(desired_pinv, pinv_matrix, 1e-10)


@pytest.mark.parametrize('array_enum', (0, 1))
def test_setup_smooth_y_array(small_data, array_enum):
    """Ensures output y is always a numpy array."""
    if array_enum == 1:
        small_data = small_data.tolist()
    y = _algorithm_setup._setup_smooth(small_data, 1)

    assert isinstance(y, np.ndarray)


def test_setup_smooth_shape(small_data):
    """Ensures output y is correctly padded."""
    pad_length = 4
    y = _algorithm_setup._setup_smooth(small_data, pad_length, mode='edge')
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


@pytest.mark.parametrize('num_knots', (5, 15, 100))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('penalized', (True, False))
def test_setup_splines_spline_basis(small_data, num_knots, spline_degree, penalized):
    """Ensures the spline basis function is correctly created."""
    _, _, _, basis, _, _ = _algorithm_setup._setup_splines(
        small_data, None, None, spline_degree, num_knots
    )

    assert basis.shape[0] == len(small_data)
    # num_knots == number of inner knots with min and max points counting as
    # the first and last inner knots; then add `degree` extra knots
    # on each end to accomodate the final polynomial on each end; therefore,
    # total number of knots = num_knots + 2 * degree; the number of basis
    # functions is total knots - (degree + 1), so the ultimate
    # shape of the basis matrix should be num_knots + degree - 1
    assert basis.shape[1] == num_knots + spline_degree - 1


@pytest.mark.parametrize('lam', (1, 20))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (5, 50, 100))
def test_setup_splines_diff_matrix(small_data, lam, diff_order, spline_degree, num_knots):
    """Ensures output difference matrix diagonal data is in desired format."""
    *_, penalty_diagonals = _algorithm_setup._setup_splines(
        small_data, None, None, spline_degree, num_knots, True, diff_order, lam
    )

    num_bases = num_knots + spline_degree - 1
    numpy_diff = np.diff(np.eye(num_bases), diff_order, axis=0)
    desired_diagonals = lam * dia_matrix(numpy_diff.T @ numpy_diff).data[::-1][diff_order:]
    if diff_order < spline_degree:
        padding = np.zeros((spline_degree - diff_order, desired_diagonals.shape[1]))
        desired_diagonals = np.concatenate((desired_diagonals, padding))

    assert_allclose(penalty_diagonals, desired_diagonals, 1e-10, 1e-12)


@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (5, 50, 100))
def test_setup_splines_too_high_diff_order(small_data, spline_degree, num_knots):
    """
    Ensures an exception is raised when the difference order is >= number of basis functions.

    The number of basis functions is equal to the number of knots + the spline degree - 1.
    Tests both difference order equal to and greater than the number of basis functions.

    """
    diff_order = num_knots + spline_degree - 1
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(
            small_data, None, None, spline_degree, num_knots, True, diff_order
        )

    diff_order += 1
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(
            small_data, None, None, spline_degree, num_knots, True, diff_order
        )


@pytest.mark.parametrize('num_knots', (0, 1))
def test_setup_splines_too_few_knots(small_data, num_knots):
    """Ensures an error is raised if the number of knots is less than 2."""
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(
            small_data, None, None, 3, num_knots, True, 1
        )


def test_setup_splines_wrong_weight_shape(small_data):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(small_data, weights=weights)


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_splines_diff_matrix_fails(small_data, diff_order):
    """Ensures using a diff_order < 1 with _setup_splines raises an exception."""
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(small_data, diff_order=diff_order)


@pytest.mark.parametrize('diff_order', (5, 6))
def test_setup_splines_diff_matrix_warns(small_data, diff_order):
    """Ensures using a diff_order > 4 with _setup_splines raises a warning."""
    with pytest.warns(ParameterWarning):
        _algorithm_setup._setup_splines(small_data, diff_order=diff_order)


def test_setup_splines_negative_lam_fails(small_data):
    """Ensures a negative lam value fails."""
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(small_data, lam=-1)


def test_setup_splines_array_lam(small_data):
    """Ensures a lam that is a single array passes while larger arrays fail."""
    _algorithm_setup._setup_whittaker(small_data, lam=[1])
    with pytest.raises(ValueError):
        _algorithm_setup._setup_splines(small_data, lam=[1, 2])


@pytest.mark.parametrize('lam', (5, [5], (5,), [[5]], np.array(5), np.array([5]), np.array([[5]])))
def test_check_lam(lam):
    """Ensures scalar lam values are correctly processed."""
    output_lam = _algorithm_setup._check_lam(lam)
    assert output_lam == 5


def test_check_lam_failures():
    """Ensures array-like values or values < or <= 0 fail."""
    # fails due to array of values
    with pytest.raises(ValueError):
        _algorithm_setup._check_lam([5, 10])

    # fails for lam <= 0 when allow_zero is False
    for lam in range(-5, 1):
        with pytest.raises(ValueError):
            _algorithm_setup._check_lam(lam)

    # test that is allows zero if allow_zero is True
    _algorithm_setup._check_lam(0, True)
    for lam in range(-5, 0):
        with pytest.raises(ValueError):
            _algorithm_setup._check_lam(lam, True)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
def test_whittaker_smooth(data_fixture, diff_order):
    """Ensures the Whittaker smoothing function performs correctly."""
    x, y = data_fixture
    lam = 1
    output = _algorithm_setup._whittaker_smooth(y, lam, diff_order)

    assert isinstance(output[0], np.ndarray)
    assert isinstance(output[1], np.ndarray)

    # construct the sparse solution and compare
    len_y = len(y)
    diff_matrix = utils.difference_matrix(len_y, diff_order, 'csc')
    penalty = lam * (diff_matrix.T @ diff_matrix)

    # solve the simple case for all weights are 1
    expected_output = spsolve(identity(len_y) + penalty, y)

    assert_allclose(output[0], expected_output, 1e-6)
