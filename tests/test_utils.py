# -*- coding: utf-8 -*-
"""Tests for pybaselines.utils.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import (
    assert_allclose, assert_almost_equal, assert_array_almost_equal, assert_array_equal
)
import pytest
from scipy.sparse import identity

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

    # output should be the same object as the input y array
    assert output is y_calc

    assert_array_almost_equal(y_calc, y_actual)


def test_interp_inplace_endpoints():
    """Tests _interp_inplace when specifying the endpoint y-values."""
    x = np.arange(10)
    y_actual = 2 + 5 * x

    # specify both the left and right points
    y_calc = np.zeros_like(y_actual)
    output = utils._interp_inplace(x, y_calc, y_actual[0], y_actual[-1])

    # output should be the same object as the input y array
    assert output is y_calc
    assert_array_almost_equal(y_calc[1:-1], y_actual[1:-1])
    # first and last values should still be 0
    assert y_calc[0] == 0
    assert y_calc[-1] == 0

    # specify only the right point
    y_calc = np.zeros_like(y_actual)
    y_calc[0] = y_actual[0]
    utils._interp_inplace(x, y_calc, None, y_actual[-1])

    assert_array_almost_equal(y_calc[:-1], y_actual[:-1])
    assert y_calc[-1] == 0

    # specify only the left point
    y_calc = np.zeros_like(y_actual)
    y_calc[-1] = y_actual[-1]
    utils._interp_inplace(x, y_calc, y_actual[0])

    assert_array_almost_equal(y_calc[1:], y_actual[1:])
    assert y_calc[0] == 0


@pytest.mark.parametrize('x', (np.array([-5, -2, 0, 1, 8]), np.array([1, 2, 3, 4, 5])))
@pytest.mark.parametrize(
    'coefs', (
        np.array([1, 2]), np.array([-1, 10, 0.2]), np.array([0, 1, 0]),
        np.array([0, 0, 0]), np.array([2, 1e-19])
    )
)
def test_convert_coef(x, coefs):
    """Checks that polynomial coefficients are correctly converted to the original domain."""
    original_domain = np.array([x.min(), x.max()])
    y = np.zeros_like(x)
    for i, coef in enumerate(coefs):
        y = y + coef * x**i

    fit_polynomial = np.polynomial.Polynomial.fit(x, y, coefs.size - 1)
    # fit_coefs correspond to the domain [-1, 1] rather than the original
    # domain of x
    fit_coefs = fit_polynomial.coef

    converted_coefs = utils._convert_coef(fit_coefs, original_domain)

    assert_allclose(converted_coefs, coefs, atol=1e-10)


@pytest.mark.parametrize('quantile', np.linspace(0, 1, 21))
def test_quantile_loss(quantile):
    """Ensures the quantile loss calculation is correct."""
    y = np.linspace(-1, 1)
    fit = np.zeros(y.shape[0])
    residual = y - fit
    eps = 1e-10
    calc_loss = utils._quantile_loss(y, fit, quantile, eps)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    denominator = np.sqrt(residual**2 + eps)

    expected_loss = numerator / denominator

    assert_allclose(calc_loss, expected_loss)


@pytest.mark.parametrize('diff_order', (0, 1, 2, 3, 4, 5))
def test_difference_matrix(diff_order):
    """Tests common differential matrices."""
    diff_matrix = utils.difference_matrix(10, diff_order).toarray()
    numpy_diff = np.diff(np.eye(10), diff_order).T

    assert_array_equal(diff_matrix, numpy_diff)


def test_difference_matrix_order_2():
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


def test_difference_matrix_order_0():
    """
    Tests the 0th order differential matrix against the actual representation.

    The 0th order differential matrix should be the same as the identity matrix,
    so double-check that it is correct.
    """
    diff_matrix = utils.difference_matrix(10, 0).toarray()
    actual_matrix = identity(10).toarray()

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_order_neg():
    """Ensures differential matrix fails for negative order."""
    with pytest.raises(ValueError):
        utils.difference_matrix(10, diff_order=-2)


def test_difference_matrix_order_over():
    """
    Tests the (n + 1)th order differential matrix against the actual representation.

    If n is the number of data points and the difference order is greater than n,
    then differential matrix should have a shape of (0, n) with 0 stored elements,
    following a similar logic as np.diff.

    """
    diff_matrix = utils.difference_matrix(10, 11).toarray()
    actual_matrix = np.empty(shape=(0, 10))

    assert_array_equal(diff_matrix, actual_matrix)


def test_difference_matrix_size_neg():
    """Ensures differential matrix fails for negative data size."""
    with pytest.raises(ValueError):
        utils.difference_matrix(-1)


@pytest.mark.parametrize('form', ('dia', 'csc', 'csr'))
def test_difference_matrix_formats(form):
    """
    Ensures that the sparse format is correctly passed to the constructor.

    Tests both 0-order and 2-order, since 0-order uses a different constructor.
    """
    assert utils.difference_matrix(10, 2, form).format == form
    assert utils.difference_matrix(10, 0, form).format == form


def test_changing_pentapy_solver():
    """Ensures a change to utils.PENTAPY_SOLVER is communicated by _pentapy_solver."""
    original_solver = utils.PENTAPY_SOLVER
    try:
        for solver in range(5):
            utils.PENTAPY_SOLVER = solver
            assert utils._pentapy_solver() == solver
    finally:
        utils.PENTAPY_SOLVER = original_solver


@pytest.mark.parametrize('kernel_size', (1, 10, 31, 1000, 2000, 4000))
@pytest.mark.parametrize('pad_mode', ('reflect', 'extrapolate'))
@pytest.mark.parametrize('list_input', (False, True))
def test_padded_convolve(kernel_size, pad_mode, list_input, data_fixture):
    """
    Ensures the output of the padded convolution is the same size as the input data.

    Notes
    -----
    `data_fixture` has 1000 data points, so test kernels with size less than, equal to,
    and greater than that size.

    """
    # make a simple uniform window kernel
    kernel = np.ones(kernel_size) / kernel_size
    _, data = data_fixture
    if list_input:
        input_data = data.tolist()
    else:
        input_data = data
    conv_output = utils.padded_convolve(input_data, kernel, pad_mode)

    assert isinstance(conv_output, np.ndarray)
    assert data.shape == conv_output.shape


def test_padded_convolve_empty_kernel():
    """Ensures convolving with an empty kernel fails."""
    with pytest.raises(ValueError):
        utils.padded_convolve(np.arange(10), np.array([]))


@pytest.mark.parametrize(
    'pad_mode', ('reflect', 'REFLECT', 'extrapolate', 'edge', 'constant')
)
@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 2000, 4000))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges(pad_mode, pad_length, list_input, data_fixture):
    """Tests various inputs for utils.pad_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    if pad_mode.lower() != 'extrapolate':
        expected_output = np.pad(data, pad_length, pad_mode.lower())
    else:
        expected_output = None

    output = utils.pad_edges(data, pad_length, pad_mode)
    assert isinstance(output, np.ndarray)
    assert len(output) == len(data) + 2 * pad_length

    if expected_output is not None:
        assert_allclose(output, expected_output)


@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 2000, 4000))
@pytest.mark.parametrize('extrapolate_window', (None, 1, 2, 10, 1001))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges_extrapolate(pad_length, list_input, extrapolate_window, data_fixture):
    """Ensures extrapolation works for utils.pad_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    output = utils.pad_edges(data, pad_length, 'extrapolate', extrapolate_window)
    assert isinstance(output, np.ndarray)
    assert len(output) == len(data) + 2 * pad_length


def test_pad_edges_extrapolate_zero_window():
    """Ensures an extrapolate_window of 0 raises an exception."""
    with pytest.raises(ValueError):
        utils.pad_edges(np.arange(10), 10, extrapolate_window=0)


@pytest.mark.parametrize('pad_mode', ('reflect', 'extrapolate'))
def test_pad_edges_negative_pad_length(pad_mode, data_fixture):
    """Ensures a negative pad length raises an exception."""
    with pytest.raises(ValueError):
        utils.pad_edges(data_fixture[1], -5, pad_mode)


@pytest.mark.parametrize('pad_mode', ('reflect', 'extrapolate'))
def test_get_edges_negative_pad_length(pad_mode, data_fixture):
    """Ensures a negative pad length raises an exception."""
    with pytest.raises(ValueError):
        utils._get_edges(data_fixture[1], -5, pad_mode)


@pytest.mark.parametrize(
    'pad_mode', ('reflect', 'REFLECT', 'extrapolate', 'edge', 'constant')
)
@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 2000, 4000))
@pytest.mark.parametrize('list_input', (False, True))
def test_get_edges(pad_mode, pad_length, list_input, data_fixture):
    """Tests various inputs for utils._get_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    if pad_length == 0:
        check_output = True
        expected_left = np.array([])
        expected_right = np.array([])
    elif pad_mode.lower() != 'extrapolate':
        check_output = True
        expected_left, _, expected_right = np.array_split(
            np.pad(data, pad_length, pad_mode.lower()), [pad_length, -pad_length]
        )
    else:
        check_output = False

    left, right = utils._get_edges(data, pad_length, pad_mode)
    assert isinstance(left, np.ndarray)
    assert len(left) == pad_length
    assert isinstance(right, np.ndarray)
    assert len(right) == pad_length

    if check_output:
        assert_allclose(left, expected_left)
        assert_allclose(right, expected_right)
