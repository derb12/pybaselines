# -*- coding: utf-8 -*-
"""Tests for pybaselines.utils.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve

from pybaselines import _banded_utils, _spline_utils, utils
from pybaselines._compat import dia_object, diags, identity

from .conftest import gaussian


@pytest.fixture(scope='module')
def _x_data():
    """x-values for testing."""
    return np.linspace(-20, 20, 50)


@pytest.fixture(scope='module')
def _z_data():
    """z-values for testing."""
    return np.linspace(-10, 10, 30)


@pytest.mark.parametrize('sigma', [0.1, 1, 10])
@pytest.mark.parametrize('center', [-10, 0, 10])
@pytest.mark.parametrize('height', [0.1, 1, 10])
def test_gaussian(_x_data, height, center, sigma):
    """Ensures that gaussian function in pybaselines.utils is correct."""
    assert_allclose(
        utils.gaussian(_x_data, height, center, sigma),
        gaussian(_x_data, height, center, sigma), 1e-12, 1e-12
    )


@pytest.mark.parametrize('sigma', (0, -1))
def test_gaussian_non_positive_sigma(_x_data, sigma):
    """Ensures a sigma value not greater than 0 raises an exception."""
    with pytest.raises(ValueError):
        utils.gaussian(_x_data, sigma=sigma)


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
    assert_allclose(np.sum(kernel), 1)


def test_gaussian_kernel_0_windowsize(data_fixture):
    """
    Ensures the gaussian kernel with 0 window size gives an array of [1].

    Also ensures the convolution with the kernel gives the unchanged input.
    """
    kernel = utils.gaussian_kernel(0, 3)

    assert kernel.size == 1
    assert kernel.shape == (1,)
    assert_array_equal(kernel, 1)
    assert_allclose(np.sum(kernel), 1)

    x, y = data_fixture
    out = utils.padded_convolve(y, kernel)
    assert_array_equal(y, out)


@pytest.mark.parametrize('sigma_x', [0.1, 1, 10])
@pytest.mark.parametrize('center_x', [-10, 0, 10])
@pytest.mark.parametrize('sigma_z', [0.1, 1, 10])
@pytest.mark.parametrize('center_z', [-10, 0, 10])
@pytest.mark.parametrize('height', [0.1, 1, 10])
def test_gaussian2d(_x_data, _z_data, height, center_x, center_z, sigma_x, sigma_z):
    """Ensures that gaussian2d function in pybaselines.utils is correct."""
    X, Z = np.meshgrid(_x_data, _z_data)

    expected = height * gaussian(X, 1, center_x, sigma_x) * gaussian(Z, 1, center_z, sigma_z)
    assert_allclose(
        utils.gaussian2d(X, Z, height, center_x, center_z, sigma_x, sigma_z),
        expected, 1e-12, 1e-12
    )


def test_gaussian2d_1d_raises(_x_data, _z_data):
    """Ensures that gaussian2d function raises an error if the input is one dimensional."""
    X, Z = np.meshgrid(_x_data, _z_data)
    with pytest.raises(ValueError):
        utils.gaussian2d(_x_data, _z_data)
    with pytest.raises(ValueError):
        utils.gaussian2d(X, _z_data)
    with pytest.raises(ValueError):
        utils.gaussian2d(_x_data, Z)


@pytest.mark.parametrize('sign', (1, -1))
def test_relative_difference_scalar(sign):
    """Tests relative_difference to ensure it uses abs for scalars."""
    old = 3.0 * sign
    new = 4
    assert_allclose(utils.relative_difference(old, new), abs((old - new) / old))


def test_relative_difference_array():
    """Tests relative_difference to ensure it uses l2 norm for arrays."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    norm_ab = np.sqrt(((a - b)**2).sum())
    norm_a = np.sqrt(((a)**2).sum())

    assert_allclose(utils.relative_difference(a, b), norm_ab / norm_a)


def test_relative_difference_array_l1():
    """Tests `norm_order` keyword for relative_difference."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    norm_ab = np.abs(a - b).sum()
    norm_a = np.abs(a).sum()

    assert_allclose(utils.relative_difference(a, b, 1), norm_ab / norm_a)


def test_relative_difference_zero():
    """Ensures relative difference works when 0 is the denominator."""
    a = np.array([0, 0, 0])
    b = np.array([4, 5, 6])
    norm_ab = np.sqrt(((a - b)**2).sum())

    assert_allclose(utils.relative_difference(a, b), norm_ab / np.finfo(float).eps)


def test_interp_inplace():
    """Tests that _interp_inplace modified the input array inplace."""
    x = np.arange(10)
    y_actual = 2 + 5 * x

    y_calc = np.empty_like(y_actual)
    y_calc[0] = y_actual[0]
    y_calc[-1] = y_actual[-1]

    output = utils._interp_inplace(x, y_calc, y_calc[0], y_calc[-1])

    # output should be the same object as the input y array
    assert output is y_calc

    assert_allclose(y_calc, y_actual, 1e-12)


@pytest.mark.parametrize('scale', (1., 10., 0.557))
@pytest.mark.parametrize('num_coeffs', (1, 2, 5))
def test_poly_transform_matrix(scale, num_coeffs):
    """
    Tests the matrix that transforms polynomial coefficients from one domain to another.

    Only tests the simple cases where the offset is 0 since more complicated cases are
    handled by the _convert_coef and _convert_coef2d tests.
    """
    transform_matrix = np.eye(num_coeffs)
    for i in range(num_coeffs):
        transform_matrix[i, i] /= scale**i

    domain = np.array([-1, 1]) * scale
    calc_matrix = utils._poly_transform_matrix(num_coeffs, domain)

    assert_allclose(calc_matrix, transform_matrix, atol=1e-12, rtol=1e-14)


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


@pytest.mark.parametrize('x', (np.linspace(-1, 1, 50), np.linspace(-13.5, 11.6, 51)))
@pytest.mark.parametrize('z', (np.linspace(-1, 1, 50), np.linspace(-13.5, 11.6, 51)))
@pytest.mark.parametrize(
    'coef', (
        np.array([
            [1, 0],
            [1, 0]
        ]),
        np.array([
            [1, 1],
            [0, 0]
        ]),
        np.array([
            [1, 0.1, 0.3, -0.5],
            [1, 0.1, 0, 1],
            [0.2, 0, 1.5, -0.3]
        ]),
    )
)
def test_convert_coef2d(x, z, coef):
    """
    Checks that polynomial coefficients are correctly converted to the original domain.

    Notes on the tested x and z values: Data from [-1, 1] has an offset of 0 and a scale
    of 1, so the coefficients are unaffected, while the second set of values has an offset
    not equal to 0 and a scale not equal to 1 so should be a good test of whether the
    conversion is successful.

    """
    x_domain = np.polynomial.polyutils.getdomain(x)
    mapped_x = np.polynomial.polyutils.mapdomain(
        x, x_domain, np.array([-1., 1.])
    )
    z_domain = np.polynomial.polyutils.getdomain(z)
    mapped_z = np.polynomial.polyutils.mapdomain(
        z, z_domain, np.array([-1., 1.])
    )
    X, Z = np.meshgrid(x, z)
    y = np.zeros_like(x)
    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            y = y + coef[i, j] * X**i * Z**j
    y_flat = y.ravel()

    vandermonde = np.polynomial.polynomial.polyvander2d(
        *np.meshgrid(mapped_x, mapped_z),
        (coef.shape[0] - 1, coef.shape[1] - 1)
    ).reshape((-1, (coef.shape[0]) * (coef.shape[1])))

    calc_coef = np.linalg.pinv(vandermonde) @ (y_flat)
    calc_y = vandermonde @ calc_coef  # corresponds to mapped domain

    # sanity check; use slightly higher atol than other checks since
    # the fit can potentially be off by a bit
    assert_allclose(calc_y, y_flat, rtol=1e-10, atol=1e-6)

    converted_coef = utils._convert_coef2d(
        calc_coef, coef.shape[0] - 1, coef.shape[1] - 1, x_domain, z_domain
    )

    mapped_X, mapped_Z = np.meshgrid(mapped_x, mapped_z)
    mapped_polynomial = np.polynomial.polynomial.polyval2d(
        mapped_X, mapped_Z, calc_coef.reshape(coef.shape)
    )

    original_polynomial = np.polynomial.polynomial.polyval2d(X, Z, converted_coef)

    # sanity check that polyval2d recreates with the mapped coefficients
    assert_allclose(mapped_polynomial, calc_y.reshape(y.shape), rtol=1e-10, atol=1e-14)

    assert_allclose(original_polynomial, mapped_polynomial, rtol=1e-10, atol=1e-14)
    assert_allclose(converted_coef, coef, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('diff_order', (0, 1, 2, 3, 4, 5))
def test_difference_matrix(diff_order):
    """Tests common differential matrices."""
    diff_matrix = utils.difference_matrix(10, diff_order).toarray()
    numpy_diff = np.diff(np.eye(10), diff_order, axis=0)

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


def pad_func(array, pad_width, axis, kwargs):
    """A custom padding function for use with numpy.pad."""
    pad_val = kwargs.get('pad_val', 0)
    array[:pad_width[0]] = pad_val
    if pad_width[1] != 0:
        array[-pad_width[1]:] = pad_val


@pytest.mark.parametrize('kernel_size', (1, 10, 31, 1000, 4000))
@pytest.mark.parametrize('pad_mode', ('reflect', 'extrapolate', pad_func))
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
    'pad_mode', ('reflect', 'REFLECT', 'extrapolate', 'edge', 'constant', pad_func)
)
@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 4000))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges(pad_mode, pad_length, list_input, data_fixture):
    """Tests various inputs for utils.pad_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    if not callable(pad_mode):
        np_pad_mode = pad_mode.lower()
    else:
        np_pad_mode = pad_mode
    if np_pad_mode != 'extrapolate':
        expected_output = np.pad(data, pad_length, np_pad_mode)
    else:
        expected_output = None

    output = utils.pad_edges(data, pad_length, pad_mode)
    assert isinstance(output, np.ndarray)
    assert len(output) == len(data) + 2 * pad_length

    if expected_output is not None:
        assert_allclose(output, expected_output)


@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 4000))
@pytest.mark.parametrize('extrapolate_window', (None, 1, 2, 10, 1001, (10, 20), (1, 1)))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges_extrapolate(pad_length, list_input, extrapolate_window, data_fixture):
    """Ensures extrapolation works for utils.pad_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    output = utils.pad_edges(data, pad_length, 'extrapolate', extrapolate_window)
    assert isinstance(output, np.ndarray)
    assert len(output) == len(data) + 2 * pad_length


def test_pad_edges_extrapolate_windows():
    """Ensures the separate extrapolate windows are correctly interpreted."""
    input_array = np.zeros(50)
    input_array[-10:] = 1.
    extrapolate_windows = [40, 10]
    pad_len = 20
    output = utils.pad_edges(
        input_array, pad_len, mode='extrapolate', extrapolate_window=extrapolate_windows
    )

    assert_allclose(output[:pad_len], np.full(pad_len, 0.), 1e-14)
    assert_allclose(output[-pad_len:], np.full(pad_len, 1.), 1e-14)


@pytest.mark.parametrize('extrapolate_window', (0, -2, (0, 0), (5, 0), (5, -1)))
def test_pad_edges_extrapolate_zero_window(extrapolate_window):
    """Ensures an extrapolate_window <= 0 raises an exception."""
    with pytest.raises(ValueError):
        utils.pad_edges(
            np.arange(10), 10, mode='extrapolate', extrapolate_window=extrapolate_window
        )


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


def test_pad_edges_custom_pad_func():
    """Ensures pad_edges works with a callable padding function, same as numpy.pad."""
    input_array = np.arange(20)
    pad_val = 20
    pad_length = 10

    edge_array = np.full(pad_length, pad_val)
    expected_output = np.concatenate((edge_array, input_array, edge_array))

    actual_output = utils.pad_edges(input_array, pad_length, pad_func, pad_val=pad_val)

    assert_allclose(actual_output, expected_output, rtol=1e-12, atol=0)


def test_get_edges_custom_pad_func():
    """Ensures _get_edges works with a callable padding function, same as numpy.pad."""
    input_array = np.arange(20)
    pad_val = 20
    pad_length = 10

    expected_output = np.full(pad_length, pad_val)

    left, right = utils._get_edges(input_array, pad_length, pad_func, pad_val=pad_val)

    assert_array_equal(left, expected_output)
    assert_array_equal(right, expected_output)


@pytest.mark.parametrize(
    'pad_mode', ('reflect', 'REFLECT', 'extrapolate', 'edge', 'constant', pad_func)
)
@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 500, 1000, 4000))
@pytest.mark.parametrize('list_input', (False, True))
def test_get_edges(pad_mode, pad_length, list_input, data_fixture):
    """Tests various inputs for utils._get_edges."""
    _, data = data_fixture
    if list_input:
        data = data.tolist()

    if not callable(pad_mode):
        np_pad_mode = pad_mode.lower()
    else:
        np_pad_mode = pad_mode

    if pad_length == 0:
        check_output = True
        expected_left = np.array([])
        expected_right = np.array([])
    elif np_pad_mode != 'extrapolate':
        check_output = True
        expected_left, _, expected_right = np.array_split(
            np.pad(data, pad_length, np_pad_mode), [pad_length, -pad_length]
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


@pytest.mark.parametrize(
    'pad_mode', ('reflect', 'REFLECT', 'extrapolate', 'edge', 'constant', pad_func)
)
@pytest.mark.parametrize('pad_length', (1, 2, 20, 53))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges2d(pad_mode, pad_length, list_input, data_fixture2d):
    """Tests various inputs for utils.pad_edges2d."""
    *_, data = data_fixture2d
    data_shape = data.shape
    if list_input:
        data = data.tolist()

    if not callable(pad_mode):
        np_pad_mode = pad_mode.lower()
    else:
        np_pad_mode = pad_mode
    if np_pad_mode != 'extrapolate':
        expected_output = np.pad(data, pad_length, np_pad_mode)
    else:
        expected_output = None

    output = utils.pad_edges2d(data, pad_length, pad_mode)
    assert isinstance(output, np.ndarray)
    assert output.ndim == 2
    assert output.shape[0] == data_shape[0] + 2 * pad_length
    assert output.shape[1] == data_shape[1] + 2 * pad_length

    if expected_output is not None:
        assert_allclose(output, expected_output)


@pytest.mark.parametrize('pad_length', (0, 1, 2, 20, 53))
@pytest.mark.parametrize('extrapolate_window', (None, 1, 2, 10, 1001, (10, 20), (1, 1)))
@pytest.mark.parametrize('list_input', (False, True))
def test_pad_edges2d_extrapolate(pad_length, list_input, extrapolate_window, data_fixture2d):
    """Ensures extrapolation works for utils.pad_edges."""
    *_, data = data_fixture2d
    data_shape = data.shape
    if list_input:
        data = data.tolist()

    if np.less_equal(pad_length, 0).any():
        with pytest.raises(NotImplementedError):
            utils.pad_edges2d(data, pad_length, 'extrapolate', extrapolate_window)
    else:
        output = utils.pad_edges2d(data, pad_length, 'extrapolate', extrapolate_window)
        assert isinstance(output, np.ndarray)
        assert output.shape[0] == data_shape[0] + 2 * pad_length
        assert output.shape[1] == data_shape[1] + 2 * pad_length


def test_pad_edges2d_extrapolate_windows():
    """Ensures the separate extrapolate windows are correctly interpreted."""
    input_array = np.zeros(400).reshape(20, 20)
    input_array[-10:] = 1.
    extrapolate_windows = [5, 10]
    pad_len = 5
    output = utils.pad_edges2d(
        input_array, pad_len, mode='extrapolate', extrapolate_window=extrapolate_windows
    )

    assert_allclose(
        output[:pad_len, pad_len:-pad_len], np.full((pad_len, input_array.shape[1]), 0.), 1e-14
    )
    assert_allclose(
        output[-pad_len:, pad_len:-pad_len], np.full((pad_len, input_array.shape[1]), 1.), 1e-14
    )


@pytest.mark.parametrize('extrapolate_window', (0, -2, (0, 0), (5, 0), (5, -1)))
def test_pad_edges2d_extrapolate_zero_window(small_data2d, extrapolate_window):
    """Ensures an extrapolate_window <= 0 raises an exception."""
    with pytest.raises(ValueError):
        utils.pad_edges2d(
            small_data2d, 10, mode='extrapolate', extrapolate_window=extrapolate_window
        )


@pytest.mark.parametrize('pad_mode', ('reflect', 'extrapolate'))
def test_pad_edges2d_negative_pad_length(pad_mode, data_fixture2d):
    """Ensures a negative pad length raises an exception."""
    with pytest.raises(ValueError):
        utils.pad_edges2d(data_fixture2d[-1], -5, pad_mode)


def test_pad_edges2d_custom_pad_func():
    """Ensures pad_edges works with a callable padding function, same as numpy.pad."""
    input_array = np.arange(2000).reshape(50, 40)
    pad_val = 20
    pad_length = 10

    expected_output = np.empty(
        (input_array.shape[0] + 2 * pad_length, input_array.shape[1] + 2 * pad_length)
    )
    expected_output[:pad_length] = pad_val
    expected_output[-pad_length:] = pad_val
    expected_output[:, :pad_length] = pad_val
    expected_output[:, -pad_length:] = pad_val
    expected_output[pad_length:-pad_length, pad_length:-pad_length] = input_array

    actual_output = utils.pad_edges(input_array, pad_length, pad_func, pad_val=pad_val)

    assert_allclose(actual_output, expected_output, rtol=1e-12, atol=0)


@pytest.mark.parametrize('seed', (123, 98765))
def test_invert_sort(seed):
    """Ensures the inverted sort works."""
    values = np.random.default_rng(seed).normal(0, 10, 1000)
    sort_order = values.argsort(kind='mergesort')

    expected_inverted_sort = sort_order.argsort(kind='mergesort')
    inverted_order = utils._inverted_sort(sort_order)

    assert_array_equal(expected_inverted_sort, inverted_order)
    assert_array_equal(values, values[sort_order][inverted_order])


@pytest.mark.parametrize('needs_sorting', (True, False))
def test_determine_sorts(needs_sorting):
    """Ensures the sort and inverted sort determinations work."""
    data = np.linspace(-1, 1, 20)
    original_data = data.copy()
    if needs_sorting:
        data[5:10] = data[5:10][::-1]

    sort_order, inverted_order = utils._determine_sorts(data)
    if not needs_sorting:
        assert sort_order is None
        assert inverted_order is None
    else:
        assert_array_equal(data[sort_order], original_data)
        assert_array_equal(sort_order, data.argsort(kind='mergesort'))
        assert_array_equal(data[sort_order][inverted_order], data)


@pytest.mark.parametrize('two_d', (True, False))
def test_sort_array_none(two_d):
    """Tests the case where the sorting array is None, which should skip sorting."""
    data = np.linspace(-1, 1, 20)
    if two_d:
        data = data[None, :]

    assert_allclose(data, utils._sort_array(data, sort_order=None), atol=0, rtol=1e-14)


@pytest.mark.parametrize('two_d', (True, False))
def test_sort_array(two_d):
    """Ensures array sorting works with 1d arrays."""
    data = np.linspace(-1, 1, 20)
    reversed_data = data[::-1]
    sort_order = np.arange(len(data))[::-1]
    if two_d:
        data = np.array([data, data])
        reversed_data = np.array([reversed_data, reversed_data])

    assert_allclose(data, utils._sort_array(reversed_data, sort_order), atol=0, rtol=1e-14)


@pytest.mark.parametrize('three_d', (True, False))
def test_sort_array2d_none(three_d):
    """Tests the case where the sorting array is None, which should skip sorting."""
    data = np.linspace(-1, 1, 20).reshape(5, 4)
    if three_d:
        data = data[None, :]

    assert_allclose(data, utils._sort_array2d(data, sort_order=None), atol=0, rtol=1e-14)


@pytest.mark.parametrize('sort_x', (True, False, None))
@pytest.mark.parametrize('three_d', (True, False))
def test_sort_array2d(three_d, sort_x):
    """
    Ensures sorting for 2d data works.

    Each of the three `sort_x` cases corresponds to how _Algorithm2D will make its _sort_order
    attribute if given only x, only z, and both x and z, respectively.
    """
    x = np.linspace(-1, 1, 20)
    z = np.linspace(-2, 2, 30)
    x_sort_order = np.arange(len(x))
    z_sort_order = np.arange(len(z))

    X, Z = np.meshgrid(x, z)
    data = X + 2 * Z

    if sort_x is None:  # sort both x and z, so reverse both x and z
        x2 = x[::-1]
        x_sort_order = x_sort_order[::-1]
        z2 = z[::-1]
        z_sort_order = z_sort_order[::-1]
        sort_order = (z_sort_order[:, None], x_sort_order[None, :])
    elif sort_x:  # sort just x, so reverse just x
        x2 = x[::-1]
        x_sort_order = x_sort_order[::-1]
        z2 = z
        sort_order = (..., x_sort_order)
    else:  # sort just z, so reverse just z
        x2 = x
        z2 = z[::-1]
        z_sort_order = z_sort_order[::-1]
        sort_order = z_sort_order

    X2, Z2 = np.meshgrid(x2, z2)
    reversed_data = X2 + 2 * Z2
    if three_d:
        data = np.array([data, data])
        reversed_data = np.array([reversed_data, reversed_data])

    assert_allclose(data, utils._sort_array2d(reversed_data, sort_order), atol=0, rtol=1e-14)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
def test_whittaker_smooth(data_fixture, diff_order):
    """Ensures the Whittaker smoothing function performs correctly."""
    x, y = data_fixture
    lam = 1
    output = utils.whittaker_smooth(y, lam, diff_order)

    assert isinstance(output, np.ndarray)

    # construct the sparse solution and compare
    len_y = len(y)
    diff_matrix = utils.difference_matrix(len_y, diff_order, 'csc')
    penalty = lam * (diff_matrix.T @ diff_matrix)

    # solve the simple case for all weights are 1
    expected_output = spsolve(identity(len_y) + penalty, y)

    assert_allclose(output, expected_output, 1e-6)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (10, 100, 200))
@pytest.mark.parametrize('spline_degree', (1, 2, 3))
def test_pspline_smooth(data_fixture, diff_order, num_knots, spline_degree):
    """Ensures the Penalized Spline smoothing function performs correctly."""
    x, y = data_fixture
    lam = 1
    output, tck = utils.pspline_smooth(
        y, x, lam=lam, diff_order=diff_order, num_knots=num_knots, spline_degree=spline_degree
    )

    assert isinstance(output, np.ndarray)

    # construct the sparse solution and compare
    len_y = len(y)
    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty_matrix = dia_object(
        (_banded_utils.diff_penalty_diagonals(num_bases, diff_order, lower_only=False),
        np.arange(diff_order, -(diff_order + 1), -1)), shape=(num_bases, num_bases)
    ).tocsr()
    weights = diags(np.ones(len_y), format='csr')

    # solve the simple case for all weights are 1
    coeffs = spsolve(basis.T @ weights @ basis + lam * penalty_matrix, basis.T @ weights @ y)

    assert_allclose(basis @ coeffs, output, 1e-6)

    # ensure tck is the knots, coefficients, and spline degree
    assert len(tck) == 3

    # now recreate the spline with scipy's BSpline and ensure it is the same
    recreated_spline = BSpline(*tck)(x)

    assert_allclose(recreated_spline, output, rtol=1e-10)


@pytest.mark.parametrize('two_d', (True, False))
def test_optimize_window(small_data2d, two_d):
    """Ensures optimize_window has the correct outputs for the dimesions of the input."""
    data = small_data2d
    if not two_d:
        data = data.flatten()

    output = utils.optimize_window(data)
    if two_d:
        assert output.shape == (2,)
        assert isinstance(output, np.ndarray)
    else:
        assert isinstance(output, int)
