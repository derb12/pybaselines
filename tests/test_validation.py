# -*- coding: utf-8 -*-
"""Tests for pybaselines._validation.

@author: Donald Erb
Created on Dec. 11, 2021

"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pybaselines import _validation


@pytest.mark.parametrize('array_enum', (0, 1))
def test_yx_arrays_output_array(small_data, array_enum):
    """Ensures output y and x are always numpy arrays and that x is not scaled."""
    if array_enum == 1:
        small_data = small_data.tolist()
    x_data = small_data.copy()
    y, x = _validation._yx_arrays(small_data, x_data)

    actual_array = np.asarray(small_data)

    assert isinstance(y, np.ndarray)
    assert_array_equal(y, actual_array)
    assert isinstance(x, np.ndarray)
    assert_array_equal(x, actual_array)


def test_yx_arrays_no_x(small_data):
    """Ensures an x array is created if None is input."""
    y, x = _validation._yx_arrays(small_data)

    assert isinstance(x, np.ndarray)
    assert_array_equal(x, np.linspace(-1., 1., y.shape[0]))


@pytest.mark.parametrize('ndim', (0, 1, 2))
def test_check_array_shape_and_dtype(ndim):
    """Ensures _check_array works as intended for the shape and dtype handling."""
    input_size = 10
    if ndim == 0:
        data = np.ones(input_size)
    else:
        data = np.ones((input_size, ndim))

    # ensure it works with 0d and 1d data and errors with 2d if ensure_1d is True
    if ndim == 2:
        with pytest.raises(ValueError):
            _validation._check_array(data, ensure_1d=True)
    else:
        for dtype in (int, float, np.float64, np.int64):
            output = _validation._check_array(data, dtype=dtype, ensure_1d=True)
            assert output.dtype == dtype
            assert output.ndim == 1
            assert output.shape == (input_size * max(ndim, 1),)

    # now just test conversion to array
    list_data = data.tolist()
    for dtype in (int, float, np.float64, np.int64):
        output = _validation._check_array(list_data, dtype=dtype, ensure_1d=False)
        assert output.dtype == dtype
        assert output.ndim == data.ndim
        assert output.shape == data.shape


@pytest.mark.parametrize('fill_scalar', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
@pytest.mark.parametrize('nested_input', (True, False))
def test_check_scalar_scalar_input(fill_scalar, list_input, nested_input):
    """Ensures _check_scalar works with scalar values."""
    input_data = 5
    desired_length = 10
    if fill_scalar:
        desired_output = np.full(desired_length, input_data)
    else:
        desired_output = np.asarray(input_data)
    if nested_input:
        input_data = [input_data]
    if list_input:
        input_data = [input_data]

    output, was_scalar = _validation._check_scalar(
        input_data, desired_length, fill_scalar=fill_scalar
    )

    assert was_scalar
    if fill_scalar:
        assert isinstance(output, np.ndarray)
    else:
        assert isinstance(output, np.integer)
    assert_array_equal(output, desired_output)


@pytest.mark.parametrize('fit_desired_length', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
@pytest.mark.parametrize('nested_input', (True, False))
def test_check_scalar_array_input(fit_desired_length, list_input, nested_input):
    """Ensures _check_scalar works with array-like inputs."""
    desired_length = 20
    fill_value = 5
    if fit_desired_length:
        input_data = np.full(desired_length, fill_value)
    else:
        input_data = np.full(desired_length - 1, fill_value)

    if nested_input:
        input_data = input_data.reshape(-1, 1)
    if list_input:
        input_data = input_data.tolist()

    if fit_desired_length:
        output, was_scalar = _validation._check_scalar(input_data, desired_length)

        assert not was_scalar
        assert isinstance(output, np.ndarray)
        assert_array_equal(output, np.asarray(input_data).reshape(-1))
    else:
        with pytest.raises(ValueError):
            _validation._check_scalar(input_data, desired_length)


def test_check_scalar_asarray_kwargs():
    """Ensures kwargs are passed to np.asarray by _check_scalar."""
    for dtype in (int, float, np.float64, np.int64):
        output, _ = _validation._check_scalar(20, 1, dtype=dtype)
        assert output.dtype == dtype

        output, _ = _validation._check_scalar(20, 10, True, dtype=dtype)
        assert output.dtype == dtype

        output, _ = _validation._check_scalar([20], 1, dtype=dtype)
        assert output.dtype == dtype

        output, _ = _validation._check_scalar(np.array([1, 2, 3]), 3, dtype=dtype)
        assert output.dtype == dtype


@pytest.mark.parametrize('coerce_0d', (False, True))
def test_check_scalar_coerce_0d(coerce_0d):
    """Ensures coerce_0d keyword maintains array when False."""
    data = [1]
    output, is_scalar = _validation._check_scalar(data, desired_length=None, coerce_0d=coerce_0d)
    if coerce_0d:
        assert output == 1
        assert is_scalar
    else:
        assert_array_equal(output, np.array([1]))
        assert output.shape == (1,)
        assert not is_scalar


def test_check_scalar_length_none():
    """Ensures a desired_length of None skips the length check."""
    for data in (0, [0]):
        output, _ = _validation._check_scalar(data, desired_length=None)
        assert output == 0

    for data in ([0, 1, 2], [[0, 1], [2, 3]]):
        output, _ = _validation._check_scalar(data, desired_length=None)
        assert output.shape == (np.array(data).flatten().size,)
        with pytest.raises(ValueError):
            _validation._check_scalar(data, desired_length=10000)


@pytest.mark.parametrize('lam', (5, [5], (5,), [[5]], np.array(5), np.array([5]), np.array([[5]])))
def test_check_lam(lam):
    """Ensures scalar lam values are correctly processed."""
    output_lam = _validation._check_lam(lam)
    assert output_lam == 5


def test_check_lam_failures():
    """Ensures array-like values or values < or <= 0 fail."""
    # fails due to array of values
    with pytest.raises(ValueError):
        _validation._check_lam([5, 10])

    # fails for lam <= 0 when allow_zero is False
    for lam in range(-5, 1):
        with pytest.raises(ValueError):
            _validation._check_lam(lam)

    # test that is allows zero if allow_zero is True
    _validation._check_lam(0, True)
    for lam in range(-5, 0):
        with pytest.raises(ValueError):
            _validation._check_lam(lam, True)


@pytest.mark.parametrize(
    'half_window', (5, 5.0, [5], (5,), [[5]], np.array(5), np.array([5]), np.array([[5]]))
)
def test_check_half_window(half_window):
    """Ensures scalar half-window values are correctly processed."""
    output_half_window = _validation._check_half_window(half_window)
    assert output_half_window == 5
    assert isinstance(output_half_window, np.intp)


def test_check_half_window_failures():
    """Ensures array-like values, non-integer values, or values < or <= 0 fail."""
    # fails due to array of values
    with pytest.raises(ValueError):
        _validation._check_half_window([5, 10])

    # fails for half_window <= 0 when allow_zero is False
    for half_window in range(-5, 1):
        with pytest.raises(ValueError):
            _validation._check_half_window(half_window)

    # test that is allows zero if allow_zero is True
    _validation._check_half_window(0, True)
    for half_window in range(-5, 0):
        with pytest.raises(ValueError):
            _validation._check_half_window(half_window, True)

    # fails due to non-integer input
    with pytest.raises(TypeError):
        _validation._check_half_window(5.01)


@pytest.mark.parametrize('list_input', (True, False))
def test_check_array_dtype(small_data, list_input):
    """Ensures dtype is correctly transferred in _check_array."""
    input_dtype = small_data.dtype
    if list_input:
        small_data = small_data.tolist()
    for dtype in (None, float, int, np.float64, np.intp):
        output = _validation._check_array(small_data, dtype=dtype)
        if dtype is None:
            assert output.dtype == input_dtype
        else:
            assert output.dtype == dtype


def test_check_array_ensure_1d():
    """Tests all valid inputs for 1d arrays."""
    original_shape = (10, 1)
    desired_shape = (10,)
    array = np.ones(original_shape)

    output = _validation._check_array(array, ensure_1d=True)
    assert output.shape == desired_shape
    assert array.shape == original_shape  # ensure it did not modify the input array

    output = _validation._check_array(array.T, ensure_1d=True)
    assert output.shape == desired_shape

    output = _validation._check_array(array.ravel(), ensure_1d=True)
    assert output.shape == desired_shape

    # also ensure the shape is ignored when ensure_1d is False
    output = _validation._check_array(array.reshape(-1, 2), ensure_1d=False)
    assert output.shape == array.reshape(-1, 2).shape


def test_check_array_ensure_1d_fails():
    """Ensure an exception is raised when the input is not 1d and ensure_1d is True."""
    array = np.ones((15, 2))
    with pytest.raises(ValueError):
        _validation._check_array(array, ensure_1d=True)


def test_check_array_check_finite():
    """Ensures the finite check is correct."""
    array = np.ones(5)
    for fill_value in (np.nan, np.inf, -np.inf):
        array[0] = fill_value
        _validation._check_array(array, check_finite=False)
        with pytest.raises(ValueError):
            _validation._check_array(array, check_finite=True)


def test_check_array_order():
    """Ensures the array order is correctly set in _check_array."""
    array = np.ones((5, 2))
    intial_c_contiguous = array.flags['C_CONTIGUOUS']
    intial_f_contiguous = array.flags['F_CONTIGUOUS']

    output = _validation._check_array(array, order=None, ensure_1d=False)
    assert output.flags['C_CONTIGUOUS'] == intial_c_contiguous
    assert output.flags['F_CONTIGUOUS'] == intial_f_contiguous

    output = _validation._check_array(array, order='C', ensure_1d=False)
    assert output.flags['C_CONTIGUOUS']

    output = _validation._check_array(array, order='F', ensure_1d=False)
    assert output.flags['F_CONTIGUOUS']


@pytest.mark.parametrize('list_input', (True, False))
def test_check_sized_array_dtype(small_data, list_input):
    """Ensures dtype is correctly transferred in _check_sized_array."""
    input_dtype = small_data.dtype
    length = small_data.shape[0]
    if list_input:
        small_data = small_data.tolist()
    for dtype in (None, float, int, np.float64, np.intp):
        output = _validation._check_sized_array(small_data, length, dtype=dtype)
        if dtype is None:
            assert output.dtype == input_dtype
        else:
            assert output.dtype == dtype


def test_check_sized_array_ensure_1d():
    """Tests all valid inputs for 1d arrays."""
    original_shape = (10, 1)
    desired_shape = (10,)
    array = np.ones(original_shape)

    output = _validation._check_sized_array(array, original_shape[0], ensure_1d=True)
    assert output.shape == desired_shape
    assert array.shape == original_shape  # ensure it did not modify the input array

    output = _validation._check_sized_array(array.T, original_shape[0], ensure_1d=True)
    assert output.shape == desired_shape

    output = _validation._check_sized_array(array.ravel(), original_shape[0], ensure_1d=True)
    assert output.shape == desired_shape

    # also ensure the shape is ignored when ensure_1d is False
    output = _validation._check_sized_array(
        array.reshape(-1, 2), array.reshape(-1, 2).shape[-1], ensure_1d=False
    )
    assert output.shape == array.reshape(-1, 2).shape

    # also ensure shape check works without changing to 1d
    output = _validation._check_sized_array(array.T, original_shape[0], ensure_1d=False)
    assert output.shape == original_shape[::-1]


def test_check_sized_array_ensure_1d_fails():
    """Ensure an exception is raised when the input is not 1d and ensure_1d is True."""
    array = np.ones((15, 2))
    with pytest.raises(ValueError):
        _validation._check_sized_array(array, array.shape[0], ensure_1d=True)


def test_check_sized_array_check_finite():
    """Ensures the finite check is correct."""
    array = np.ones(5)
    for fill_value in (np.nan, np.inf, -np.inf):
        array[0] = fill_value
        _validation._check_sized_array(array, array.shape[0], check_finite=False)
        with pytest.raises(ValueError):
            _validation._check_sized_array(array, array.shape[0], check_finite=True)


def test_check_sized_array_order():
    """Ensures the array order is correctly set in _check_sized_array."""
    array = np.ones((5, 2))
    length = array.shape[-1]
    intial_c_contiguous = array.flags['C_CONTIGUOUS']
    intial_f_contiguous = array.flags['F_CONTIGUOUS']

    output = _validation._check_sized_array(array, length, order=None, ensure_1d=False)
    assert output.flags['C_CONTIGUOUS'] == intial_c_contiguous
    assert output.flags['F_CONTIGUOUS'] == intial_f_contiguous

    output = _validation._check_sized_array(array, length, order='C', ensure_1d=False)
    assert output.flags['C_CONTIGUOUS']

    output = _validation._check_sized_array(array, length, order='F', ensure_1d=False)
    assert output.flags['F_CONTIGUOUS']


def test_check_sized_array_axis():
    """Ensures the axis kwarg works for _check_sized_array."""
    shape = (5, 2)
    array = np.ones(shape)

    _validation._check_sized_array(array, shape[0], axis=0, ensure_1d=False)
    _validation._check_sized_array(array, shape[1], axis=1, ensure_1d=False)

    # ensure failure with mismatched axis and length
    with pytest.raises(ValueError):
        _validation._check_sized_array(array, shape[0], axis=1, ensure_1d=False)
    with pytest.raises(ValueError):
        _validation._check_sized_array(array, shape[1], axis=0, ensure_1d=False)


def test_check_sized_array_name():
    """Ensures the name kwarg is passed to the raised exception for _check_sized_array."""
    length = 5
    array = np.ones(length)

    for name in ('weights', 'x-data', 'data'):
        with pytest.raises(ValueError, match=f'length mismatch for {name}'):
            _validation._check_sized_array(array, length + 1, name=name)


@pytest.mark.parametrize('list_input', (True, False))
def test_optional_array_output(small_data, list_input):
    """Ensures output y and x are always numpy arrays and that x is not scaled."""
    if list_input == 1:
        small_data = small_data.tolist()
    output = _validation._check_optional_array(len(small_data), small_data)

    actual_array = np.asarray(small_data)

    assert isinstance(output, np.ndarray)
    assert_array_equal(output, actual_array)


def test_optional_array_no_input():
    """Ensures an array of ones is created if None is input."""
    length = 10
    output = _validation._check_optional_array(length, None)

    assert isinstance(output, np.ndarray)
    assert_array_equal(output, np.ones(length))
