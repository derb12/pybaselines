# -*- coding: utf-8 -*-
"""Tests for pybaselines._validation.

@author: Donald Erb
Created on Dec. 11, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import dia_matrix, identity
from scipy.sparse.linalg import spsolve

from pybaselines import _validation, utils


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

    output, was_scalar = _validation._check_scalar(input_data, desired_length)

    assert was_scalar
    assert isinstance(output, np.ndarray)
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
