# -*- coding: utf-8 -*-
"""Tests for pybaselines._compat.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy
from scipy import integrate, sparse

from pybaselines import _compat


def test_prange():
    """
    Ensures that prange outputs the same as range.

    prange should work exactly as range, regardless of whether or not
    numba is installed.

    """
    start = 3
    stop = 9
    step = 2

    expected = list(range(start, stop, step))
    output = list(_compat.prange(start, stop, step))

    assert expected == output


def _add(a, b):
    """
    Simple function that adds two things.

    Will be decorated for testing.

    """
    output = a + b
    return output


# ignore the change of the default nopython value for numba's jit
@pytest.mark.filterwarnings("ignore:.*The 'nopython' keyword argument was not supplied.*")
def test_jit():
    """Ensures the jit decorator works regardless of whether or not numba is installed."""
    input_1 = 5
    input_2 = 6
    expected = input_1 + input_2
    output = _compat.jit(_add)(input_1, input_2)

    assert_array_equal(expected, output)


def test_jit_kwargs():
    """Ensure the jit decorator works with kwargs whether or not numba is installed."""
    input_1 = 5
    input_2 = 6
    expected = input_1 + input_2
    output = _compat.jit(_add, cache=True, nopython=True)(input_1, b=input_2)

    assert_array_equal(expected, output)


# ignore the change of the default nopython value for numba's jit
@pytest.mark.filterwarnings("ignore:.*The 'nopython' keyword argument was not supplied.*")
def test_jit_no_parentheses():
    """Ensure the jit decorator works with no parentheses whether or not numba is installed."""

    @_compat.jit
    def _add2(a, b):
        """
        Simple function that adds two things.

        For testing whether the jit decorator works without parentheses.

        """
        output = a + b
        return output

    input_1 = 5
    input_2 = 6
    expected = input_1 + input_2
    output = _add2(input_1, input_2)

    assert_array_equal(expected, output)


# ignore the change of the default nopython value for numba's jit
@pytest.mark.filterwarnings("ignore:.*The 'nopython' keyword argument was not supplied.*")
def test_jit_no_inputs():
    """Ensure the jit decorator works with no arguments whether or not numba is installed."""

    @_compat.jit()
    def _add3(a, b):
        """
        Simple function that adds two things.

        For testing whether the jit decorator works without any arguments.

        """
        output = a + b
        return output

    input_1 = 5
    input_2 = 6
    expected = input_1 + input_2
    output = _add3(input_1, input_2)

    assert_array_equal(expected, output)


def test_jit_signature():
    """Ensure the jit decorator works with a signature whether or not numba is installed."""

    @_compat.jit('int64(int64, int64)', nopython=True)
    def _add4(a, b):
        """
        Simple function that adds two things.

        For testing whether the jit decorator works with a function signature.

        """
        output = a + b
        return output

    input_1 = 5
    input_2 = 6
    expected = input_1 + input_2
    output = _add4(input_1, input_2)

    assert_array_equal(expected, output)


def test_trapezoid():
    """
    Ensures the trapezoid integration function within scipy is correctly used.

    Rather than checking equality with the expected function, just check that
    it works correctly.
    """
    data = [1., 2., 3.]
    output = _compat.trapezoid(data)
    assert_allclose(output, 4.0, rtol=0, atol=1e-14)

    if hasattr(integrate, 'trapezoid'):
        comparison_func = integrate.trapezoid
    else:
        comparison_func = integrate.trapz

    assert_allclose(output, comparison_func(data), rtol=0, atol=1e-14)


def _scipy_below_1_12():
    """
    Checks that the installed SciPy version is new enough to use sparse arrays.

    This check is wrapped into a function just in case it fails so that pybaselines
    can still be imported without error.

    Returns
    -------
    bool
        True if the installed SciPy version is below 1.12; False otherwise.

    Notes
    -----
    SciPy introduced its sparse arrays in version 1.8, but the interface and helper
    functions were not stable until version 1.12; a warning will be emitted in SciPy
    1.13 when using the matrix interface, so want to use the sparse array interface
    as early as possible.

    """
    try:
        _scipy_version = [int(val) for val in scipy.__version__.lstrip('v').split('.')[:2]]
    except Exception as e:
        # raise the exception so that version parsing can be changed if needed
        raise ValueError('Issue parsing SciPy version') from e

    return not (_scipy_version[0] > 1 or (_scipy_version[0] == 1 and _scipy_version[1] >= 12))


def test_use_sparse_arrays():
    """
    Ensures the SciPy version check works correctly.

    Use try-finally so that even if the test fails, the mocked values do
    not remain, which would cause subsequent tests to fail.
    """
    try:
        _compat._use_sparse_arrays.cache_clear()
        # sanity check that cache was cleared
        assert _compat._use_sparse_arrays.cache_info().currsize == 0
        with mock.patch.object(scipy, '__version__', '0.1'):
            assert not _compat._use_sparse_arrays()

        _compat._use_sparse_arrays.cache_clear()
        # sanity check that cache was cleared
        assert _compat._use_sparse_arrays.cache_info().currsize == 0
        with mock.patch.object(scipy, '__version__', '1.11'):
            assert not _compat._use_sparse_arrays()

        _compat._use_sparse_arrays.cache_clear()
        # sanity check that cache was cleared
        assert _compat._use_sparse_arrays.cache_info().currsize == 0
        with mock.patch.object(scipy, '__version__', '1.12'):
            assert _compat._use_sparse_arrays()

        _compat._use_sparse_arrays.cache_clear()
        # sanity check that cache was cleared
        assert _compat._use_sparse_arrays.cache_info().currsize == 0
        with mock.patch.object(scipy, '__version__', '2.0'):
            assert _compat._use_sparse_arrays()

        _compat._use_sparse_arrays.cache_clear()
        # sanity check that cache was cleared
        assert _compat._use_sparse_arrays.cache_info().currsize == 0
        # check that it returns True when an error reading the scipy version occurs
        with mock.patch.object(scipy, '__version__', 'abc'):
            assert _compat._use_sparse_arrays()
    finally:
        _compat._use_sparse_arrays.cache_clear()
    # ensure the cache is cleared so the correct value can be filled so the next call
    # to it is correct
    assert _compat._use_sparse_arrays.cache_info().currsize == 0


def test_np_ge_2():
    """
    Ensures the NumPy version check works correctly.

    Use try-finally so that even if the test fails, the mocked values do
    not remain, which would cause subsequent tests to fail.
    """
    # check that the version parsing within np_gt_2 is safe
    try:
        numpy_ge_2 = int(np.__version__.lstrip('v').split('.')[0]) >= 2
    except Exception as e:
        # raise the exception so that version parsing can be changed if needed
        raise ValueError('Issue parsing NumPy version') from e
    else:
        assert _compat._np_ge_2() == numpy_ge_2

    try:
        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        with mock.patch.object(np, '__version__', '0.1'):
            assert not _compat._np_ge_2()

        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        with mock.patch.object(np, '__version__', '1.11'):
            assert not _compat._np_ge_2()

        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        with mock.patch.object(np, '__version__', '2.0'):
            assert _compat._np_ge_2()

        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        with mock.patch.object(np, '__version__', '2.1'):
            assert _compat._np_ge_2()

        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        with mock.patch.object(np, '__version__', '3.1.0'):
            assert _compat._np_ge_2()

        _compat._np_ge_2.cache_clear()
        # sanity check that cache was cleared
        assert _compat._np_ge_2.cache_info().currsize == 0
        # check that it returns True when an error reading the scipy version occurs
        with mock.patch.object(np, '__version__', 'abc'):
            assert _compat._np_ge_2()
    finally:
        _compat._np_ge_2.cache_clear()
    # ensure the cache is cleared so the correct value can be filled so the next call
    # to it is correct
    assert _compat._np_ge_2.cache_info().currsize == 0


@pytest.mark.parametrize('dtype', (float, int))
def test_dia_object(dtype):
    """Ensures the compatibilty for dia_matrix and dia_array works as intended."""
    data = np.array([
        [1, 2, 0],
        [4, 5, 6],
        [0, 8, 9]
    ])
    offsets = [-1, 0, 1]
    output = _compat.dia_object((data, offsets), shape=(3, 3), dtype=dtype)

    expected_output = np.array([
        [4, 8, 0],
        [1, 5, 9],
        [0, 2, 6]
    ])

    assert output.dtype == dtype
    assert sparse.issparse(output)
    assert output.format == 'dia'
    assert_allclose(output.toarray(), expected_output, rtol=0, atol=1e-14)
    if _scipy_below_1_12():
        assert sparse.isspmatrix(output)
    else:
        assert not sparse.isspmatrix(output)


@pytest.mark.parametrize('dtype', (float, int))
def test_csr_object(dtype):
    """Ensures the compatibilty for csr_matrix and csr_array works as intended."""
    row = np.array([0, 1, 1, 2])
    col = np.array([0, 0, 2, 0])
    data = np.array([3, 5, 7, 9])
    output = _compat.csr_object((data, (row, col)), shape=(3, 3), dtype=dtype)

    expected_output = np.array([
        [3, 0, 0],
        [5, 0, 7],
        [9, 0, 0]
    ])

    assert output.dtype == dtype
    assert sparse.issparse(output)
    assert output.format == 'csr'
    assert_allclose(output.toarray(), expected_output, rtol=0, atol=1e-14)
    if _scipy_below_1_12():
        assert sparse.isspmatrix(output)
    else:
        assert not sparse.isspmatrix(output)


@pytest.mark.parametrize('sparse_format', ('csc', 'csr', 'dia'))
@pytest.mark.parametrize('size', (1, 3, 6))
def test_identity(size, sparse_format):
    """Ensures the sparse identity function works correctly."""
    output = _compat.identity(size, format=sparse_format)

    assert sparse.issparse(output)
    assert output.format == sparse_format
    assert_allclose(output.toarray(), np.eye(size), rtol=0, atol=1e-14)
    if _scipy_below_1_12():
        assert sparse.isspmatrix(output)
    else:
        assert not sparse.isspmatrix(output)


@pytest.mark.parametrize('dtype', (float, int))
@pytest.mark.parametrize('sparse_format', ('csc', 'csr', 'dia'))
def test_diags(sparse_format, dtype):
    """Ensures the sparse diags function works as intended."""
    data = [-1, 2, 1]
    offsets = [-1, 0, 1]
    output = _compat.diags(
        data, offsets=offsets, shape=(3, 3), dtype=dtype, format=sparse_format
    )

    expected_output = np.array([
        [2, 1, 0],
        [-1, 2, 1],
        [0, -1, 2]
    ])

    assert output.dtype == dtype
    assert sparse.issparse(output)
    assert output.format == sparse_format
    assert_allclose(output.toarray(), expected_output, rtol=0, atol=1e-14)
    if _scipy_below_1_12():
        assert sparse.isspmatrix(output)
    else:
        assert not sparse.isspmatrix(output)


def test_allow_1d_slice():
    """Uses version checking rather than brute force to ensure sparse slicing is available.

    The actual implementation in pybaselines directly checks if 1d slicing can be done on
    sparse matrices, which should be slightly more robust than a simple version check, but
    they should match regardless.

    """
    try:
        _scipy_version = [int(val) for val in scipy.__version__.lstrip('v').split('.')[:2]]
    except Exception as e:
        # raise the exception so that version parsing can be changed if needed
        raise ValueError('Issue parsing SciPy version') from e

    # sparse 1d slicing was first available in version 1.15.0
    expected = (_scipy_version[0] > 1 or (_scipy_version[0] == 1 and _scipy_version[1] >= 15))
    output = _compat._allows_1d_slice()

    assert expected == output


@pytest.mark.parametrize('sparse_format', ('csc', 'csr', 'dia'))
def test_sparse_col_index(sparse_format):
    """Ensures sparse matrix column indexing works as expected."""
    matrix = np.arange(20, dtype=float).reshape(5, 4)
    sparse_matrix = _compat.csr_object(matrix).asformat(sparse_format)

    expected_shape = (matrix.shape[0],)
    for col_index in range(matrix.shape[1]):
        expected_col = matrix[:, col_index]
        output = _compat._sparse_col_index(sparse_matrix, col_index)

        assert_allclose(output, expected_col, rtol=1e-15, atol=1e-15)
        assert output.shape == expected_shape
        assert isinstance(output, np.ndarray)
