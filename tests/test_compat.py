# -*- coding: utf-8 -*-
"""Tests for pybaselines._compat.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import _compat

from .conftest import _HAS_PENTAPY


def test_pentapy_installation():
    """Ensure proper setup with pentapy."""
    assert _compat._HAS_PENTAPY == _HAS_PENTAPY
    if not _HAS_PENTAPY:
        with pytest.raises(NotImplementedError):
            _compat._pentapy_solve()


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
    output = _compat.jit(_add, cache=True)(input_1, b=input_2)

    assert_array_equal(expected, output)


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

    @_compat.jit('int64(int64, int64)')
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


def test_norm():
    """
    Ensures _compat.norm acts as intended.

    Several functions depend on the fact that norm(non-finite value) is non-finite,
    so also test that behavior.

    """
    test_array = np.ones(5)

    assert_allclose(_compat.norm(test_array), np.linalg.norm(test_array), 1e-14)

    test_array[0] = np.nan
    assert np.isnan(_compat.norm(test_array))

    test_array[0] = np.inf
    assert np.isinf(_compat.norm(test_array))
