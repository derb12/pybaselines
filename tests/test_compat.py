# -*- coding: utf-8 -*-
"""Tests for pybaselines._compat.

@author: Donald Erb
Created on March 20, 2021

"""

from numpy.testing import assert_array_equal
import pytest

from pybaselines import _compat

from .conftest import has_pentapy, no_pentapy


@has_pentapy
def test_pentapy_installed():
    """Ensure proper setup when pentapy is installed."""
    assert _compat._HAS_PENTAPY


@no_pentapy
def test_pentapy_not_installed():
    """Ensure proper setup when pentapy is not installed."""
    assert not _compat._HAS_PENTAPY

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
    # boundscheck is used since it has no effect on the result
    output = _compat.jit(_add, boundscheck=True)(input_1, b=input_2)

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
