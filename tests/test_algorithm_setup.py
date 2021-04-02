# -*- coding: utf-8 -*-
"""Tests for pybaselines._algorithm_setup.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pybaselines import _algorithm_setup


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, 5))
def test_difference_matrix(diff_order):
    """Tests all covered differential matrices."""
    diff_matrix = _algorithm_setup.difference_matrix(10, diff_order).toarray()
    numpy_diff = np.diff(np.eye(10), diff_order).T

    assert_array_equal(diff_matrix, numpy_diff)


def test_difference_matrix_order2():
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


def test_difference_matrix_fail():
    """Ensures differential matrix fails for non-covered order."""
    with pytest.raises(ValueError):
        _algorithm_setup.difference_matrix(10, diff_order=10)
