# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import whittaker

from .conftest import AlgorithmTester, get_data, has_pentapy


def test_shift_rows_2_diags():
    """Ensures rows are correctly shifted for a matrix with two off-diagonals on either side."""
    matrix = np.array([
        [1, 2, 9, 0, 0],
        [1, 2, 3, 4, 0],
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 8],
        [0, 0, 1, 2, 3]
    ])
    expected = np.array([
        [0, 0, 1, 2, 9],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 8, 0],
        [1, 2, 3, 0, 0]
    ])
    output = whittaker._shift_rows(matrix, 2)

    assert_array_equal(expected, output)
    # matrix should also be shifted since the changes are done in-place
    assert_array_equal(expected, matrix)


def test_shift_rows_1_diag():
    """Ensures rows are correctly shifted for a matrix with one off-diagonal on either side."""
    matrix = np.array([
        [1, 2, 3, 8, 0],
        [1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4],
    ])
    expected = np.array([
        [0, 1, 2, 3, 8],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 0],
    ])
    output = whittaker._shift_rows(matrix, 1)

    assert_array_equal(expected, output)
    # matrix should also be shifted since the changes are done in-place
    assert_array_equal(expected, matrix)


class TestAsLS(AlgorithmTester):
    """Class for testing asls baseline."""

    func = whittaker.asls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestIAsLS(AlgorithmTester):
    """Class for testing iasls baseline."""

    func = whittaker.iasls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, p=p)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestAirPLS(AlgorithmTester):
    """Class for testing airpls baseline."""

    func = whittaker.airpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e3, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestArPLS(AlgorithmTester):
    """Class for testing arpls baseline."""

    func = whittaker.arpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestDrPLS(AlgorithmTester):
    """Class for testing drpls baseline."""

    func = whittaker.drpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestIArPLS(AlgorithmTester):
    """Class for testing iarpls baseline."""

    func = whittaker.iarpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestAsPLS(AlgorithmTester):
    """Class for testing aspls baseline."""

    func = whittaker.aspls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(
            self.y, self.y, checked_keys=('weights', 'alpha', 'iterations', 'last_tol')
        )

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_wrong_alpha_shape(self):
        """Ensures that an exception is raised if input alpha and data are different shapes."""
        alpha = np.ones(self.y.shape[0] + 1)
        with pytest.raises(ValueError):
            self._call_func(self.y, alpha=alpha)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestPsalsa(AlgorithmTester):
    """Class for testing psalsa baseline."""

    func = whittaker.psalsa

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        self._test_output(self.y, self.y, checked_keys=('weights', 'iterations', 'last_tol'))

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)
