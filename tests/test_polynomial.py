# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

import pytest

from pybaselines import polynomial

from .conftest import get_data, AlgorithmTester


class TestPoly(AlgorithmTester):
    """Class for testing regular polynomial baseline."""

    func = polynomial.poly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestModPoly(AlgorithmTester):
    """Class for testing ModPoly baseline."""

    func = polynomial.modpoly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIModPoly(AlgorithmTester):
    """Class for testing IModPoly baseline."""

    func = polynomial.imodpoly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestPenalizedPoly(AlgorithmTester):
    """Class for testing penalized_poly baseline."""

    func = polynomial.penalized_poly

    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'symmetric_truncated_quadratic',
            'a_truncated_quadratic',  # test that 'a' and 's' work as well
            's_truncated_quadratic',
            'asymmetric_huber',
            'symmetric_huber',
            'asymmetric_indec',
            'symmetric_indec'
        )
    )
    def test_unchanged_data(self, data_fixture, cost_function):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber'))
    def test_unknown_cost_function_prefix_fails(self, data_fixture, cost_function):
        x, y = get_data()
        with pytest.raises(ValueError):
            super()._test_unchanged_data(data_fixture, y, x, y, x, cost_function=cost_function)

    def test_unknown_cost_function_fails(self, data_fixture):
        x, y = get_data()
        with pytest.raises(KeyError):
            super()._test_unchanged_data(data_fixture, y, x, y, x, cost_function='a_hub')

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestLoess(AlgorithmTester):
    """Class for testing LOESS baseline."""

    func = polynomial.loess

    @pytest.mark.parametrize('use_threshold', (False, True))
    def test_unchanged_data(self, data_fixture, use_threshold):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, use_threshold=use_threshold)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
