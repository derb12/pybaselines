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
