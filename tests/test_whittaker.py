# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

from pybaselines import whittaker

from .conftest import get_data, AlgorithmTester


class TestAsLS(AlgorithmTester):
    """Class for testing asls baseline."""

    func = whittaker.asls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIAsLS(AlgorithmTester):
    """Class for testing iasls baseline."""

    func = whittaker.iasls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAirPLS(AlgorithmTester):
    """Class for testing airpls baseline."""

    func = whittaker.airpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestArPLS(AlgorithmTester):
    """Class for testing arpls baseline."""

    func = whittaker.arpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestDrPLS(AlgorithmTester):
    """Class for testing drpls baseline."""

    func = whittaker.airpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIArPLS(AlgorithmTester):
    """Class for testing iarpls baseline."""

    func = whittaker.iarpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAsPLS(AlgorithmTester):
    """Class for testing aspls baseline."""

    func = whittaker.aspls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
