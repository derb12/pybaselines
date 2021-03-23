# -*- coding: utf-8 -*-
"""Tests for pybaselines.window.

@author: Donald Erb
Created on March 20, 2021

"""

from pybaselines import window

from .conftest import get_data, AlgorithmTester


class TestNoiseMedian(AlgorithmTester):
    """Class for testing noise median baseline."""

    func = window.noise_median

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y, 15)

    def test_output(self):
        super()._test_output(self.y, self.y, 15)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))


class TestSNIP(AlgorithmTester):
    """Class for testing snip baseline."""

    func = window.snip

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y, 15)

    def test_output(self):
        super()._test_output(self.y, self.y, 15)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))
