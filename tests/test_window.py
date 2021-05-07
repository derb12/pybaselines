# -*- coding: utf-8 -*-
"""Tests for pybaselines.window.

@author: Donald Erb
Created on March 20, 2021

"""

import pytest

from pybaselines import window

from .conftest import AlgorithmTester, get_data


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

    @pytest.mark.parametrize('max_half_window', (15, [15], [12, 20]))
    def test_max_half_window_inputs(self, max_half_window):
        super()._test_output(self.y, self.y, max_half_window)

    @pytest.mark.parametrize('filter_order', (2, 4, 6, 8))
    def test_filter_orders(self, filter_order):
        super()._test_output(self.y, self.y, 15, filter_order=filter_order)

    def test_wrong_filter_order(self):
        with pytest.raises(ValueError):
            super()._call_func(self.y, 15, filter_order=1)

    @pytest.mark.parametrize('max_half_window', (1000000, [1000000], [1000000, 1000000]))
    def test_too_large_max_half_window(self, max_half_window):
        with pytest.warns(UserWarning):
            super()._call_func(self.y, max_half_window)


class TestSwima(AlgorithmTester):
    """Class for testing swima baseline."""

    func = window.swima

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
