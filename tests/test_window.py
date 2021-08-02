# -*- coding: utf-8 -*-
"""Tests for pybaselines.window.

@author: Donald Erb
Created on March 20, 2021

"""

import pytest

from pybaselines import window
from pybaselines.utils import ParameterWarning

from .conftest import AlgorithmTester, get_data


class TestNoiseMedian(AlgorithmTester):
    """Class for testing noise median baseline."""

    func = window.noise_median

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y, 15)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, 15, checked_keys=())

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))


class TestSNIP(AlgorithmTester):
    """Class for testing snip baseline."""

    func = window.snip

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y, 15)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, 15, checked_keys=())

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))

    @pytest.mark.parametrize('max_half_window', (15, [15], [12, 20], (12, 15, 20)))
    def test_max_half_window_inputs(self, max_half_window):
        """Tests valid inputs for `max_half_window`."""
        self._test_output(self.y, self.y, max_half_window)

    @pytest.mark.parametrize('filter_order', (2, 4, 6, 8))
    def test_filter_orders(self, filter_order):
        """Ensures all filter orders work."""
        self._test_output(self.y, self.y, 15, filter_order=filter_order)

    def test_wrong_filter_order(self):
        """Ensures a non-covered filter order fails."""
        with pytest.raises(ValueError):
            self._call_func(self.y, 15, filter_order=1)

    @pytest.mark.parametrize('max_half_window', (1000000, [1000000], [1000000, 1000000]))
    def test_too_large_max_half_window(self, max_half_window):
        """Ensures a warning emitted when max_half_window is greater than (len(data) - 1) // 2."""
        with pytest.warns(ParameterWarning):
            self._call_func(self.y, max_half_window)


class TestSwima(AlgorithmTester):
    """Class for testing swima baseline."""

    func = window.swima

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window', 'converged'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
