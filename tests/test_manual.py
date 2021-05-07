# -*- coding: utf-8 -*-
"""Tests for pybaselines.manual.

@author: Donald Erb
Created on March 20, 2021

"""

from pybaselines import manual

from .conftest import AlgorithmTester, get_data


class TestLinearInterp(AlgorithmTester):
    """Class for testing linear_interp baseline."""

    func = manual.linear_interp
    points = ((5, 10), (10, 20), (90, 100))

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, None, x, x, self.points)

    def test_output(self):
        super()._test_output(self.x, self.x, self.points)

    def test_list_input(self):
        x_list = self.x.tolist()
        super()._test_algorithm_list(
            array_args=(self.x, self.points), list_args=(x_list, self.points)
        )
