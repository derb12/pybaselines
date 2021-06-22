# -*- coding: utf-8 -*-
"""Tests for pybaselines.misc.

@author: Donald Erb
Created on March 20, 2021

"""

import pytest

from pybaselines import misc

from .conftest import AlgorithmTester, get_data


class TestInterpPts(AlgorithmTester):
    """Class for testing interp_pts baseline."""

    func = misc.interp_pts
    points = ((5, 10), (10, 20), (90, 100))

    @pytest.mark.parametrize('interp_method', ('linear', 'slinear', 'quadratic'))
    def test_unchanged_data(self, data_fixture, interp_method):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(
            data_fixture, None, x, x, self.points, interp_method=interp_method
        )

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.x, self.x, self.points, checked_keys=())

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        x_list = self.x.tolist()
        self._test_algorithm_list(
            array_args=(self.x, self.points), list_args=(x_list, self.points)
        )
