# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pybaselines import optimizers

from .conftest import get_data, AlgorithmTester


class TestCollabPLS(AlgorithmTester):
    """Class for testing collab_pls baseline."""

    func = optimizers.collab_pls

    @staticmethod
    def _stack(data):
        return np.vstack((data, data))

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()

        data_x, data_y = data_fixture
        stacked_data = (self._stack(data_x), self._stack(data_y))
        stacked_y = self._stack(y)
        super()._test_unchanged_data(stacked_data, stacked_y, None, stacked_y)

    def test_output(self):
        stacked_y = self._stack(self.y)
        super()._test_output(stacked_y, stacked_y)

    def test_list_input(self):
        y_list = self.y.tolist()
        stacked_y = self._stack(self.y)
        super()._test_algorithm_list(array_args=(stacked_y,), list_args=([y_list, y_list],))


class TestOptimizeExtendedRange(AlgorithmTester):
    """Class for testing optimize_extended_range baseline."""

    func = optimizers.optimize_extended_range

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, data_fixture, side):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, 'asls', side=side)

    def test_no_x(self):
        super()._test_algorithm_no_x(
            with_args=(self.y, self.x, 'asls'), without_args=(self.y, None, 'asls')
        )

    def test_x_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]
        regular_inputs_result = self._call_func(self.y, self.x, 'asls')[0]
        reverse_inputs_result = self._call_func(reverse_y, reverse_x, 'asls')[0]

        assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

    def test_output(self):
        super()._test_output(self.y, self.y, None, 'asls')

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(
            array_args=(self.y, None, 'asls'), list_args=(y_list, None, 'asls')
        )
