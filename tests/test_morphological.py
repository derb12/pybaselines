# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pybaselines import morphological

from .conftest import get_data, AlgorithmTester


class TestMPLS(AlgorithmTester):
    """Class for testing mpls baseline."""

    func = morphological.mpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestMor(AlgorithmTester):
    """Class for testing mor baseline."""

    func = morphological.mor

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIMor(AlgorithmTester):
    """Class for testing imor baseline."""

    func = morphological.imor

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAMorMol(AlgorithmTester):
    """Class for testing amormol baseline."""

    func = morphological.amormol

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestMorMol(AlgorithmTester):
    """Class for testing mormol baseline."""

    func = morphological.mormol

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestRollingBall(AlgorithmTester):
    """Class for testing rolling ball baseline."""

    func = morphological.rolling_ball

    _y = AlgorithmTester.y

    @pytest.mark.parametrize('half_window', (None, 1, np.full(_y.shape[0], 1), [1] * _y.shape[0]))
    @pytest.mark.parametrize(
        'smooth_half_window', (None, 1, np.full(_y.shape[0], 1), [1] * _y.shape[0])
    )
    def test_unchanged_data(self, data_fixture, half_window, smooth_half_window):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y, half_window, smooth_half_window)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_incorrect_half_window_fails(self):
        with pytest.raises(ValueError):
            super()._call_func(self.y, np.array([1, 1]))

    def test_incorrect_smooth_half_window_fails(self):
        with pytest.raises(ValueError):
            super()._call_func(self.y, 1, np.array([1, 1]))

    def test_array_half_window_output(self):
        baseline_1 = super()._call_func(self.y, 1, 1)[0]
        baseline_2 = super()._call_func(self.y, np.full(self.y.shape[0], 1), 1)[0]

        assert_array_almost_equal(baseline_1, baseline_2)

    def test_array_smooth_half_window_output(self):
        baseline_1 = super()._call_func(self.y, 1, 1)[0]
        baseline_2 = super()._call_func(self.y, 1, np.full(self.y.shape[0], 1))[0]

        # avoid the edges since the two smoothing techniques will give slighly
        # different  results on the edges
        data_slice = slice(1, -1)
        assert_array_almost_equal(baseline_1[data_slice], baseline_2[data_slice])

    def test_different_array_half_window_output(self):
        """Ensures that the output is different when using changing window sizes."""
        baseline_1 = super()._call_func(self.y, 1, 1)[0]

        half_windows = 1 + np.linspace(0, 5, self.y.shape[0], dtype=int)
        baseline_2 = super()._call_func(self.y, half_windows, 1)[0]

        assert not np.allclose(baseline_1, baseline_2)

    def test_different_array_smooth_half_window_output(self):
        """Ensures that the output is different when using changing smoothing window sizes."""
        baseline_1 = super()._call_func(self.y, 1, 1)[0]

        smooth_half_windows = 1 + np.linspace(0, 5, self.y.shape[0], dtype=int)
        baseline_2 = super()._call_func(self.y, 1, smooth_half_windows)[0]

        # avoid the edges since the two smoothing techniques will give slighly
        # different  results on the edges and want to ensure the rest of the
        # data is also non-equal
        data_slice = slice(max(smooth_half_windows), -max(smooth_half_windows))
        assert not np.allclose(baseline_1[data_slice], baseline_2[data_slice])
