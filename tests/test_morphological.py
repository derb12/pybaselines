# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import pytest

from pybaselines import morphological

from .conftest import AlgorithmTester, get_data, has_pentapy


class TestMPLS(AlgorithmTester):
    """Class for testing mpls baseline."""

    func = morphological.mpls

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('weights', 'half_window'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(morphological, '_HAS_PENTAPY', False):
            scipy_output = self._call_func(self.y)[0]
        pentapy_output = self._call_func(self.y)[0]

        assert_allclose(pentapy_output, scipy_output, 1e-4)


class TestMor(AlgorithmTester):
    """Class for testing mor baseline."""

    func = morphological.mor

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIMor(AlgorithmTester):
    """Class for testing imor baseline."""

    func = morphological.imor

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window', 'iterations', 'last_tol'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAMorMol(AlgorithmTester):
    """Class for testing amormol baseline."""

    func = morphological.amormol

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window', 'iterations', 'last_tol'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestMorMol(AlgorithmTester):
    """Class for testing mormol baseline."""

    func = morphological.mormol

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window', 'iterations', 'last_tol'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestRollingBall(AlgorithmTester):
    """Class for testing rolling ball baseline."""

    func = morphological.rolling_ball

    _y = AlgorithmTester.y

    @pytest.mark.parametrize('half_window', (None, 1, np.full(_y.shape[0], 1), [1] * _y.shape[0]))
    @pytest.mark.parametrize(
        'smooth_half_window', (None, 1, np.full(_y.shape[0], 1), [1] * _y.shape[0])
    )
    def test_unchanged_data(self, data_fixture, half_window, smooth_half_window):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y, half_window, smooth_half_window)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_incorrect_half_window_fails(self):
        """Ensures an exception is raised if half_window is an array with len != len(data)."""
        with pytest.raises(ValueError):
            self._call_func(self.y, np.array([1, 1]))

    def test_incorrect_smooth_half_window_fails(self):
        """Ensures exception is raised if smooth_half_window is an array with len != len(data)."""
        with pytest.raises(ValueError):
            self._call_func(self.y, 1, np.array([1, 1]))

    def test_array_half_window_output(self):
        """
        Checks the array-based rolling ball versus the constant value implementation.

        Ensures that both give the same answer if the array of half-window values
        is the same value as the single half-window.
        """
        baseline_1 = self._call_func(self.y, 1, 1)[0]
        baseline_2 = self._call_func(self.y, np.full(self.y.shape[0], 1), 1)[0]

        assert_array_almost_equal(baseline_1, baseline_2)

    def test_array_smooth_half_window_output(self):
        """
        Checks the smoothing array-based rolling ball versus the constant value implementation.

        Ensures that both give the same answer if the array of smooth-half-window
        values is the same value as the single smooth-half-window.
        """
        baseline_1 = self._call_func(self.y, 1, 1)[0]
        baseline_2 = self._call_func(self.y, 1, np.full(self.y.shape[0], 1))[0]

        # avoid the edges since the two smoothing techniques will give slighly
        # different  results on the edges
        data_slice = slice(1, -1)
        assert_array_almost_equal(baseline_1[data_slice], baseline_2[data_slice])

    def test_different_array_half_window_output(self):
        """Ensures that the output is different when using changing window sizes."""
        baseline_1 = self._call_func(self.y, 1, 1)[0]

        half_windows = 1 + np.linspace(0, 5, self.y.shape[0], dtype=int)
        baseline_2 = self._call_func(self.y, half_windows, 1)[0]

        assert not np.allclose(baseline_1, baseline_2)

    def test_different_array_smooth_half_window_output(self):
        """Ensures that the output is different when using changing smoothing window sizes."""
        baseline_1 = self._call_func(self.y, 1, 1)[0]

        smooth_half_windows = 1 + np.linspace(0, 5, self.y.shape[0], dtype=int)
        baseline_2 = self._call_func(self.y, 1, smooth_half_windows)[0]

        # avoid the edges since the two smoothing techniques will give slighly
        # different  results on the edges and want to ensure the rest of the
        # data is also non-equal
        data_slice = slice(max(smooth_half_windows), -max(smooth_half_windows))
        assert not np.allclose(baseline_1[data_slice], baseline_2[data_slice])


class TestMWMV(AlgorithmTester):
    """Class for testing mwmv baseline."""

    func = morphological.mwmv

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestTophat(AlgorithmTester):
    """Class for testing tophat baseline."""

    func = morphological.tophat

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('half_window',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
