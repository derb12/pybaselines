# -*- coding: utf-8 -*-
"""Tests for pybaselines.window.

@author: Donald Erb
Created on March 20, 2021

"""

from numpy.testing import assert_allclose
import pytest

from pybaselines import window
from pybaselines.utils import ParameterWarning

from .conftest import AlgorithmTester, get_data


@pytest.mark.parametrize(
    'func_and_args', (
        [window.noise_median, (15,)],
        [window.snip, (15,)],
        [window.swima, ()],
        [window.ipsa, ()],
        [window.ria, ()],

    )
)
def test_deprecations(func_and_args):
    """Ensures deprecations are emitted for all functions in pybaselines.window."""
    func, args = func_and_args
    x, y = get_data()
    with pytest.warns(DeprecationWarning, match='pybaselines.window is deprecated'):
        func(y, *args)


@pytest.mark.filterwarnings('ignore:pybaselines.window is deprecated')
class TestNoiseMedian(AlgorithmTester):
    """Class for testing noise median baseline."""

    func = window.noise_median

    @pytest.mark.parametrize('smooth_hw', (None, 0, 2))
    def test_unchanged_data(self, data_fixture, smooth_hw):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y, 15, smooth_half_window=smooth_hw)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, 15, checked_keys=())

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))


@pytest.mark.filterwarnings('ignore:pybaselines.window is deprecated')
class TestSNIP(AlgorithmTester):
    """Class for testing snip baseline."""

    func = window.snip

    @pytest.mark.parametrize('smooth_hw', (-1, 0, 2))
    @pytest.mark.parametrize('decreasing', (True, False))
    def test_unchanged_data(self, data_fixture, smooth_hw, decreasing):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(
            data_fixture, y, None, y, 15, smooth_half_window=smooth_hw, decreasing=decreasing
        )

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, 15, checked_keys=())

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y, 15), list_args=(y_list, 15))

    @pytest.mark.parametrize('max_half_window', (15, [15], [12, 20], (12, 15)))
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

    @pytest.mark.parametrize('max_half_window', ([10, 20, 30], (5, 10, 15, 20)))
    def test_too_many_max_half_windows_fails(self, max_half_window):
        """Ensures an error is raised if max-half-windows has more than two items."""
        with pytest.raises(ValueError):
            self._call_func(self.y, max_half_window)


@pytest.mark.filterwarnings('ignore:pybaselines.window is deprecated')
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


@pytest.mark.filterwarnings('ignore:pybaselines.window is deprecated')
class TestIpsa(AlgorithmTester):
    """Class for testing ipsa baseline."""

    func = window.ipsa

    @pytest.mark.parametrize('original_criteria', (True, False))
    def test_unchanged_data(self, data_fixture, original_criteria):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y, original_criteria=original_criteria)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('tol_history',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


@pytest.mark.filterwarnings('ignore:pybaselines.window is deprecated')
class TestRIA(AlgorithmTester):
    """Class for testing ria baseline."""

    func = window.ria

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, data_fixture, side):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, side=side)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('tol_history',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(
            with_args=(self.y, self.x), without_args=(self.y, None)
        )

    def test_x_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]
        regular_inputs_result = self._call_func(self.y, self.x)[0]
        reverse_inputs_result = self._call_func(reverse_y, reverse_x)[0]

        assert_allclose(regular_inputs_result, reverse_inputs_result[::-1])

    def test_unknown_side_fails(self):
        """Ensures function fails when the input side is not 'left', 'right', or 'both'."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, side='east')

    def test_exit_conditions(self):
        """
        Tests the three exit conditions of the algorithm.

        Can either exit due to reaching max iterations, reaching the desired tolerance,
        or if the removed area exceeds the initial area of the extended portions.

        """
        high_max_iter = 500
        low_max_iter = 5
        high_tol = 1e-1
        low_tol = -1
        # set tol rather high so that it is guaranteed to be reached
        regular_output = self._call_func(self.y, self.x, tol=high_tol, max_iter=high_max_iter)[1]
        assert len(regular_output['tol_history']) < high_max_iter
        assert regular_output['tol_history'][-1] < high_tol

        # should reach maximum iterations before reaching tol
        iter_output = self._call_func(self.y, self.x, tol=low_tol, max_iter=low_max_iter)[1]
        assert len(iter_output['tol_history']) == low_max_iter
        assert iter_output['tol_history'][-1] > low_tol

        # removed area should be exceeded before maximum iterations or tolerance are reached
        area_output = self._call_func(self.y, self.x, tol=low_tol, max_iter=high_max_iter)[1]
        assert len(area_output['tol_history']) < high_max_iter
        assert area_output['tol_history'][-1] > low_tol
