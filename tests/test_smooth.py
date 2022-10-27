# -*- coding: utf-8 -*-
"""Tests for pybaselines.smooth.

@author: Donald Erb
Created on March 20, 2021

"""

from numpy.testing import assert_allclose
import pytest

from pybaselines import smooth
from pybaselines.utils import ParameterWarning

from .conftest import BaseTester


class SmoothTester(BaseTester):
    """Base testing class for whittaker functions."""

    module = smooth
    algorithm_base = smooth._Smooth


class TestNoiseMedian(SmoothTester):
    """Class for testing noise median baseline."""

    func_name = 'noise_median'
    required_kwargs = {'half_window': 15}

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('smooth_hw', (None, 0, 2))
    def test_unchanged_data(self, use_class, smooth_hw):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, smooth_half_window=smooth_hw)

    @pytest.mark.parametrize('half_window', (None, 15))
    def test_half_windows(self, half_window):
        """Tests possible inputs for `half_window`."""
        self.class_func(self.y, half_window=half_window)


class TestSNIP(SmoothTester):
    """Class for testing snip median baseline."""

    func_name = 'snip'
    required_kwargs = {'max_half_window': 15}

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('smooth_hw', (-1, 0, 2))
    @pytest.mark.parametrize('decreasing', (True, False))
    def test_unchanged_data(self, use_class, smooth_hw, decreasing):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, smooth_half_window=smooth_hw, decreasing=decreasing
        )

    @pytest.mark.parametrize('max_half_window', (15, [15], [12, 20], (12, 15)))
    def test_max_half_window_inputs(self, max_half_window):
        """Tests valid inputs for `max_half_window`."""
        self.class_func(self.y, max_half_window)

    @pytest.mark.parametrize('filter_order', (2, 4, 6, 8))
    def test_filter_orders(self, filter_order):
        """Ensures all filter orders work."""
        self.class_func(self.y, 15, filter_order=filter_order)

    def test_wrong_filter_order(self):
        """Ensures a non-covered filter order fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, 15, filter_order=1)

    @pytest.mark.parametrize('max_half_window', (1000000, [1000000], [1000000, 1000000]))
    def test_too_large_max_half_window(self, max_half_window):
        """Ensures a warning emitted when max_half_window is greater than (len(data) - 1) // 2."""
        with pytest.warns(ParameterWarning):
            self.class_func(self.y, max_half_window)

    @pytest.mark.parametrize('max_half_window', ([10, 20, 30], (5, 10, 15, 20)))
    def test_too_many_max_half_windows_fails(self, max_half_window):
        """Ensures an error is raised if max-half-windows has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, max_half_window)

    @pytest.mark.parametrize('max_half_window', (None, 15))
    def test_max_half_windows(self, max_half_window):
        """Tests possible inputs for `max_half_window`."""
        self.class_func(self.y, max_half_window=max_half_window)


class TestSwima(SmoothTester):
    """Class for testing swima median baseline."""

    func_name = 'swima'
    checked_keys = ('half_window', 'converged')


class TestIpsa(SmoothTester):
    """Class for testing ipsa median baseline."""

    func_name = 'ipsa'
    checked_keys = ('tol_history',)

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('original_criteria', (True, False))
    def test_unchanged_data(self, use_class, original_criteria):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, original_criteria=original_criteria)


class TestRIA(SmoothTester):
    """Class for testing ria median baseline."""

    func_name = 'ria'
    checked_keys = ('tol_history',)

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, use_class, side):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, side=side)

    def test_unknown_side_fails(self):
        """Ensures function fails when the input side is not 'left', 'right', or 'both'."""
        with pytest.raises(ValueError):
            self.class_func(self.y, side='east')

    def test_x_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]

        reverse_fitter = self.algorithm_base(reverse_x, assume_sorted=False)

        # test both True and False for use_original
        regular_inputs_result = self.class_func(self.y)[0]
        reverse_inputs_result = reverse_fitter.ria(reverse_y)[0]

        assert_allclose(regular_inputs_result, reverse_inputs_result[::-1], 1e-10)

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
        regular_output = self.class_func(self.y, tol=high_tol, max_iter=high_max_iter)[1]
        assert len(regular_output['tol_history']) < high_max_iter
        assert regular_output['tol_history'][-1] < high_tol

        # should reach maximum iterations before reaching tol
        iter_output = self.class_func(self.y, tol=low_tol, max_iter=low_max_iter)[1]
        assert len(iter_output['tol_history']) == low_max_iter
        assert iter_output['tol_history'][-1] > low_tol

        # removed area should be exceeded before maximum iterations or tolerance are reached
        area_output = self.class_func(self.y, tol=low_tol, max_iter=high_max_iter)[1]
        assert len(area_output['tol_history']) < high_max_iter
        assert area_output['tol_history'][-1] > low_tol
