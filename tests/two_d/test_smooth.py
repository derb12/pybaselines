# -*- coding: utf-8 -*-
"""Tests for pybaselines.smooth.

@author: Donald Erb
Created on January 14, 2024

"""

import pytest

from pybaselines.two_d import smooth

from ..conftest import BaseTester2D


class SmoothTester(BaseTester2D):
    """Base testing class for whittaker functions."""

    module = smooth
    algorithm_base = smooth._Smooth


class TestNoiseMedian(SmoothTester):
    """Class for testing noise median baseline."""

    func_name = 'noise_median'
    required_kwargs = {'half_window': 15}

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('smooth_hw', (None, 0, 2))
    def test_unchanged_data(self, new_instance, smooth_hw):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(new_instance, smooth_half_window=smooth_hw)

    @pytest.mark.parametrize('half_window', (None, 15))
    def test_half_windows(self, half_window):
        """Tests possible inputs for `half_window`."""
        self.class_func(self.y, half_window=half_window)
