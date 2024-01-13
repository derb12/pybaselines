# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

import pytest

from pybaselines.two_d import morphological

from ..conftest import BaseTester2D


class MorphologicalTester(BaseTester2D):
    """Base testing class for morphological functions."""

    module = morphological
    algorithm_base = morphological._Morphological
    checked_keys = ('half_window',)


class IterativeMorphologicalTester(MorphologicalTester):
    """Base testing class for iterative morphological functions."""

    checked_keys = ('half_window', 'tol_history')

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


class TestMor(MorphologicalTester):
    """Class for testing mor baseline."""

    func_name = 'mor'


class TestIMor(IterativeMorphologicalTester):
    """Class for testing imor baseline."""

    func_name = 'imor'


class TestRollingBall(MorphologicalTester):
    """Class for testing rolling_ball baseline."""

    func_name = 'rolling_ball'

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('half_window', (None, 10, [10, 12]))
    @pytest.mark.parametrize('smooth_half_window', (None, 0, 1))
    def test_unchanged_data(self, new_instance, half_window, smooth_half_window):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            new_instance, half_window=half_window, smooth_half_window=smooth_half_window
        )

    @pytest.mark.parametrize('smooth_half_window', (None, 0, 10))
    def test_smooth_half_windows(self, smooth_half_window):
        """Ensures smooth-half-window is correctly processed."""
        output = self.class_func(self.y, smooth_half_window=smooth_half_window)

        assert output[0].shape == self.y.shape


class TestTophat(MorphologicalTester):
    """Class for testing tophat baseline."""

    func_name = 'tophat'
