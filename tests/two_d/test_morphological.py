# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines.two_d import morphological

from ..base_tests import BaseTester2D, ensure_deprecation


class MorphologicalTester(BaseTester2D):
    """Base testing class for morphological functions."""

    module = morphological
    checked_keys = ('half_window',)

    @pytest.mark.parametrize('half_window', (None, 10, [10, 12], np.array([12, 10])))
    def test_half_window(self, half_window):
        """Ensures that different inputs for half_window work."""
        self.class_func(self.y, half_window=half_window)

    @ensure_deprecation(1, 4)
    def test_kwargs_deprecation(self):
        """Ensure passing kwargs outside of the window_kwargs keyword is deprecated."""
        with pytest.warns(DeprecationWarning):
            output, _ = self.class_func(self.y, min_half_window=2)
        output_2, _ = self.class_func(self.y, window_kwargs={'min_half_window': 2})

        # ensure the outputs are still the same
        assert_allclose(output_2, output, rtol=1e-12, atol=1e-12)


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

    @pytest.mark.parametrize('smooth_half_window', (None, 0, 10, [0, 0], [10, 10]))
    def test_smooth_half_windows(self, smooth_half_window):
        """Ensures smooth-half-window is correctly processed."""
        output = self.class_func(self.y, smooth_half_window=smooth_half_window)

        assert output[0].shape == self.y.shape

    @pytest.mark.parametrize('smooth_half_window', (-1, [5, -1], [-1, 5], [-2, -3]))
    def test_negative_smooth_half_window_fails(self, smooth_half_window):
        """Ensures a negative smooth-half-window raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, smooth_half_window=smooth_half_window)


class TestTophat(MorphologicalTester):
    """Class for testing tophat baseline."""

    func_name = 'tophat'
