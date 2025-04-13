# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import morphological

from .base_tests import BaseTester, InputWeightsMixin, RecreationMixin, ensure_deprecation


class MorphologicalTester(BaseTester):
    """Base testing class for morphological functions."""

    module = morphological
    algorithm_base = morphological._Morphological
    checked_keys = ('half_window',)

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


class TestMPLS(MorphologicalTester, InputWeightsMixin, RecreationMixin):
    """Class for testing mpls baseline."""

    func_name = 'mpls'
    checked_keys = ('half_window', 'weights')

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('pentadiagonal_solver', (1, 2))
    def test_pentadiagonal_solver(self, pentadiagonal_solver):
        """Ensure pentadiagonal solvers give similar result to SciPy's solvers."""
        self.algorithm.banded_solver = pentadiagonal_solver
        pentapy_output = self.class_func(self.y, diff_order=2)[0]

        self.algorithm.banded_solver = 3  # use solveh_banded
        solveh_output = self.class_func(self.y, diff_order=2)[0]
        self.algorithm.banded_solver = 4  # use solve_banded
        solve_output = self.class_func(self.y, diff_order=2)[0]

        assert_allclose(pentapy_output, solveh_output, rtol=5e-5, atol=1e-8)
        assert_allclose(pentapy_output, solve_output, rtol=5e-5, atol=1e-8)

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @ensure_deprecation(1, 4)
    def test_tol_deprecation(self):
        """Ensures a DeprecationWarning is given when tol is input."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, tol=1e-3)

    @ensure_deprecation(1, 4)
    def test_max_iter_deprecation(self):
        """Ensures a DeprecationWarning is given when max_iter is input."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, max_iter=20)

    @ensure_deprecation(1, 4)
    def test_recreation(self):
        """Ignores the warning emitted by inputting tol within RecreationMixin.test_recreation."""
        with pytest.warns(DeprecationWarning):
            super().test_recreation()


class TestMor(MorphologicalTester):
    """Class for testing mor baseline."""

    func_name = 'mor'


class TestIMor(IterativeMorphologicalTester):
    """Class for testing imor baseline."""

    func_name = 'imor'


class TestAMorMol(IterativeMorphologicalTester):
    """Class for testing amormol baseline."""

    func_name = 'amormol'


class TestMorMol(IterativeMorphologicalTester):
    """Class for testing mormol baseline."""

    func_name = 'mormol'

    def test_smooth_half_windows(self):
        """Ensures smooth-half-window is correctly processed.

        For mormol, smooth_half_window values of None, 0, and 1 should all produce no smoothing.
        """
        half_window = 15
        no_smooth_output = self.class_func(
            self.y, half_window=half_window, smooth_half_window=1
        )[0]
        for smooth_half_window in (None, 0):
            output = self.class_func(
                self.y, half_window=half_window, smooth_half_window=smooth_half_window
            )[0]
            assert_allclose(output, no_smooth_output, rtol=1e-12, atol=1e-12)


class TestRollingBall(MorphologicalTester):
    """Class for testing rolling_ball baseline."""

    func_name = 'rolling_ball'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('half_window', (None, 10))
    @pytest.mark.parametrize('smooth_half_window', (None, 0, 1))
    def test_unchanged_data(self, use_class, half_window, smooth_half_window):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, half_window=half_window, smooth_half_window=smooth_half_window
        )

    @pytest.mark.parametrize('smooth_half_window', (None, 0, 10))
    def test_smooth_half_windows(self, smooth_half_window):
        """Ensures smooth-half-window is correctly processed."""
        output = self.class_func(self.y, smooth_half_window=smooth_half_window)

        assert output[0].shape == self.y.shape


class TestMWMV(MorphologicalTester):
    """Class for testing mwmv baseline."""

    func_name = 'mwmv'

    @pytest.mark.parametrize('smooth_half_window', (None, 0, 10))
    def test_smooth_half_windows(self, smooth_half_window):
        """Ensures smooth-half-window is correctly processed."""
        output = self.class_func(self.y, smooth_half_window=smooth_half_window)

        assert output[0].shape == self.y.shape


class TestTophat(MorphologicalTester):
    """Class for testing tophat baseline."""

    func_name = 'tophat'


class TestMpspline(MorphologicalTester, InputWeightsMixin, RecreationMixin):
    """Class for testing mpspline baseline."""

    func_name = 'mpspline'
    checked_keys = ('half_window', 'weights')

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e1, 3: 1e6}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('spline_degree', (2, 4))
    def test_spline_degrees(self, spline_degree):
        """Ensure that other spline degrees work."""
        self.class_func(self.y, spline_degree=spline_degree)

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)


class TestJBCD(MorphologicalTester):
    """Class for testing jbcd baseline."""

    func_name = 'jbcd'
    checked_keys = ('half_window', 'tol_history', 'signal')

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('robust_opening', (False, True))
    def test_unchanged_data(self, use_class, robust_opening):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, robust_opening=robust_opening)

    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        factor = {2: 1e4, 3: 1e10}[diff_order]
        self.class_func(self.y, beta=factor, gamma=factor, diff_order=diff_order)

    @pytest.mark.parametrize('pentadiagonal_solver', (1, 2))
    def test_pentadiagonal_solver(self, pentadiagonal_solver):
        """Ensure pentadiagonal solvers give similar result to SciPy's solvers."""
        self.algorithm.banded_solver = pentadiagonal_solver
        pentapy_output = self.class_func(self.y, diff_order=2)[0]

        self.algorithm.banded_solver = 3  # use solveh_banded
        solveh_output = self.class_func(self.y, diff_order=2)[0]
        self.algorithm.banded_solver = 4  # use solve_banded
        solve_output = self.class_func(self.y, diff_order=2)[0]

        assert_allclose(pentapy_output, solveh_output, rtol=5e-5, atol=1e-8)
        assert_allclose(pentapy_output, solve_output, rtol=5e-5, atol=1e-8)

    def test_zero_gamma_passes(self):
        """Ensures gamma can be 0, which just does baseline correction without denoising."""
        self.class_func(self.y, gamma=0)

    def test_zero_beta_fails(self):
        """Ensures a beta equal to 0 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, beta=0)

    def test_array_beta_gamma_fails(self):
        """Ensures array-like beta or gamma values raise an exception."""
        array_vals = np.ones_like(self.y)
        with pytest.raises(ValueError):
            self.class_func(self.y, beta=array_vals)
        with pytest.raises(ValueError):
            self.class_func(self.y, gamma=array_vals)

    def test_tol_history(self):
        """
        Ensures the 'tol_history' item in the parameter output is correct.

        Ensures that iterations continue as long as the tolerance is less than
        either `tol` or `tol_2`.

        """
        max_iter = 5
        expected_iters = max_iter + 1

        for kwargs in ({'tol': -1}, {'tol_2': -1}):
            _, params = self.class_func(self.y, max_iter=max_iter, **kwargs)

            assert params['tol_history'].size == 2 * expected_iters
            assert params['tol_history'][:, 0].size == expected_iters
            assert params['tol_history'][:, 1].size == expected_iters
