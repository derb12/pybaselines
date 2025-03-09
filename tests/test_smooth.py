# -*- coding: utf-8 -*-
"""Tests for pybaselines.smooth.

@author: Donald Erb
Created on March 20, 2021

"""

from numpy.testing import assert_allclose
import pytest

from pybaselines import smooth
from pybaselines.utils import ParameterWarning

from .conftest import BaseTester, ensure_deprecation, get_data


class SmoothTester(BaseTester):
    """Base testing class for whittaker functions."""

    module = smooth
    algorithm_base = smooth._Smooth
    uses_padding = True  # TODO remove after version 1.4 when kwargs are deprecated

    @ensure_deprecation(1, 4)
    def test_kwargs_deprecation(self):
        """Ensure passing kwargs outside of the pad_kwargs keyword is deprecated."""
        if not self.uses_padding:
            return
        with pytest.warns(DeprecationWarning):
            output, _ = self.class_func(self.y, mode='edge')
        output_2, _ = self.class_func(self.y, pad_kwargs={'mode': 'edge'})

        # ensure the outputs are still the same
        assert_allclose(output_2, output, rtol=1e-12, atol=1e-12)

        # also ensure both pad_kwargs and **kwargs are passed to pad_edges; some algorithms do
        # the padding outside of setup_smooth, so have to do this to cover those cases
        with pytest.raises(TypeError):
            with pytest.warns(DeprecationWarning):
                self.class_func(self.y, pad_kwargs={'mode': 'extrapolate'}, mode='extrapolate')


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


def nieve_directional_min_moving_avg(y, data_len, half_window):
    """A simpler version of pybaselines.smooth.nieve_directional_min_moving_avg for testing."""
    output = y.copy()
    for i in range(1, data_len - 1):
        if i - half_window < 0:
            hw = i
        else:
            hw = half_window
        # half_window could also be too large on the right side; in actual
        # implementation, half_window is set to (data_len - 1) // 2 to prevent
        # this occurring
        if i + hw > data_len - 1:
            hw = data_len - 1 - i
        output[i] = min(output[i], output[i - hw:i + hw + 1].mean())

    return output


@pytest.mark.parametrize('half_window', (1, 5, 50, 249, 251))
def test_directional_min_moving_avg(half_window):
    """Ensures the output of _directional_min_moving_avg is correct."""
    _, y = get_data(num_points=500)
    len_y = len(y)
    y_input = y.copy()

    expected = nieve_directional_min_moving_avg(y, len_y, half_window)
    output = smooth._directional_min_moving_avg(y_input, len_y, half_window)

    assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
    # y_input should be modified inplace
    assert_allclose(y_input, expected, rtol=1e-12, atol=1e-12)

    # now do the second direction
    expected = nieve_directional_min_moving_avg(expected[::-1], len_y, half_window)[::-1]
    output = smooth._directional_min_moving_avg(y_input[::-1], len_y, half_window)[::-1]

    assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
    assert_allclose(y_input, expected, rtol=1e-12, atol=1e-12)


class TestPeakFilling(SmoothTester):
    """Class for testing peak_filling baseline."""

    func_name = 'peak_filling'
    checked_keys = ('x_fit', 'baseline_fit')
    uses_padding = False  # TODO after version 1.4 when passing kwargs is deprecated

    @pytest.mark.parametrize('half_window', (None, 15))
    def test_half_windows(self, half_window):
        """Tests possible inputs for `half_window`."""
        self.class_func(self.y, half_window=half_window)

    @pytest.mark.parametrize('half_window', (0, -5))
    def test_non_positive_half_window_fails(self, half_window):
        """Ensures an exception is raised when `half_window` is non-positive."""
        with pytest.raises(ValueError):
            self.class_func(self.y, half_window=half_window)

    def test_too_large_half_window_warns(self):
        """Ensures an exception is raised when `half_window` is too large."""
        # warning emitted when half_window > (sections - 1) // 2
        sections = 10
        half_window = ((sections - 1) // 2) + 1
        with pytest.warns(ParameterWarning):
            self.class_func(self.y, half_window=half_window, sections=sections)

    @pytest.mark.parametrize('sections', (0, -5))
    def test_non_positive_sections_fails(self, sections):
        """Ensures an exception is raised when `sections` is non-positive."""
        with pytest.raises(ValueError):
            self.class_func(self.y, sections=sections)

    def test_too_large_sections_fails(self):
        """Ensures an exception is raised when `sections` is larger than the data length."""
        with pytest.raises(ValueError):
            self.class_func(self.y, sections=len(self.y) + 1)
