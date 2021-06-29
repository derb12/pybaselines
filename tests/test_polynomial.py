# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from pybaselines import polynomial
from pybaselines.utils import ParameterWarning

from .conftest import AlgorithmTester, get_data


class TestPoly(AlgorithmTester):
    """Class for testing regular polynomial baseline."""

    func = polynomial.poly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestModPoly(AlgorithmTester):
    """Class for testing ModPoly baseline."""

    func = polynomial.modpoly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIModPoly(AlgorithmTester):
    """Class for testing IModPoly baseline."""

    func = polynomial.imodpoly

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestPenalizedPoly(AlgorithmTester):
    """Class for testing penalized_poly baseline."""

    func = polynomial.penalized_poly

    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'symmetric_truncated_quadratic',
            'a_truncated_quadratic',  # test that 'a' and 's' work as well
            's_truncated_quadratic',
            'asymmetric_huber',
            'symmetric_huber',
            'asymmetric_indec',
            'symmetric_indec'
        )
    )
    def test_unchanged_data(self, data_fixture, cost_function):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self._call_func(self.y, self.x, cost_function='a_hub')

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
    def test_weighting(self, weight_enum):
        """
        Tests that weighting is correctly applied by comparing to other algorithms.

        Weights were not included in the original penalized_poly method developed
        in [1]_, so need to ensure that their usage in pybaselines is correct.

        According to [1]_ (and independently verified), the penalized_poly function
        with the asymmetric truncated quadratic cost function, a threshold of 0, and
        an alpha_factor of 1 should be the same as the output of the ModPoly algorithm.

        Furthermore, the penalized_poly with any symmetric cost function and a threshold
        of infinity should equal to the output of a regular polynomial fit.

        Therefore, to ensure that weighting is correct for the penalized_poly, check
        both conditions.

        References
        ----------
        .. [1] Mazet, V., et al. Background removal from spectra by designing and
               minimising a non-quadratic cost function. Chemometrics and Intelligent
               Laboratory Systems, 2005, 76(2), 121–133.

        """
        if weight_enum == 0:
            # all weights = 1
            weights = None
        elif weight_enum == 1:
            # same as all weights = 1, but would cause issues if weights were
            # incorrectly multiplied
            weights = 2 * np.ones_like(self.y)
        elif weight_enum == 2:
            # binary mask, only fitting the first half of the data
            weights = np.ones_like(self.y)
            weights[self.x < 0.5 * (np.max(self.x) + np.min(self.x))] = 0
        else:
            # weight array where the two endpoints have weighting >> 1
            weights = np.ones_like(self.y)
            fraction = max(1, ceil(self.y.shape[0] * 0.1))
            weights[:fraction] = 100
            weights[-fraction:] = 100

        poly_order = 2
        tol = 1e-3

        poly_baseline = polynomial.poly(self.y, self.x, poly_order, weights=weights)[0]
        penalized_poly_1 = self._call_func(
            self.y, self.x, poly_order, cost_function='s_huber',
            threshold=1e10, weights=weights
        )[0]

        assert_array_almost_equal(poly_baseline, penalized_poly_1)

        modpoly_baseline = polynomial.modpoly(
            self.y, self.x, poly_order, tol=tol, weights=weights, use_original=True
        )[0]
        penalized_poly_2 = self._call_func(
            self.y, self.x, poly_order, cost_function='a_truncated_quadratic',
            threshold=0, weights=weights, alpha_factor=1, tol=tol
        )[0]

        assert_array_almost_equal(modpoly_baseline, penalized_poly_2)


class TestLoess(AlgorithmTester):
    """Class for testing LOESS baseline."""

    func = polynomial.loess

    @pytest.mark.parametrize('conserve_memory', (True, False))
    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_unchanged_data(self, data_fixture, use_threshold, conserve_memory):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(
            data_fixture, y, x, y, x, use_threshold=use_threshold, conserve_memory=conserve_memory
        )

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['weights', 'iterations', 'last_tol']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_x_ordering(self, use_threshold):
        """Ensures arrays are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]

        if use_threshold:
            # test both True and False for use_original
            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold, use_original=False
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold, use_original=False
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold, use_original=True
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold, use_original=True
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

        else:
            regular_inputs_result = self._call_func(
                self.y, self.x, use_threshold=use_threshold
            )[0]
            reverse_inputs_result = self._call_func(
                reverse_y, reverse_x, use_threshold=use_threshold
            )[0]

            assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

    @pytest.mark.parametrize('fraction', (-0.1, 1.1, 5))
    def test_wrong_fraction_fails(self, fraction):
        """Ensures a fraction value outside of (0, 1) raises an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, fraction)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3))
    def test_too_small_window_fails(self, poly_order):
        """Ensures a window smaller than poly_order + 1 raises an exception."""
        for num_points in range(poly_order + 1):
            with pytest.raises(ValueError):
                self._call_func(self.y, self.x, total_points=num_points, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    def test_high_polynomial_order_warns(self, poly_order):
        """Ensure a warning is emitted when using a polynomial order above 2."""
        if poly_order > 2:
            with pytest.warns(ParameterWarning):
                self._call_func(self.y, self.x, poly_order=poly_order)
        else:  # no warning should be emitted
            self._call_func(self.y, self.x, poly_order=poly_order)

    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_use_threshold_weights_reset(self, conserve_memory):
        """Ensures weights are reset to 1 after first iteration if use_threshold is True."""
        weights = np.arange(self.y.shape[0])
        one_weights = np.ones(self.y.shape[0])
        # will exit fitting loop before weights are reset on first loop
        _, params_first_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=1e10
        )
        assert_array_equal(weights, params_first_iter['weights'])

        # will exit fitting loop after first iteration but after reassigning weights
        _, params_second_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=-1, max_iter=1
        )
        # will exit fitting loop after second iteration
        _, params_third_iter = self._call_func(
            self.y, self.x, weights=weights, conserve_memory=conserve_memory,
            use_threshold=True, tol=-1, max_iter=2
        )

        assert_array_equal(one_weights, params_second_iter['weights'])
        assert_array_equal(one_weights, params_third_iter['weights'])

