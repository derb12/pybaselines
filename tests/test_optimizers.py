# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pybaselines import optimizers

from .conftest import AlgorithmTester, get_data


@pytest.mark.parametrize('method', ('collab_pls', 'COLLAB_pls'))
def test_get_function(method):
    """Ensures _get_function gets the correct method, regardless of case."""
    selected_func = optimizers._get_function(method, [optimizers])
    assert selected_func is optimizers.collab_pls


def test_get_function_fails_wrong_method():
    """Ensures _get_function fails when an no function with the input name is available."""
    with pytest.raises(AttributeError):
        optimizers._get_function('unknown function', [optimizers])


def test_get_function_fails_no_module():
    """Ensures _get_function fails when not given any modules to search."""
    with pytest.raises(AttributeError):
        optimizers._get_function('collab_pls', [])


class TestCollabPLS(AlgorithmTester):
    """Class for testing collab_pls baseline."""

    func = optimizers.collab_pls

    @staticmethod
    def _stack(data):
        """Makes the data two-dimensional with shape (M, N) as required by collab_pls."""
        return np.vstack((data, data))

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()

        data_x, data_y = data_fixture
        stacked_data = (self._stack(data_x), self._stack(data_y))
        stacked_y = self._stack(y)
        self._test_unchanged_data(stacked_data, stacked_y, None, stacked_y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        stacked_y = self._stack(self.y)
        # will need to change checked_keys if default method is changed
        self._test_output(
            stacked_y, stacked_y,
            checked_keys=('average_weights', 'weights', 'tol_history')
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        stacked_y = self._stack(self.y)
        self._test_algorithm_list(array_args=(stacked_y,), list_args=([y_list, y_list],))

    @pytest.mark.parametrize(
        'method',
        ('asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa')
    )
    def test_all_methods(self, method):
        """Ensures all available methods work."""
        self._call_func(self._stack(self.y), method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self._call_func(self._stack(self.y), method='unknown function')


class TestOptimizeExtendedRange(AlgorithmTester):
    """Class for testing optimize_extended_range baseline."""

    func = optimizers.optimize_extended_range

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, data_fixture, side):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, side=side)

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

        assert_array_almost_equal(regular_inputs_result, reverse_inputs_result[::-1])

    def test_output(self):
        """Ensures that the output has the desired format."""
        # will need to change checked_keys if default method is changed
        self._test_output(
            self.y, self.y,
            checked_keys=('weights', 'tol_history', 'optimal_parameter', 'min_rmse')
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(
            array_args=(self.y, None), list_args=(y_list, None)
        )

    @pytest.mark.parametrize(
        'method',
        ('asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
         'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg', 'goldindec')
    )
    def test_all_methods(self, method):
        """Tests all methods that should work with optimize_extended_range."""
        if method == 'loess':
            # reduce number of calculations for loess since it is much slower
            kwargs = {'min_value': 1, 'max_value': 2}
        elif 'poly' not in method:
            # speed up whittaker tests
            kwargs = {'min_value': 4}
        else:
            kwargs = {}
        # use height_scale=0.1 to avoid exponential overflow warning for arpls and aspls
        self._call_func(self.y, self.x, method=method, height_scale=0.1, **kwargs)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self._call_func(self.y, self.x, method='unknown function')

    def test_unknown_side_fails(self):
        """Ensures function fails when the input side is not 'left', 'right', or 'both'."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, side='east')

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_polynomial_float_value_fails(self, key):
        """Ensures function fails when using a polynomial method with a float poly_order value."""
        with pytest.raises(TypeError):
            self._call_func(self.y, self.x, method='modpoly', **{key: 1.5})

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_whittaker_high_value_fails(self, key):
        """
        Ensures function fails when using a Whittaker method and input lambda exponent is too high.

        Since the function uses 10**exponent, do not want to allow a high exponent to be used,
        since the user probably thought the actual lam value had to be specficied rather than
        just the exponent.

        """
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, method='asls', **{key: 1e4})


class TestAdaptiveMinMax(AlgorithmTester):
    """Class for testing adaptive minmax baseline."""

    func = optimizers.adaptive_minmax

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(
            self.y, self.y, checked_keys=('weights', 'constrained_weights', 'poly_order')
        )

    def test_list_output(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('method', ('modpoly', 'imodpoly'))
    def test_methods(self, method):
        """Ensures all available methods work."""
        self._test_output(self.y, self.y, self.x, method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(KeyError):
            self._test_output(self.y, self.y, method='unknown')

    @pytest.mark.parametrize('poly_order', (None, 0, [0], (0, 1)))
    def test_polyorder_inputs(self, poly_order):
        """Tests valid inputs for poly_order."""
        self._test_output(self.y, self.y, self.x, poly_order)

    @pytest.mark.parametrize('poly_order', (0, [0], (0, 1)))
    def test_polyorder_outputs(self, poly_order):
        """Ensures that the correct polynomial orders were used."""
        _, params = self._call_func(self.y, self.x, poly_order)
        assert params['poly_order'] == (0, 1)
