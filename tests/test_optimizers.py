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
    optimizers._get_function(method, [optimizers])


def test_get_function_fails_wrong_method():
    with pytest.raises(AttributeError):
        optimizers._get_function('unknown function', [optimizers])


def test_get_function_fails_no_module():
    with pytest.raises(AttributeError):
        optimizers._get_function('collab_pls', [])


class TestCollabPLS(AlgorithmTester):
    """Class for testing collab_pls baseline."""

    func = optimizers.collab_pls

    @staticmethod
    def _stack(data):
        return np.vstack((data, data))

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()

        data_x, data_y = data_fixture
        stacked_data = (self._stack(data_x), self._stack(data_y))
        stacked_y = self._stack(y)
        super()._test_unchanged_data(stacked_data, stacked_y, None, stacked_y)

    def test_output(self):
        stacked_y = self._stack(self.y)
        self._test_output(stacked_y, stacked_y, checked_keys=('weights',))

    def test_list_input(self):
        y_list = self.y.tolist()
        stacked_y = self._stack(self.y)
        super()._test_algorithm_list(array_args=(stacked_y,), list_args=([y_list, y_list],))

    @pytest.mark.parametrize(
        'method',
        ('asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa')
    )
    def test_all_methods(self, method):
        super()._call_func(self._stack(self.y), method=method)

    def test_unknown_method_fails(self):
        with pytest.raises(AttributeError):
            super()._call_func(self._stack(self.y), method='unknown function')


class TestOptimizeExtendedRange(AlgorithmTester):
    """Class for testing optimize_extended_range baseline."""

    func = optimizers.optimize_extended_range

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, data_fixture, side):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, side=side)

    def test_no_x(self):
        super()._test_algorithm_no_x(
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
        # will need to change checked_keys if default method is changed
        self._test_output(
            self.y, self.y,
            checked_keys=('weights', 'iterations', 'last_tol', 'optimal_parameter', 'min_rmse')
        )

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(
            array_args=(self.y, None), list_args=(y_list, None)
        )

    @pytest.mark.parametrize(
        'method',
        ('asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
         'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess')
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
        super()._call_func(self.y, self.x, method=method, height_scale=0.1, **kwargs)

    def test_unknown_method_fails(self):
        with pytest.raises(AttributeError):
            super()._call_func(self.y, self.x, method='unknown function')

    def test_unknown_side_fails(self):
        with pytest.raises(ValueError):
            super()._call_func(self.y, self.x, side='east')

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_polynomial_float_value_fails(self, key):
        with pytest.raises(TypeError):
            super()._call_func(self.y, self.x, method='modpoly', **{key: 1.5})

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_whittaker_high_value_fails(self, key):
        with pytest.raises(ValueError):
            super()._call_func(self.y, self.x, method='asls', **{key: 1e4})


class TestAdaptiveMinMax(AlgorithmTester):
    """Class for testing adaptive minmax baseline."""

    func = optimizers.adaptive_minmax

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        self._test_output(
            self.y, self.y, checked_keys=('weights', 'constrained_weights', 'poly_order')
        )

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('method', ('modpoly', 'imodpoly'))
    def test_methods(self, method):
        super()._test_output(self.y, self.y, self.x, method=method)

    def test_unknown_method_fails(self):
        with pytest.raises(KeyError):
            super()._test_output(self.y, self.y, method='unknown')

    @pytest.mark.parametrize('poly_order', (None, 0, [0], (0, 1)))
    def test_polyorder_inputs(self, poly_order):
        super()._test_output(self.y, self.y, self.x, poly_order)

    @pytest.mark.parametrize('poly_order', (0, [0], (0, 1)))
    def test_polyorder_outputs(self, poly_order):
        """Ensures that the correct polynomial orders were used."""
        _, params = super()._call_func(self.y, self.x, poly_order)
        assert params['poly_order'] == (0, 1)
