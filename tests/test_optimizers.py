# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import optimizers, polynomial, utils

from .conftest import AlgorithmTester, get_data


class TestCollabPLS(AlgorithmTester):
    """Class for testing collab_pls baseline."""

    func = optimizers.collab_pls

    @staticmethod
    def _stack(data):
        """Makes the data two-dimensional with shape (M, N) as required by collab_pls."""
        return np.vstack((data, data))

    @pytest.mark.parametrize('average_dataset', (True, False))
    def test_unchanged_data(self, data_fixture, average_dataset):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()

        data_x, data_y = data_fixture
        stacked_data = (self._stack(data_x), self._stack(data_y))
        stacked_y = self._stack(y)
        self._test_unchanged_data(
            stacked_data, stacked_y, None, stacked_y, average_dataset=average_dataset
        )

    @pytest.mark.parametrize('average_dataset', (True, False))
    def test_output(self, average_dataset):
        """Ensures that the output has the desired format."""
        stacked_y = self._stack(self.y)
        # will need to change checked_keys if default method is changed
        self._test_output(
            stacked_y, stacked_y, average_dataset=average_dataset,
            checked_keys=('average_weights', 'weights', 'tol_history')
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        stacked_y = self._stack(self.y)
        self._test_algorithm_list(array_args=(stacked_y,), list_args=([y_list, y_list],))

    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'fabc'
        )
    )
    def test_all_methods(self, method):
        """Ensures all available methods work."""
        self._call_func(self._stack(self.y), method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self._call_func(self._stack(self.y), method='unknown function')

    def test_single_dataset_fails(self):
        """Ensures an error is raised if the input has the shape (N,)."""
        with pytest.raises(ValueError, match='the input data must'):
            self._call_func(self.y)


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

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_ordering(self, side):
        """Ensures arrays and weights are correctly sorted within the function."""
        reverse_x = self.x[::-1]
        reverse_y = self.y[::-1]
        input_weights = np.ones(len(self.y))
        input_weights[-1] = 0.9
        original_weights = input_weights.copy()
        method_kwargs = {'weights': input_weights}
        input_kwargs = method_kwargs.copy()

        regular_inputs_result = self._call_func(self.y, self.x, side=side)[0]
        reverse_inputs_result = self._call_func(
            reverse_y, reverse_x, side=side, method_kwargs=method_kwargs
        )[0]

        assert_allclose(regular_inputs_result, reverse_inputs_result[::-1], 1e-10)
        # also ensure the input weights are unchanged when sorting
        assert_array_equal(original_weights, input_weights)
        # ensure method_kwargs dict was unchanged
        total_dict = input_kwargs.copy()
        total_dict.update(method_kwargs)
        for key in total_dict.keys():
            if key not in input_kwargs or key not in method_kwargs:
                raise AssertionError('keys in method_kwargs were alterred')
            else:
                assert_array_equal(input_kwargs[key], method_kwargs[key])

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
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg', 'goldindec',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'dietrich', 'cwt_br', 'fabc'
        )
    )
    def test_all_methods(self, method):
        """Tests all methods that should work with optimize_extended_range."""
        if method == 'loess':
            # reduce number of calculations for loess since it is much slower
            kwargs = {'min_value': 1, 'max_value': 2}
        else:
            kwargs = {}
        # use height_scale=0.1 to avoid exponential overflow warning for arpls and aspls
        self._call_func(
            self.y, self.x, method=method, height_scale=0.1,
            pad_kwargs={'extrapolate_window': 100}, **kwargs
        )

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

    def test_kwargs_deprecation(self):
        """Ensures a warning is emitted for passing kwargs meant for the fitting function."""
        with pytest.warns(DeprecationWarning):
            self._call_func(self.y, self.x, method='asls', lam=1e8)


@pytest.mark.parametrize(
    'baseline_ptp', (0.01, 0.1, 0.3, 0.5, 1, 5, 10, 40, 100, 200, 300, 500, 600, 1000)
)
def test_determine_polyorders(baseline_ptp):
    """Ensures the correct polynomials are selected based on the signal to baseline ratio."""
    x = np.linspace(0, 100, 1000)
    # set y such that max(y) - min(y) is ~ 1 so that
    # ptp(baseline) / ptp(y) ~= ptp(baseline)
    y = (
        utils.gaussian(x, 1, 25)
        + utils.gaussian(x, 0.5, 50)
        + utils.gaussian(x, 1, 75)
    )
    # use a linear baseline so that it's easy to set the peak-to-peak of the baseline
    true_baseline = x * baseline_ptp / (x.max() - x.min())

    # double check to make sure the system is setup as expected
    assert_allclose(np.ptp(true_baseline), baseline_ptp, 0, 1e-3)
    assert_allclose(np.ptp(y), 1, 0, 1e-3)

    fit_baseline = polynomial.modpoly(y + true_baseline, x, 1)[0]
    # sanity check to make sure internal baseline fit was correct
    assert_allclose(np.ptp(fit_baseline), baseline_ptp, 0, 1e-3)

    if baseline_ptp < 0.2:
        expected_orders = (1, 2)
    elif baseline_ptp < 0.75:
        expected_orders = (2, 3)
    elif baseline_ptp < 8.5:
        expected_orders = (3, 4)
    elif baseline_ptp < 55:
        expected_orders = (4, 5)
    elif baseline_ptp < 240:
        expected_orders = (5, 6)
    elif baseline_ptp < 517:
        expected_orders = (6, 7)
    else:
        expected_orders = (6, 8)

    output_orders = optimizers._determine_polyorders(
        y + true_baseline, x, 1, None, polynomial.modpoly
    )

    assert output_orders == expected_orders


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
        with pytest.raises(AttributeError):
            self._test_output(self.y, self.y, method='unknown')

    @pytest.mark.parametrize('poly_order', (None, 0, [0], (0, 1)))
    def test_polyorder_inputs(self, poly_order):
        """Tests valid inputs for poly_order."""
        self._test_output(self.y, self.y, self.x, poly_order)

    @pytest.mark.parametrize('poly_order', (0, [0], (0, 1)))
    def test_polyorder_outputs(self, poly_order):
        """Ensures that the correct polynomial orders were used."""
        _, params = self._call_func(self.y, self.x, poly_order)
        assert_array_equal(params['poly_order'], np.array([0, 1]))

    @pytest.mark.parametrize('poly_order', ([0, 1, 2], (0, 1, 2, 3)))
    def test_too_many_polyorders_fails(self, poly_order):
        """Ensures an error is raised if poly_order has more than two items."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, poly_order)

    @pytest.mark.parametrize('constrained_fraction', (0.01, [0.01], (0, 0.01), [0.01, 1]))
    def test_constrained_fraction_inputs(self, constrained_fraction):
        """Tests valid inputs for constrained_fraction."""
        self._test_output(self.y, self.y, self.x, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_fraction', ([0.01, 0.02, 0.02], (0.01, 0.01, 0.01, 0.01)))
    def test_too_many_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction has more than two items."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_fraction', (-0.5, [-0.01, 0.02], 1.1, [0.05, 1.1]))
    def test_invalid_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction is outside of [0, 1]."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_weight', (1e5, [1e5], (1e3, 1e5)))
    def test_constrained_weight_inputs(self, constrained_weight):
        """Tests valid inputs for constrained_weight."""
        self._test_output(self.y, self.y, self.x, constrained_weight=constrained_weight)

    @pytest.mark.parametrize('constrained_weight', ([1e4, 1e2, 1e5], (1e3, 1e3, 1e3, 1e3)))
    def test_too_many_constrained_weight(self, constrained_weight):
        """Ensures an error is raised if constrained_weight has more than two items."""
        with pytest.raises(ValueError):
            self._call_func(self.y, self.x, constrained_weight=constrained_weight)
