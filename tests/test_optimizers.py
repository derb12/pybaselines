# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import optimizers, polynomial, utils

from .conftest import BaseTester, InputWeightsMixin


class OptimizerInputWeightsMixin(InputWeightsMixin):
    """Passes weights within the `method_kwargs` dictionary."""

    def test_input_weights(self, assertion_kwargs=None, **kwargs):
        """
        Ensures arrays are correctly sorted within the function.

        Returns the output for further testing.

        """
        # TODO replace with np.random.default_rng when min numpy version is >= 1.17
        weights = np.random.RandomState(0).normal(0.8, 0.05, len(self.x))
        weights = np.clip(weights, 0, 1).astype(float, copy=False)

        reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=False)

        regular_output, regular_output_params = self.class_func(
            data=self.y, method_kwargs={'weights': weights}, **self.kwargs, **kwargs
        )
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), method_kwargs={'weights': weights[::-1]},
            **self.kwargs, **kwargs
        )

        if assertion_kwargs is None:
            assertion_kwargs = {}
        if 'rtol' not in assertion_kwargs:
            assertion_kwargs['rtol'] = 1e-10
        if 'atol' not in assertion_kwargs:
            assertion_kwargs['atol'] = 1e-14

        for key in self.weight_keys:
            assert_allclose(
                regular_output_params[key], reverse_output_params[key][::-1],
                **assertion_kwargs
            )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), **assertion_kwargs
        )

        return regular_output, regular_output_params, reverse_output, reverse_output_params


class OptimizersTester(BaseTester):
    """Base testing class for optimizer functions."""

    module = optimizers
    algorithm_base = optimizers.Optimizers


class TestCollabPLS(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing collab_pls baseline."""

    func_name = "collab_pls"
    # will need to change checked_keys if default method is changed
    checked_keys = ('average_weights', 'weights', 'tol_history')
    two_d = True
    weight_keys = ('average_weights',)

    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'fabc'
        )
    )
    def test_all_methods(self, method):
        """Ensures all available methods work."""
        self.class_func(self.y, method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown function')

    def test_single_dataset_fails(self):
        """Ensures an error is raised if the input has the shape (N,)."""
        with pytest.raises(ValueError, match='the input data must'):
            self.class_func(np.arange(self.x.shape[0]))

    @pytest.mark.parametrize('average_dataset', (True, False))
    def test_input_weights(self, average_dataset):
        """Ensures the input weights are sorted correctly."""
        output = super().test_input_weights(average_dataset=average_dataset)
        regular_output, regular_output_params, reverse_output, reverse_output_params = output

        assert_allclose(
            regular_output_params['weights'],
            np.asarray(reverse_output_params['weights'])[..., ::-1],
            rtol=1e-12, atol=1e-14
        )


class TestOptimizeExtendedRange(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing collab_pls baseline."""

    func_name = "optimize_extended_range"
    # will need to change checked_keys if default method is changed
    checked_keys = ('weights', 'tol_history', 'optimal_parameter', 'min_rmse')
    required_kwargs = {'pad_kwargs': {'extrapolate_window': 100}}

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, use_class, side):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, side=side)

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_input_weights(self, side):
        """Ensures arrays are correctly sorted within the function."""
        super().test_input_weights(side=side)

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
        output = self.class_func(
            self.y, method=method, height_scale=0.1, **kwargs, **self.kwargs
        )
        if 'weights' in output[1]:
            assert self.y.shape == output[1]['weights'].shape
        elif 'alpha' in output[1]:
            assert self.y.shape == output[1]['alpha'].shape

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown function')

    def test_unknown_side_fails(self):
        """Ensures function fails when the input side is not 'left', 'right', or 'both'."""
        with pytest.raises(ValueError):
            self.class_func(self.y, side='east')

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_polynomial_float_value_fails(self, key):
        """Ensures function fails when using a polynomial method with a float poly_order value."""
        with pytest.raises(TypeError):
            self.class_func(self.y, method='modpoly', **{key: 1.5})

    @pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
    def test_whittaker_high_value_fails(self, key):
        """
        Ensures function fails when using a Whittaker method and input lambda exponent is too high.

        Since the function uses 10**exponent, do not want to allow a high exponent to be used,
        since the user probably thought the actual lam value had to be specficied rather than
        just the exponent.

        """
        with pytest.raises(ValueError):
            self.class_func(self.y, method='asls', **{key: 1e4})

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_aspls_alpha_ordering(self, side):
        """Ensures the `alpha` array for the aspls method is currectly processed."""
        alpha = np.random.RandomState(0).normal(0.8, 0.05, len(self.x))
        alpha = np.clip(alpha, 0, 1).astype(float, copy=False)

        reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=False)

        regular_output, regular_output_params = self.class_func(
            data=self.y, method='aspls', side=side, method_kwargs={'alpha': alpha},
            **self.kwargs
        )
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.y[::-1], method='aspls', side=side,
            method_kwargs={'alpha': alpha[::-1]}, **self.kwargs
        )

        for key in ('weights', 'alpha'):
            assert_allclose(
                regular_output_params[key], reverse_output_params[key][::-1],
                rtol=1e-10, atol=1e-14
            )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), rtol=1e-10, atol=1e-14
        )

    def test_kwargs_deprecation(self):
        """Ensures a warning is emitted for passing kwargs meant for the fitting function."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, method='asls', lam=1e8)


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

    fitter = polynomial.Polynomial(x, check_finite=False, assume_sorted=True)

    fit_baseline = fitter.modpoly(y + true_baseline, poly_order=1)[0]
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
        y + true_baseline, poly_order=1, weights=None, fit_function=fitter.modpoly
    )

    assert_array_equal(output_orders, expected_orders)


class TestAdaptiveMinMax(OptimizersTester, InputWeightsMixin):
    """Class for testing adaptive minmax baseline."""

    func_name = 'adaptive_minmax'
    checked_keys = ('weights', 'constrained_weights', 'poly_order')
    weight_keys = ('weights', 'constrained_weights')

    @pytest.mark.parametrize('method', ('modpoly', 'imodpoly'))
    def test_methods(self, method):
        """Ensures all available methods work."""
        self.class_func(self.y, method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown')

    @pytest.mark.parametrize('poly_order', (None, 0, [0], (0, 1)))
    def test_polyorder_inputs(self, poly_order):
        """Tests valid inputs for poly_order."""
        self.class_func(self.y, poly_order)

    @pytest.mark.parametrize('poly_order', (0, [0], (0, 1)))
    def test_polyorder_outputs(self, poly_order):
        """Ensures that the correct polynomial orders were used."""
        _, params = self.class_func(self.y, poly_order)
        assert_array_equal(params['poly_order'], np.array([0, 1]))

    @pytest.mark.parametrize('poly_order', ([0, 1, 2], (0, 1, 2, 3)))
    def test_too_many_polyorders_fails(self, poly_order):
        """Ensures an error is raised if poly_order has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, poly_order)

    @pytest.mark.parametrize('constrained_fraction', (0.01, [0.01], (0, 0.01), [0.01, 1]))
    def test_constrained_fraction_inputs(self, constrained_fraction):
        """Tests valid inputs for constrained_fraction."""
        self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_fraction', ([0.01, 0.02, 0.02], (0.01, 0.01, 0.01, 0.01)))
    def test_too_many_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_fraction', (-0.5, [-0.01, 0.02], 1.1, [0.05, 1.1]))
    def test_invalid_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction is outside of [0, 1]."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_weight', (1e5, [1e5], (1e3, 1e5)))
    def test_constrained_weight_inputs(self, constrained_weight):
        """Tests valid inputs for constrained_weight."""
        self.class_func(self.y, constrained_weight=constrained_weight)

    @pytest.mark.parametrize('constrained_weight', ([1e4, 1e2, 1e5], (1e3, 1e3, 1e3, 1e3)))
    def test_too_many_constrained_weight(self, constrained_weight):
        """Ensures an error is raised if constrained_weight has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_weight=constrained_weight)

    def test_input_weights(self):
        """Ensures the input weights are sorted correctly."""
        # use different weightings and constrained fractions for left and right
        # sides that that if weights are reversed, there is a clear difference
        weightings = np.array([1e4, 1e5])
        constrained_fractions = np.array([0.01, 0.02])
        super().test_input_weights(
            constrained_weight=weightings, constrained_fraction=constrained_fractions
        )
