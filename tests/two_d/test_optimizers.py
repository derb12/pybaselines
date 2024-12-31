# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on January 14, 2024

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import utils
from pybaselines.two_d import optimizers, polynomial

from ..conftest import BaseTester2D, InputWeightsMixin


class OptimizerInputWeightsMixin(InputWeightsMixin):
    """Passes weights within the `method_kwargs` dictionary."""

    def test_input_weights(self, assertion_kwargs=None, **kwargs):
        """Ensures arrays are correctly sorted within the function."""
        weights = np.random.default_rng(0).normal(0.8, 0.05, self.y.shape[-2:])
        weights = np.clip(weights, 0, 1, dtype=float)

        reverse_fitter = self.algorithm_base(self.x[::-1], self.z[::-1], assume_sorted=False)

        regular_output, regular_output_params = self.class_func(
            data=self.y, method_kwargs={'weights': weights}, **self.kwargs, **kwargs
        )
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), method_kwargs={'weights': self.reverse_array(weights)},
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
                regular_output_params[key], self.reverse_array(reverse_output_params[key]),
                **assertion_kwargs
            )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), **assertion_kwargs
        )


class OptimizersTester(BaseTester2D):
    """Base testing class for optimizer functions."""

    module = optimizers
    algorithm_base = optimizers._Optimizers


class TestCollabPLS(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing collab_pls baseline."""

    func_name = "collab_pls"
    # will need to change checked_keys if default method is changed
    checked_keys = ('average_weights', 'weights', 'tol_history')
    three_d = True
    weight_keys = ('average_weights', 'weights')

    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'mixture_model', 'irsqr', 'pspline_asls',
            'pspline_airpls', 'pspline_arpls',
            'pspline_iarpls', 'pspline_psalsa', 'brpls', 'pspline_brpls'
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
        """Ensures an error is raised if the input has the shape (M, N)."""
        with pytest.raises(ValueError, match='the input data must'):
            self.class_func(np.arange(self.y[0].size).reshape(self.y.shape[-2:]))


@pytest.mark.parametrize(
    'baseline_ptp', (0.01, 0.1, 0.3, 0.5, 1, 5, 10, 40, 100, 200, 300, 500, 600, 1000)
)
def test_determine_polyorders(baseline_ptp):
    """Ensures the correct polynomials are selected based on the signal to baseline ratio."""
    x = np.linspace(0, 100, 500)
    z = np.linspace(0, 100, 400)
    X, Z = np.meshgrid(x, z, indexing='ij')
    # set y such that max(y) - min(y) is ~ 1 so that
    # ptp(baseline) / ptp(y) ~= ptp(baseline)
    y = (
        utils.gaussian2d(X, Z, 1, 25, 25, 2, 2)
        + utils.gaussian2d(X, Z, 0.5, 50, 50, 2, 2)
        + utils.gaussian2d(X, Z, 1, 75, 75, 2, 2)
    )
    # use a linear baseline so that it's easy to set the peak-to-peak of the baseline
    true_baseline = X * baseline_ptp / (x.max() - x.min())

    # double check to make sure the system is setup as expected
    assert_allclose(np.ptp(true_baseline), baseline_ptp, 0, 1e-3)
    assert_allclose(np.ptp(y), 1, 0, 1e-3)

    fitter = polynomial._Polynomial(x, z, check_finite=False, assume_sorted=True)

    fit_baseline = fitter.modpoly(y + true_baseline, poly_order=1)[0]
    # sanity check to make sure internal baseline fit was correct
    assert_allclose(np.ptp(fit_baseline), baseline_ptp, 0, 5e-3)

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
    """Class for testing adaptive_minmax baseline."""

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

    @pytest.mark.parametrize(
        'constrained_fraction', (0.01, [0.01], (0, 0.01), [0.01, 1], [0.01, 0.01, 0.01, 0.01])
    )
    def test_constrained_fraction_inputs(self, constrained_fraction):
        """Tests valid inputs for constrained_fraction."""
        self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize(
        'constrained_fraction', ([0.01, 0.02, 0.02], (0.01, 0.01, 0.01, 0.01, 0.01))
    )
    def test_too_many_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_fraction', (-0.5, [-0.01, 0.02], 1.1, [0.05, 1.1]))
    def test_invalid_constrained_fraction(self, constrained_fraction):
        """Ensures an error is raised if constrained_fraction is outside of [0, 1]."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_fraction=constrained_fraction)

    @pytest.mark.parametrize('constrained_weight', (1e5, [1e5], (1e3, 1e5), [1e3, 1e3, 1e3, 1e3]))
    def test_constrained_weight_inputs(self, constrained_weight):
        """Tests valid inputs for constrained_weight."""
        self.class_func(self.y, constrained_weight=constrained_weight)

    @pytest.mark.parametrize('constrained_weight', ([1e4, 1e2, 1e5], (1e3, 1e3, 1e3, 1e3, 1e3)))
    def test_too_many_constrained_weight(self, constrained_weight):
        """Ensures an error is raised if constrained_weight has more than two items."""
        with pytest.raises(ValueError):
            self.class_func(self.y, constrained_weight=constrained_weight)

    def test_input_weights(self):
        """Ensures the input weights are sorted correctly."""
        # use different weightings and constrained fractions for left and right
        # sides that that if weights are reversed, there is a clear difference
        weightings = np.array([1e4, 1e5, 1e4, 1e5])
        constrained_fractions = np.array([0.01, 0.02, 0.01, 0.02])
        super().test_input_weights(
            constrained_weight=weightings, constrained_fraction=constrained_fractions
        )


class TestIndividualAxes(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing individual_axes baseline."""

    func_name = 'individual_axes'
    checked_keys = ()

    @pytest.mark.parametrize('axes', (0, 1, [0], [1], (0, 1), [1, 0]))
    def test_output(self, axes):
        """Tests the output fitting the rows and/or columns."""
        if isinstance(axes, int):
            input_axes = [axes]
        else:
            input_axes = axes

        additional_keys = []
        if 0 in input_axes:
            additional_keys.extend(['params_rows', 'baseline_rows'])
        if 1 in input_axes:
            additional_keys.extend(['params_columns', 'baseline_columns'])

        super().test_output(additional_keys=additional_keys, axes=axes)

    def test_xz_ordering(self):
        """
        Ensures x and z are correctly sorted.

        Uses a relatively low tolerance since the fits are independent of each other
        so they are not guaranteed to be the same when the inputs are reversed.

        """
        super().test_xz_ordering(assertion_kwargs={'rtol': 1e-6})

    @pytest.mark.parametrize('axes', (0, 1, [0], [1], (0, 1), [1, 0]))
    def test_input_weights(self, axes):
        """Ensures arrays are correctly sorted within the function."""
        if isinstance(axes, int) or len(axes) == 1:
            input_axes = [axes]
            scalar_axes = True
        else:
            input_axes = axes
            scalar_axes = False

        if scalar_axes:
            method_kwargs = {}
            reverse_method_kwargs = {}
        else:
            method_kwargs = [{}, {}]
            reverse_method_kwargs = [{}, {}]

        check_rows = False
        check_columns = False
        if 0 in input_axes:
            check_rows = True
            weights_rows = np.random.default_rng(0).normal(0.8, 0.05, self.x.size)
            weights_rows = np.clip(weights_rows, 0, 1, dtype=float)
            if scalar_axes:
                method_kwargs['weights'] = weights_rows
                reverse_method_kwargs['weights'] = weights_rows[::-1]
            else:
                method_kwargs[input_axes.index(0)]['weights'] = weights_rows
                reverse_method_kwargs[input_axes.index(0)]['weights'] = weights_rows[::-1]
        if 1 in input_axes:
            check_columns = True
            weights_cols = np.random.default_rng(0).normal(0.8, 0.05, self.z.size)
            weights_cols = np.clip(weights_cols, 0, 1, dtype=float)
            if scalar_axes:
                method_kwargs['weights'] = weights_cols
                reverse_method_kwargs['weights'] = weights_cols[::-1]
            else:
                method_kwargs[input_axes.index(1)]['weights'] = weights_cols
                reverse_method_kwargs[input_axes.index(1)]['weights'] = weights_cols[::-1]

        reverse_fitter = self.algorithm_base(self.x[::-1], self.z[::-1], assume_sorted=False)

        regular_output, regular_output_params = self.class_func(
            data=self.y, axes=axes, method_kwargs=method_kwargs, **self.kwargs
        )
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), axes=axes, method_kwargs=reverse_method_kwargs,
            **self.kwargs
        )

        # use relatively low tolerances since the fits are independent of each other
        # so they are not guaranteed to be the same
        for key in self.weight_keys:
            if check_rows:
                assert_allclose(
                    regular_output_params['params_rows'][key],
                    self.reverse_array(reverse_output_params['params_rows'][key]),
                    rtol=1e-6, atol=1e-6
                )
            if check_columns:
                assert_allclose(
                    regular_output_params['params_columns'][key],
                    self.reverse_array(reverse_output_params['params_columns'][key]),
                    rtol=1e-6, atol=1e-6
                )

        assert_allclose(
            regular_output, self.reverse_array(reverse_output), rtol=1e-6, atol=1e-6
        )

    def test_method_kwargs_too_large_fails(self):
        """Ensures an error is raised if method_kwargs has a size large than the number of axes."""
        # ensure it works correctly with length 1 method_kwargs
        self.class_func(data=self.y, axes=0, method_kwargs=[{}])
        with pytest.raises(ValueError, match='Method kwargs must have the same length'):
            self.class_func(data=self.y, axes=0, method_kwargs=[{}, {}])
        with pytest.raises(ValueError, match='Method kwargs must have the same length'):
            self.class_func(data=self.y, axes=[0], method_kwargs=[{}, {}])
        with pytest.raises(ValueError, match='Method kwargs must have the same length'):
            self.class_func(data=self.y, axes=[0, 1], method_kwargs=[{}, {}, {}])

    def test_repeated_axis_fails(self):
        """Ensures an error is raised if axes are repreated."""
        with pytest.raises(ValueError, match='Fitting the same axis twice'):
            self.class_func(data=self.y, axes=[0, 0])
        with pytest.raises(ValueError, match='Fitting the same axis twice'):
            self.class_func(data=self.y, axes=[1, 1])
