# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import Baseline, optimizers, polynomial, utils

from .base_tests import (
    BaseTester, InputWeightsMixin, ensure_deprecation, gaussian_alt, snr_to_sigma
)


class OptimizerInputWeightsMixin(InputWeightsMixin):
    """Passes weights within the `method_kwargs` dictionary."""

    def test_input_weights(self, assertion_kwargs=None, **kwargs):
        """
        Ensures arrays are correctly sorted within the function.

        Returns the output for further testing.

        """
        weights = np.random.default_rng(0).normal(0.8, 0.05, len(self.x))
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
            if key in regular_output_params:
                assert_allclose(
                    regular_output_params[key],
                    self.reverse_array(reverse_output_params[key]),
                    **assertion_kwargs
                )
            else:
                assert_allclose(
                    regular_output_params['method_params'][key],
                    self.reverse_array(reverse_output_params['method_params'][key]),
                    **assertion_kwargs
                )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), **assertion_kwargs
        )


class OptimizersTester(BaseTester):
    """Base testing class for optimizer functions."""

    module = optimizers
    checked_method_keys = None

    def test_output(self, additional_keys=None, additional_method_keys=None, **kwargs):
        """Ensures the keys within the method_params dictionary are also checked."""
        if additional_keys is None:
            added_keys = ['method_params']
        else:
            added_keys = list(additional_keys) + ['method_params']
        if additional_method_keys is None:
            optimizer_keys = self.checked_method_keys
        elif self.checked_method_keys is None:
            optimizer_keys = additional_method_keys
        else:
            optimizer_keys = list(self.checked_method_keys) + list(additional_method_keys)
        super().test_output(
            additional_keys=added_keys, optimizer_keys=optimizer_keys, **kwargs
        )


class TestCollabPLS(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing collab_pls baseline."""

    func_name = "collab_pls"
    checked_keys = ('average_weights',)
    # will need to change checked_keys if default method is changed
    checked_method_keys = ('weights', 'tol_history', 'result', 'success')
    two_d = True
    weight_keys = ('average_weights', 'weights')
    supports_mask = True

    @ensure_deprecation(1, 5)  # remove the warnings filter after version 1.5
    @pytest.mark.filterwarnings('ignore:"pspline_mpls" is deprecated')
    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'fabc', 'pspline_asls',
            'pspline_iasls', 'pspline_airpls', 'pspline_arpls', 'pspline_drpls',
            'pspline_iarpls', 'pspline_aspls', 'pspline_psalsa', 'pspline_derpsalsa',
            'brpls', 'pspline_brpls', 'pspline_mpls', 'lsrpls', 'pspline_lsrpls'
        )
    )
    def test_all_methods(self, method):
        """Ensures all available methods work."""
        self.class_func(self.y, method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown function')

    @pytest.mark.parametrize('method', ('mor', 'rolling_ball', 'snip', 'beads'))
    def test_disallowed_method_fails(self, method):
        """Ensures function fails when a method that does not work is given."""
        with pytest.raises(ValueError, match=f'{method} is not a supported method'):
            self.class_func(self.y, method=method)

    def test_single_dataset_fails(self):
        """Ensures an error is raised if the input has the shape (N,)."""
        with pytest.raises(ValueError, match='the input data must'):
            self.class_func(np.arange(self.x.shape[0]))

    @pytest.mark.parametrize('average_dataset', (True, False))
    def test_input_weights(self, average_dataset):
        """Ensures the input weights are sorted correctly."""
        super().test_input_weights(average_dataset=average_dataset)

    @pytest.mark.parametrize('average_dataset', (True, False))
    def test_output_alpha(self, average_dataset):
        """Ensures the output alpha values are sorted correctly when using aspls."""
        regular_output, regular_output_params = self.class_func(
            data=self.y, average_dataset=average_dataset, method='aspls',
        )
        reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=False)
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), average_dataset=average_dataset, method='aspls',
        )

        assert_allclose(
            regular_output_params['method_params']['alpha'],
            self.reverse_array(reverse_output_params['method_params']['alpha']),
            rtol=1e-12, atol=1e-14
        )

    @pytest.mark.parametrize('average_dataset', (True, False))
    @pytest.mark.parametrize('use_mask', (True, False))
    def test_replicate_data(self, average_dataset, use_mask):
        """Ensures logic within collab_pls by fitting several of the same data.

        Whether average_dataset is True or False, fitting repeats of the same data should
        result in collab_pls producing the same fit as simply calling the underlying method.

        Note that TestCollabPLS technically already uses repeated y-datasets for self.y, but set
        the repeated y in this test in case the setup ever changes.

        """
        repeats = 5
        multi_y = np.repeat(self.y[0][None, :], repeats, axis=0)
        if use_mask:
            mask = np.zeros_like(self.y[0], dtype=bool)
            mask[5:30] = True
        else:
            mask = None
        fitter = self.algorithm_base(mask=mask)

        fit, params = fitter.collab_pls(
            multi_y, average_dataset=average_dataset, method='asls'
        )
        fit_single, params_single = fitter.asls(self.y[0])

        assert_allclose(
            fit, np.repeat(fit_single[None, :], repeats, axis=0), atol=1e-15, rtol=1e-15
        )
        assert_allclose(
            params['average_weights'], params_single['weights'], atol=1e-15, rtol=1e-15
        )
        assert_allclose(
            params['method_params']['weights'],
            np.repeat(params_single['weights'][None, :], repeats, axis=0),
            atol=1e-15, rtol=1e-15
        )


class TestOptimizeExtendedRange(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing optimize_extended_range baseline."""

    func_name = "optimize_extended_range"
    checked_keys = ('optimal_parameter', 'min_rmse', 'rmse')
    # will need to change checked_keys if default method is changed
    checked_method_keys = ('weights', 'tol_history', 'result', 'success')
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
            'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg', 'goldindec',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'dietrich', 'cwt_br', 'fabc',
            'pspline_asls', 'pspline_iasls', 'pspline_airpls', 'pspline_arpls', 'pspline_drpls',
            'pspline_iarpls', 'pspline_aspls', 'pspline_psalsa', 'pspline_derpsalsa', 'rubberband',
            'beads', 'jbcd',
        )
    )
    def test_all_methods(self, method):
        """Tests all methods that should work with optimize_extended_range."""
        # reduce number of calculations since this is just checking that calling works
        kwargs = {'min_value': 1, 'max_value': 3}
        # use height_scale=0.1 to avoid exponential overflow warning for arpls and aspls
        output, params = self.class_func(
            self.y, method=method, height_scale=0.1, **kwargs, **self.kwargs
        )
        for key in ('weights', 'alpha', 'signal', 'mask', 'opening'):
            if key in params['method_params']:
                assert self.y.shape == params['method_params'][key].shape

        # more general check than above in case other methods add array-like keys later
        for key, value in params['method_params'].items():
            if isinstance(value, np.ndarray) and key != 'tol_history':
                assert self.y.shape == value.shape

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown function')

    @pytest.mark.parametrize('method', ('mor', 'rolling_ball', 'snip'))
    def test_disallowed_method_fails(self, method):
        """Ensures function fails when a method that does not work is given."""
        with pytest.raises(ValueError, match=f'{method} is not a supported method'):
            self.class_func(self.y, method=method)

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
        since the user probably thought the actual lam value had to be specifiied rather than
        just the exponent.

        """
        with pytest.raises(ValueError):
            self.class_func(self.y, method='asls', **{key: 16})

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_aspls_alpha_ordering(self, side):
        """Ensures the `alpha` array for the aspls method is correctly processed."""
        alpha = np.random.default_rng(0).normal(0.8, 0.05, len(self.x))
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
                regular_output_params['method_params'][key],
                reverse_output_params['method_params'][key][::-1],
                rtol=1e-10, atol=1e-14
            )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), rtol=1e-10, atol=1e-14
        )

    def test_kwargs_raises(self):
        """Ensures an exception is raised for passing kwargs meant for the fitting function."""
        with pytest.raises(TypeError):
            self.class_func(self.y, method='asls', lam=1e8)

    @ensure_deprecation(1, 4)
    def test_min_rmse_deprecation(self):
        """Placeholder to ensure 'min_rmse' is removed from the output in version 1.4."""

    def test_optimal_parameter(self):
        """Ensures the output optimal parameter is the correct value.

        For polynomial methods, `optimal_parameter` should be the polynomial degree; for
        other methods, `optimal_parameter` should be the actual `lam` value, not log(lam)
        as returned in versions earlier than 1.2.0.
        """
        min_value = 2
        _, params = self.class_func(self.y, method='asls', min_value=min_value, max_value=8)
        assert params['optimal_parameter'] >= 10**min_value

        max_value = 6
        _, params2 = self.class_func(self.y, method='modpoly', min_value=2, max_value=max_value)
        assert params2['optimal_parameter'] <= max_value

    @pytest.mark.parametrize('method', ('asls', 'modpoly'))
    def test_no_step(self, method):
        """Ensures a fit is still done if step is zero or min and max values are equal."""
        min_value = 2
        # case 1: step == 0
        with pytest.warns(utils.ParameterWarning):
            fit_1, params_1 = self.class_func(
                self.y, method=method, min_value=min_value, max_value=min_value + 5, step=0
            )
        # case 2: min and max value are equal
        with pytest.warns(utils.ParameterWarning):
            fit_2, params_2 = self.class_func(
                self.y, method=method, min_value=min_value, max_value=min_value
            )
        # case 3: step is too large
        with pytest.warns(utils.ParameterWarning):
            fit_3, params_3 = self.class_func(
                self.y, method=method, min_value=min_value, max_value=min_value + 1, step=5
            )

        # fits, optimal parameter, and rmse should all be the same
        assert_allclose(fit_2, fit_1, rtol=1e-12, atol=1e-12)
        assert_allclose(fit_3, fit_1, rtol=1e-12, atol=1e-12)
        assert_allclose(
            params_2['optimal_parameter'], params_1['optimal_parameter'], rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            params_3['optimal_parameter'], params_1['optimal_parameter'], rtol=1e-12, atol=1e-12
        )
        assert_allclose(params_2['rmse'], params_1['rmse'], rtol=1e-8, atol=1e-12)
        assert_allclose(params_3['rmse'], params_1['rmse'], rtol=1e-8, atol=1e-12)
        assert len(params_1['rmse']) == 1
        assert len(params_2['rmse']) == 1
        assert len(params_3['rmse']) == 1

    def test_value_range(self):
        """Ensures the correct number of parameters to fit are generated."""
        min_value = 2
        max_value = 6
        for step in (1, 2, 3):
            expected_tested_values = np.arange(min_value, max_value, step)
            _, params_1 = self.class_func(
                self.y, method='modpoly', min_value=min_value, max_value=max_value, step=step
            )
            _, params_2 = self.class_func(
                self.y, method='asls', min_value=min_value, max_value=max_value, step=step
            )
            # both methods should have the same number of tested values
            assert len(params_1['rmse']) == len(expected_tested_values)
            assert len(params_2['rmse']) == len(expected_tested_values)

        # also test float inputs for lam-based methods
        min_value = 1.
        max_value = 5.5
        step = 0.5
        expected_tested_values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        _, params_3 = self.class_func(
            self.y, method='asls', min_value=min_value, max_value=max_value, step=step
        )
        # both methods should have the same number of tested values
        assert len(params_3['rmse']) == len(expected_tested_values)

    @pytest.mark.parametrize('method', ('asls', 'modpoly'))
    def test_default_step(self, method):
        """Ensures default step is determined by the type of baseline method."""
        min_value = 1
        max_value = 5
        expected_tested_values = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        expected_step = 0.5 if method == 'asls' else 1
        expected_tested_values = np.arange(min_value, max_value, expected_step)

        _, params = self.class_func(
            self.y, method=method, min_value=min_value, max_value=max_value, step=None
        )
        assert len(params['rmse']) == len(expected_tested_values)

    def test_correctness(self):
        """Compares the calculated optimal value to its literature example.

        Data is based on Figure 3 of Zhang, et al., and the lam vs RMSE plot should
        follow Figure 4.

        References
        ----------
        Zhang, F., et al. An Automatic Baseline Correction Method Based on
        the Penalized Least Squares Method. Sensors, 2020, 20(7), 2015.

        """
        x = np.linspace(0, 1200, 1200)
        peaks = (
            gaussian_alt(x, 2, 100, 20)
            + gaussian_alt(x, 1, 200, 20)
            + gaussian_alt(x, 2, 400, 40)
            + gaussian_alt(x, 1, 500, 30)
            + gaussian_alt(x, 4, 800, 50)
            + gaussian_alt(x, 0.5, 1000, 15)
            + gaussian_alt(x, 1, 1100, 20)
        )
        baseline = np.sin(np.pi * x / 1200)

        noise = np.random.default_rng(0).normal(0, snr_to_sigma(30, peaks), x.shape)
        y = peaks + baseline + noise

        fit, params = self.algorithm_base().optimize_extended_range(
            y, method='aspls', side='right', min_value=3, max_value=12.2, step=0.3, width_scale=0.2,
            pad_kwargs={'extrapolate_window': 50},
        )
        assert_allclose(np.log10(params['optimal_parameter']), 9.6, rtol=0, atol=0.05)


def test_param_grid():
    """Ensures basic functionality of _param_grid."""
    min_value = 1
    max_value = 5
    step = 1

    expected_values = np.arange(min_value, max_value, step)
    output = optimizers._param_grid(min_value, max_value, step, polynomial_fit=True)

    assert_array_equal(output, expected_values)

    output2 = optimizers._param_grid(min_value, max_value, step, polynomial_fit=False)
    assert_allclose(output2, 10**expected_values, rtol=1e-15, atol=1e-15)

    # also ensure floats are properly handled
    step = 0.5
    expected_values = 10**np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
    output3 = optimizers._param_grid(min_value, max_value, step, polynomial_fit=False)
    assert_allclose(output3, expected_values, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('polynomial_fit', (True, False))
def test_param_grid_no_step(polynomial_fit):
    """Ensures a single param is still output if step is zero or min and max values are equal."""
    min_value = 2
    expected_value = np.array([min_value])
    if not polynomial_fit:
        expected_value = 10.0**expected_value

    # case 1: step == 0
    with pytest.warns(utils.ParameterWarning):
        output1 = optimizers._param_grid(
            min_value=min_value, max_value=min_value + 5, step=0, polynomial_fit=polynomial_fit
        )
    # case 2: min and max value are equal
    with pytest.warns(utils.ParameterWarning):
        output2 = optimizers._param_grid(
            min_value=min_value, max_value=min_value, step=1, polynomial_fit=polynomial_fit
        )
    # case 3: step is too large
    with pytest.warns(utils.ParameterWarning):
        output3 = optimizers._param_grid(
            min_value=min_value, max_value=min_value + 1, step=5, polynomial_fit=polynomial_fit
        )

    assert_allclose(output1, expected_value, rtol=1e-15, atol=1e-15)
    assert_allclose(output2, expected_value, rtol=1e-15, atol=1e-15)
    assert_allclose(output3, expected_value, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
def test_param_grid_float_poly_fails(key):
    """Ensures non-integer values raise an error for polynomial fits."""
    if key == 'min_value':
        kwargs = {'max_value': 5, 'step': 1}
    elif key == 'max_value':
        kwargs = {'min_value': 1, 'step': 1}
    else:
        kwargs = {'min_value': 1, 'max_value': 5}
    kwargs[key] = 2.5
    with pytest.raises(TypeError):
        optimizers._param_grid(**kwargs, polynomial_fit=True)


@pytest.mark.parametrize('key', ('min_value', 'max_value', 'step'))
def test_param_grid_nonpoly_fails(key):
    """
    Ensures function fails when using a Whittaker method and input lambda exponent is too high.

    Since the function uses 10**exponent, do not want to allow a high exponent to be used,
    since the user probably thought the actual lam value had to be specifiied rather than
    just the exponent.

    """
    if key == 'min_value':
        kwargs = {'max_value': 5, 'step': 1}
    elif key == 'max_value':
        kwargs = {'min_value': 1, 'step': 1}
    else:
        kwargs = {'min_value': 1, 'max_value': 5}
    kwargs[key] = 16
    with pytest.raises(ValueError):
        optimizers._param_grid(**kwargs, polynomial_fit=False)


@pytest.mark.parametrize('polynomial_fit', (True, False))
def test_param_grid_bounds_errors(polynomial_fit):
    """Ensures a step < 0 or min_value > max_value both raise exceptions."""
    with pytest.raises(ValueError):
        optimizers._param_grid(min_value=1, max_value=5, step=-1, polynomial_fit=polynomial_fit)

    with pytest.raises(ValueError):
        optimizers._param_grid(min_value=10, max_value=5, step=1, polynomial_fit=polynomial_fit)

    with pytest.raises(ValueError):
        optimizers._param_grid(min_value=10, max_value=5, step=-1, polynomial_fit=polynomial_fit)


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

    fitter = polynomial._Polynomial(x, check_finite=False, assume_sorted=True)

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
        y + true_baseline, poly_order=1, weights=None, fit_function=fitter.modpoly,
        fitter=fitter
    )

    assert_array_equal(output_orders, expected_orders)


class TestAdaptiveMinMax(OptimizersTester, InputWeightsMixin):
    """Class for testing adaptive_minmax baseline."""

    func_name = 'adaptive_minmax'
    checked_keys = ('poly_order',)
    checked_method_keys = ('weights', 'tol_history', 'success')
    weight_keys = ()
    supports_mask = True

    @pytest.mark.parametrize('method', ('modpoly', 'imodpoly'))
    def test_methods(self, method):
        """Ensures all available methods work."""
        self.class_func(self.y, method=method)

    def test_unknown_method_fails(self):
        """Ensures function fails when an unknown function is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='unknown')

    @pytest.mark.parametrize('method', ('mor', 'rolling_ball', 'snip', 'arpls', 'mixture_model'))
    def test_disallowed_method_fails(self, method):
        """Ensures function fails when a method that does not work is given."""
        with pytest.raises(ValueError, match=f'{method} is not a supported method'):
            self.class_func(self.y, method=method)

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

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures the polynomial coefficients are output if `return_coef` is True."""
        if return_coef:
            additional_method_keys = ['coef']
        else:
            additional_method_keys = None
        super().test_output(
            additional_method_keys=additional_method_keys,
            method_kwargs={'return_coef': return_coef}
        )


class TestCustomBC(OptimizersTester):
    """Class for testing custom_bc baseline."""

    func_name = 'custom_bc'
    checked_keys = ('y_fit', 'x_fit', 'baseline_fit')
    # will need to change checked_keys if default method is changed
    checked_method_keys = ('weights', 'tol_history', 'result', 'success')
    required_kwargs = {'sampling': 5}

    @pytest.mark.parametrize(
        'method',
        (
            'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'asls', 'airpls', 'arpls',
            'mpls', 'mor', 'imor', 'mixture_model', 'irsqr', 'corner_cutting', 'pspline_asls',
            'pspline_airpls', 'noise_median', 'snip', 'dietrich', 'std_distribution', 'fabc'
        )
    )
    def test_methods(self, method):
        """
        Ensures most available methods work.

        Does not test all methods since the function can be used for all methods within
        pybaselines; instead, it just tests a few methods from each module.

        """
        self.class_func(self.y, method=method)

    def test_x_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        super().test_x_ordering(assertion_kwargs={'rtol': 1e-6})

    @pytest.mark.parametrize('lam', (None, 1))
    def test_output_smoothing(self, lam):
        """Ensures the smoothing is done properly if specified."""
        diff_order = 2
        output, params = self.class_func(self.y, method='asls', lam=lam, diff_order=diff_order)

        truncated_baseline = Baseline(params['x_fit']).asls(params['y_fit'])[0]
        expected_baseline = np.interp(self.x, params['x_fit'], truncated_baseline)
        if lam is not None:
            expected_baseline = utils.whittaker_smooth(
                expected_baseline, lam=lam, diff_order=diff_order
            )

        assert_allclose(output, expected_baseline, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize('roi_and_samplings', (
        [((None, None),), 5],
        [((None, None),), 1],
        [((None, None),), 10000000],
        [((0, 20), (20, 30)), (3, 2)],
        [((0, 1), (20, 30)), (3, 2)],
        [((0, 20), (20, 30)), (33,)],
        [((0, 20), (20, 30), (30, None)), (33, 5, 50)],
    ))
    def test_unique_x(self, roi_and_samplings):
        """Ensures the fit uses only unique values and that x and y match dimensions."""
        regions, sampling = roi_and_samplings
        output, params = self.class_func(self.y, regions=regions, sampling=sampling)

        assert_allclose(params['x_fit'], np.unique(params['x_fit']), rtol=1e-12, atol=1e-14)
        assert params['x_fit'].shape == params['y_fit'].shape
        assert len(params['x_fit']) > 2  # should at least include first, middle, and last values

    def test_roi_sampling_mixmatch_fails(self):
        """Ensures an exception is raised if regions and sampling do not have the same shape."""
        with pytest.raises(ValueError):
            self.class_func(self.y, regions=((None, None),), sampling=[1, 2])
        with pytest.raises(ValueError):
            self.class_func(self.y, regions=((None, 10), (20, 30), (30, 40)), sampling=[1, 2])

    @pytest.mark.parametrize('sampling', (-1, [-1], [5, -5]))
    def test_negative_sampling_fails(self, sampling):
        """Ensures an exception is raised if sampling is negative."""
        if isinstance(sampling, int):
            num_samplings = 1
        else:
            num_samplings = len(sampling)
        regions = []
        for i in range(num_samplings):
            regions.append([i * 10, (i + 1) * 10])
        with pytest.raises(ValueError):
            self.class_func(self.y, regions=regions, sampling=sampling)

    @pytest.mark.parametrize('regions', (((-1, 5),), ((0, 10), (20, -30)), ((0, 10000),)))
    def test_bad_region_values_fails(self, regions):
        """Ensures an exception is raised if regions has a negative value or a too large value."""
        with pytest.raises(ValueError):
            self.class_func(self.y, regions=regions)

    def test_overlapping_regions_fails(self):
        """Ensures an exception is raised if regions overlap."""
        with pytest.raises(ValueError):
            self.class_func(self.y, regions=((0, 10), (9, 13)))


class TestOptimizePLS(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing optimize_pls baseline."""

    func_name = "optimize_pls"
    checked_keys = ('optimal_parameter', 'metric')
    # will need to change checked_keys if default method is changed
    checked_method_keys = ('weights', 'tol_history', 'result', 'success')
    # by default only run a few optimization steps
    required_kwargs = {'min_value': 2, 'max_value': 3}

    @pytest.mark.parametrize('opt_method', ('V-curve', 'U-curve', 'GCV', 'BIC'))
    def test_output(self, opt_method):
        """Ensures correct output parameters for different optimization methods."""
        if opt_method in ('GCV', 'BIC'):
            additional_keys = ['trace', 'wrss']
        else:
            additional_keys = ['penalty', 'fidelity']
        super().test_output(additional_keys=additional_keys, opt_method=opt_method)

    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'fabc', 'rubberband',
            'pspline_asls', 'pspline_iasls', 'pspline_airpls', 'pspline_arpls', 'pspline_drpls',
            'pspline_iarpls', 'pspline_aspls', 'pspline_psalsa', 'pspline_derpsalsa'
        )
    )
    @pytest.mark.parametrize('opt_method', ('U-Curve', 'GCV'))
    def test_all_methods(self, method, opt_method):
        """Tests most methods that should work with optimize_pls."""
        output = self.class_func(self.y, method=method, opt_method=opt_method, **self.kwargs)
        if 'weights' in output[1]['method_params']:
            assert self.y.shape == output[1]['method_params']['weights'].shape
        elif 'alpha' in output[1]['method_params']:
            assert self.y.shape == output[1]['method_params']['alpha'].shape

    @pytest.mark.parametrize('opt_method', ('V-curve', 'U-Curve', 'GCV', 'BIC'))
    def test_beads(self, opt_method):
        """Ensures beads is also supported for L-curve based optimization methods."""
        if opt_method in ('GCV', 'BIC'):
            with pytest.raises(
                NotImplementedError, match='optimize_pls does not support the beads method'
            ):
                self.class_func(self.y, method='beads', opt_method=opt_method, **self.kwargs)
        else:
            # just ensure calling does not produce errors
            self.class_func(self.y, method='beads', opt_method=opt_method, **self.kwargs)

    def test_unknown_method_fails(self):
        """Ensures method fails when an unknown baseline method is given."""
        with pytest.raises(AttributeError):
            self.class_func(self.y, method='aaaaa')

    @pytest.mark.parametrize('method', ('mor', 'rolling_ball', 'snip'))
    def test_disallowed_method_fails(self, method):
        """Ensures function fails when a method that does not work is given."""
        with pytest.raises(ValueError, match=f'{method} is not a supported method'):
            self.class_func(self.y, method=method)

    def test_unknown_opt_method_fails(self):
        """Ensures method fails when an unknown opt_method is given."""
        with pytest.raises(ValueError):
            self.class_func(self.y, opt_method='aaaaa')

    @pytest.mark.parametrize('opt_method', ('V-Curve', 'U-Curve', 'GCV', 'BIC'))
    def test_single_value(self, opt_method):
        """Ensures all optimization methods work if only a single value is fit."""
        min_val = 2.
        with pytest.warns(utils.ParameterWarning, match='min_value, max_value, and step'):
            fit, params = self.class_func(
                self.y, method='asls', opt_method=opt_method, min_value=min_val, step=0
            )
        if opt_method in ('GCV', 'BIC'):
            additional_keys = ['trace', 'wrss']
        else:
            additional_keys = ['penalty', 'fidelity']
        for key in ['metric'] + additional_keys:
            assert key in params
            value = params[key]
            assert isinstance(value, np.ndarray)
            assert value.shape == (1,)
        assert isinstance(params['optimal_parameter'], float)

        # should be same as just fitting the minimum value
        single_fit, _ = self.algorithm_base().asls(self.y, lam=10.**min_val)
        assert_allclose(fit, single_fit, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('pspline', (True, False))
    def test_vcurve(self, pspline):
        """
        Tests the V-curve metric against literature.

        Data is based on Figure 3 from the reference; some of Frasso's and Eilers's
        other V-curve/L-curve publications also have reference L-curves, but Frasso's
        thesis had the easiest to reproduce figures.

        Note that within Figure 3, the optimal lam values is stated to be ~7943.28 (10**3.9),
        but from the metric plots, the optimum is ~1e4. Step size affects this, so just allow
        an atol of 0.15 when comparing the optimal lam. Also `log` within the thesis refers
        to `log10` rather than the natural log.

        References
        ----------
        Frasso, G. Smoothing parameter selection using the L-curve. 2012, Leiden University,
        MS Thesis. https://hdl.handle.net/1887/3597367.

        """
        x = np.linspace(0, 2 * np.pi, 200)
        signal = 5 * np.sin(x)
        noise = np.random.default_rng(0).normal(0, 0.5, x.size)
        y = signal + noise
        step = 0.1
        kwargs = {'tol': np.inf}
        if pspline:
            # paper only used Whittaker smoothing, so set P-spline up so that it's equivalent to
            # Whittaker smoothing
            method = 'pspline_asls'
            kwargs.update({'num_knots': len(y), 'spline_degree': 1})
        else:
            method = 'asls'

        fit, params = self.algorithm_base().optimize_pls(
            y, method=method, opt_method='V-curve', min_value=0, max_value=8, step=step,
            method_kwargs=kwargs
        )

        assert_allclose(np.log10(params['optimal_parameter']), 4, rtol=0, atol=0.15)
        assert_allclose(fit, signal, rtol=1e-3, atol=0.3)

        # simple tests for bounds of the L-curve based on Fig. 3b
        log_penalty = np.log10(params['penalty'])
        log_fidelity = np.log10(params['fidelity'])
        assert log_penalty.min() > -7.5
        assert log_penalty.max() < 0.9
        assert log_fidelity.min() > 1.3
        assert log_fidelity.max() < 3.1

    @pytest.mark.parametrize('baseline_type', (0, 1, 2))
    @pytest.mark.parametrize('pspline', (True, False))
    def test_ucurve(self, pspline, baseline_type):
        """
        Tests the U-curve metric against literature.

        Data is expected to follow Figure 5 from Park, et al.

        References
        ----------
        Park, A., et al. Automatic Selection of Optimal Parameter for Baseline Correction
        using Asymmetrically Reweighted Penalized Least Squares. Journal of the Institute
        of Electronics and Information Engineers, 2016, 53(3), 124-131.

        """
        x = np.linspace(1, 1000, 1000)
        signal = (
            utils.gaussian(x, 200, 200, 5)
            + utils.gaussian(x, 200, 400, 20)
            + utils.gaussian(x, 400, 800, 10)
        )
        if baseline_type == 0:
            baseline = utils.gaussian(x, x + 100, 0, 1200)
            # taken as minimum point from Fig. 5(a); Table 1 suggests that lam_opt=7, but
            # that's based on RMSE with the fit, not based on the calculated metric
            lam_opt = 6.5
        elif baseline_type == 1:
            baseline = utils.gaussian(x, 1000, 600, 400)
            lam_opt = 7
        else:
            baseline = utils.gaussian(x, 800, 100, 300) + utils.gaussian(x, 1000, 900, 300)
            lam_opt = 6

        noise = np.random.default_rng(0).normal(0, snr_to_sigma(10, signal), x.size)
        y = signal + baseline + noise
        kwargs = {}
        if pspline:
            # paper only used Whittaker smoothing, so set P-spline up so that it's equivalent to
            # Whittaker smoothing
            method = 'pspline_arpls'
            kwargs.update({'num_knots': len(y), 'spline_degree': 1})
        else:
            method = 'arpls'

        fit, params = self.algorithm_base().optimize_pls(
            y, method=method, opt_method='U-curve', min_value=3, max_value=12.4, step=0.5,
            method_kwargs=kwargs
        )

        assert_allclose(np.log10(params['optimal_parameter']), lam_opt, rtol=0, atol=0.1)

        # not universal, but for the tested baselines, the metric never exceeds 1 (see Fig. 5)
        assert params['metric'].min() >= 0
        assert params['metric'].max() <= 1

    @pytest.mark.parametrize('pspline', (True, False))
    @pytest.mark.parametrize('weight_enum', (0, 1, 2))
    def test_gcv(self, weight_enum, pspline):
        """
        Compares against the 'WH' R package for ensuring GCV calculation is correct.

        The R code to generate the values are::

            library(WH)

            options(digits=14)
            file_path = r"(unquoted file path here)"
            y = read.csv(file_path, header=FALSE)$V1
            wts = rep(1, length(y))
            for(i in 0:2){
                if (i == 0) {
                    fill_value = 1
                } else if (i == 1) {
                    fill_value = 0
                } else {
                    fill_value = 0.5
                }
                wts[60:120] = fill_value
                fit = WH(y=y, wt=wts, criterion="GCV")
                # ref_lam, ref_metric, ref_edf, ref_wrss
                print(c(fit$lambda, fit$diagnosis$GCV, fit$diagnosis$sum_edf, fit$diagnosis$dev))
            }

        using WH version 2.0.0 and R version 4.2.3.

        Alternatively, could use the 'mgcv' R package, in which case the additional R code
        for fitting is::

            library(mgcv)
            x = 1:length(y)
            # note that knots must use same key naming scheme that the spline uses or
            # else the knots seem to be ignored
            fit2 = gam(y ~ s(x, bs="ps", k=length(y), m=c(0, 2)), family=gaussian(),
                       knots=list(x=0:(length(y) + 1)), method="GCV.Cp", weights=wts)
            # ref_lam, ref_metric, ref_edf, ref_wrss
            print(c(fit2$sp / 16, fit2$gcv.ubre.dev, sum(fit2$edf), fit2$deviance))

        Not sure why the need to divide mgcv's lambda by 16, but it's consistent with the
        same divergence observed in SciPy (https://github.com/scipy/scipy/pull/22580);
        within fit2$smooth, it does say S.scale is 16, but not sure how that's set or
        whether that's the actual cause... The fit lambda value from WH matches that of
        pybaselines, rather than needing the division by 16.

        Given the above weirdness of mgcv's reported smoothing parameter, rather than
        using `gam` to use GCV with P-Splines, just set it up so that P-Spline option
        replicates Whittaker smoothing. Note that for 1D, it still seemed off by a factor
        of 16 for P-Splines as well, but 2D fits were ... confusing, so just do the
        Whittaker emulation.

        The comparison tolerances have to be fairly large since pybaselines uses
        a grid-search rather than the scalar minimization used by the R packages.

        """
        x = np.linspace(0, 10 * np.pi, 50)
        y = np.sin(np.linspace(0, 10 * np.pi, 50)) + np.random.default_rng(0).normal(0, 0.3, len(x))

        wts = np.ones_like(y)
        if weight_enum == 0:
            fill_value = 1
            ref_lam = 0.249364010297557
            ref_metric = 0.082886220851521
            ref_edf = 29.138156309150428
            ref_wrss = 0.721469055515233
        elif weight_enum == 1:
            fill_value = 0
            ref_lam = 0.161447222423454
            ref_metric = 0.080461742310573
            ref_edf = 25.886100950162117
            ref_wrss = 0.354803992207319
        else:
            fill_value = 0.5
            ref_lam = 0.18222712865323
            ref_metric = 0.06887879138412
            ref_edf = 30.31931076759921
            ref_wrss = 0.53357579603285
        wts[9:20] = fill_value

        kwargs = {'tol': np.inf, 'weights': wts}
        if pspline:
            method = 'pspline_asls'
            kwargs.update({'num_knots': len(y), 'spline_degree': 1})
        else:
            method = 'asls'

        fit, params = self.algorithm_base().optimize_pls(
            y, method=method, opt_method='GCV', min_value=-2, max_value=1, step=0.01,
            method_kwargs=kwargs, rho=1
        )

        best_idx = params['metric'].argmin()
        assert_allclose(params['optimal_parameter'], ref_lam, rtol=1e-4, atol=0.005)
        assert_allclose(params['metric'][best_idx], ref_metric, rtol=1e-4, atol=1e-3)
        assert_allclose(params['trace'][best_idx], ref_edf, rtol=1e-2, atol=1e-1)
        assert_allclose(params['wrss'][best_idx], ref_wrss, rtol=1e-2, atol=1e-3)

        if weight_enum == 0:
            file_name = 'WH_diff2_lam1'
        else:
            file_name = f'WH_wt{weight_enum - 1}'
        expected_output = np.loadtxt(
            Path(__file__).parent.joinpath(f'data/{file_name}.csv'), delimiter=','
        )
        assert_allclose(fit, expected_output, rtol=5e-3, atol=1e-6)
