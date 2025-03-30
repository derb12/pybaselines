# -*- coding: utf-8 -*-
"""Tests for pybaselines.optimizers.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import Baseline, optimizers, polynomial, utils

from .base_tests import BaseTester, InputWeightsMixin, ensure_deprecation


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
    algorithm_base = optimizers._Optimizers
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
    checked_method_keys = ('weights', 'tol_history')
    two_d = True
    weight_keys = ('average_weights', 'weights')

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


class TestOptimizeExtendedRange(OptimizersTester, OptimizerInputWeightsMixin):
    """Class for testing collab_pls baseline."""

    func_name = "optimize_extended_range"
    checked_keys = ('optimal_parameter', 'min_rmse', 'rmse')
    # will need to change checked_keys if default method is changed
    checked_method_keys = ('weights', 'tol_history')
    required_kwargs = {'pad_kwargs': {'extrapolate_window': 100}}

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_unchanged_data(self, new_instance, side):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(new_instance, side=side)

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_input_weights(self, side):
        """Ensures arrays are correctly sorted within the function."""
        super().test_input_weights(side=side)

    @pytest.mark.parametrize(
        'method',
        (
            'asls', 'iasls', 'airpls', 'mpls', 'arpls', 'drpls', 'iarpls', 'aspls', 'psalsa',
            'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'quant_reg', 'goldindec',
            'derpsalsa', 'mpspline', 'mixture_model', 'irsqr', 'dietrich', 'cwt_br', 'fabc',
            'pspline_asls', 'pspline_iasls', 'pspline_airpls', 'pspline_arpls', 'pspline_drpls',
            'pspline_iarpls', 'pspline_aspls', 'pspline_psalsa', 'pspline_derpsalsa'
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
        if 'weights' in output[1]['method_params']:
            assert self.y.shape == output[1]['method_params']['weights'].shape
        elif 'alpha' in output[1]['method_params']:
            assert self.y.shape == output[1]['method_params']['alpha'].shape

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
            self.class_func(self.y, method='asls', **{key: 16})

    @pytest.mark.parametrize('side', ('left', 'right', 'both'))
    def test_aspls_alpha_ordering(self, side):
        """Ensures the `alpha` array for the aspls method is currectly processed."""
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
    def test_min_max_ordering(self, method):
        """Ensures variable ordering handles min and max values correctly."""
        min_value = 2
        max_value = 6
        fit_1, params_1 = self.class_func(
            self.y, method=method, min_value=min_value, max_value=max_value
        )
        # should simply do the fittings in the reversed order
        fit_2, params_2 = self.class_func(
            self.y, method=method, min_value=max_value, max_value=min_value
        )

        # fits and optimal parameter should be the same
        assert_allclose(fit_2, fit_1, rtol=1e-12, atol=1e-12)
        assert_allclose(
            params_1['optimal_parameter'], params_2['optimal_parameter'], rtol=1e-12, atol=1e-12
        )
        # rmse should be reversed
        assert_allclose(params_1['rmse'], params_2['rmse'][::-1], rtol=1e-8, atol=1e-12)

    @pytest.mark.parametrize('method', ('asls', 'modpoly'))
    def test_no_step(self, method):
        """Ensures a fit is still done if step is zero or min and max values are equal."""
        min_value = 2
        # case 1: step == 0
        fit_1, params_1 = self.class_func(
            self.y, method=method, min_value=min_value, max_value=min_value + 5, step=0
        )
        # case 2: min and max value are equal
        fit_2, params_2 = self.class_func(
            self.y, method=method, min_value=min_value, max_value=min_value
        )

        # fits, optimal parameter, and rmse should all be the same
        assert_allclose(fit_2, fit_1, rtol=1e-12, atol=1e-12)
        assert_allclose(
            params_1['optimal_parameter'], params_2['optimal_parameter'], rtol=1e-12, atol=1e-12
        )
        assert_allclose(params_1['rmse'], params_2['rmse'], rtol=1e-8, atol=1e-12)
        assert len(params_1['rmse']) == 1
        assert len(params_2['rmse']) == 1


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
        y + true_baseline, poly_order=1, weights=None, fit_function=fitter.modpoly
    )

    assert_array_equal(output_orders, expected_orders)


class TestAdaptiveMinMax(OptimizersTester, InputWeightsMixin):
    """Class for testing adaptive_minmax baseline."""

    func_name = 'adaptive_minmax'
    checked_keys = ('weights', 'constrained_weights', 'poly_order')
    checked_method_keys = ('weights', 'tol_history')
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
    checked_method_keys = ('weights', 'tol_history')
    required_kwargs = {'sampling': 5}

    @pytest.mark.parametrize(
        'method',
        (
            'poly', 'modpoly', 'imodpoly', 'penalized_poly', 'loess', 'asls', 'airpls', 'arpls',
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
