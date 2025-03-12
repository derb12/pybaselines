# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import polynomial
from pybaselines.utils import ParameterWarning

from .conftest import BasePolyTester, InputWeightsMixin
from .data import (
    LOESS_X, LOESS_Y, QUANTILE_Y, STATSMODELS_LOESS_DELTA, STATSMODELS_LOESS_ITER,
    STATSMODELS_QUANTILES
)


class PolynomialTester(BasePolyTester, InputWeightsMixin):
    """Base testing class for polynomial functions."""

    module = polynomial
    algorithm_base = polynomial._Polynomial
    checked_keys = ('weights',)


class IterativePolynomialTester(PolynomialTester):
    """Base testing class for iterative polynomial functions."""

    checked_keys = ('weights', 'tol_history')
    allows_zero_iteration = True  # whether max_iter=0 will return an initial baseline

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        if self.allows_zero_iteration:
            assert params['tol_history'].size == max_iter
        else:
            assert params['tol_history'].size == max_iter + 1


class TestPoly(PolynomialTester):
    """Class for testing regular polynomial baseline."""

    func_name = 'poly'


class TestModPoly(IterativePolynomialTester):
    """Class for testing modpoly baseline."""

    func_name = 'modpoly'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, use_class, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_original=use_original, mask_initial_peaks=mask_initial_peaks
        )


class TestIModPoly(IterativePolynomialTester):
    """Class for testing imodpoly baseline."""

    func_name = 'imodpoly'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, use_class, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_original=use_original, mask_initial_peaks=mask_initial_peaks
        )

    @pytest.mark.parametrize('num_std', (-1, -0.01, 0, 1))
    def test_negative_num_std_fails(self, num_std):
        """Ensures `num_std` values less than 0 raise an exception."""
        if num_std < 0:
            with pytest.raises(ValueError):
                self.class_func(self.y, num_std=num_std)
        else:
            self.class_func(self.y, num_std=num_std)


class TestPenalizedPoly(IterativePolynomialTester):
    """Class for testing penalized_poly baseline."""

    func_name = 'penalized_poly'

    @pytest.mark.parametrize('use_class', (True, False))
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
    def test_unchanged_data(self, use_class, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function='a_hub')

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
        penalized_poly_1 = self.class_func(
            self.y, poly_order, cost_function='s_huber',
            threshold=1e10, weights=weights
        )[0]

        assert_allclose(poly_baseline, penalized_poly_1, 1e-10)

        modpoly_baseline = polynomial.modpoly(
            self.y, self.x, poly_order, tol=tol, weights=weights, use_original=True
        )[0]
        penalized_poly_2 = self.class_func(
            self.y, poly_order, cost_function='a_truncated_quadratic',
            threshold=0, weights=weights, alpha_factor=1, tol=tol
        )[0]

        assert_allclose(modpoly_baseline, penalized_poly_2, 1e-10)

    @pytest.mark.parametrize('alpha_factor', (-0.1, 0, 1.01))
    def test_wrong_alpha_factor_fails(self, alpha_factor):
        """Ensures an alpha factor outside of (0, 1] fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha_factor=alpha_factor)


class TestLoess(IterativePolynomialTester):
    """Class for testing loess baseline."""

    func_name = 'loess'
    allows_zero_iteration = False
    requires_unique_x = True

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('delta', (0, 0.01))
    @pytest.mark.parametrize('conserve_memory', (True, False))
    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_unchanged_data(self, use_class, use_threshold, conserve_memory, delta):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_threshold=use_threshold,
            conserve_memory=conserve_memory, delta=delta
        )

    @pytest.mark.parametrize('use_threshold', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    def test_x_ordering(self, use_threshold, use_original):
        """Ensures arrays are correctly sorted within the function."""
        super().test_x_ordering(use_threshold=use_threshold, use_original=use_original)

    @pytest.mark.parametrize('fraction', (-0.1, 1.1, 5))
    def test_wrong_fraction_fails(self, fraction):
        """Ensures a fraction value outside of (0, 1) raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, fraction)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3))
    def test_too_small_window_fails(self, poly_order):
        """Ensures a window smaller than poly_order + 1 raises an exception."""
        for num_points in range(poly_order + 1):
            with pytest.raises(ValueError):
                self.class_func(self.y, total_points=num_points, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    def test_high_polynomial_order_warns(self, poly_order):
        """Ensure a warning is emitted when using a polynomial order above 2."""
        if poly_order > 2:
            with pytest.warns(ParameterWarning):
                self.class_func(self.y, poly_order=poly_order)
        else:  # no warning should be emitted
            self.class_func(self.y, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (1, 2))
    @pytest.mark.parametrize('delta', (0, 0.01))
    def test_output_coefs(self, poly_order, delta):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self.class_func(
            self.y, return_coef=True, poly_order=poly_order, delta=delta
        )
        # have to build the polynomial using the coefficients for each x-value
        recreated_poly = np.empty_like(baseline)
        for i, coef in enumerate(params['coef']):
            recreated_poly[i] = np.polynomial.Polynomial(coef)(self.x[i])

        # ignore coefficients that are all 0 since that means no fitting was done for
        # that x-value, so there are no actual coefficients available
        if delta > 0:
            mask = np.all(params['coef'] == 0, axis=1)
            baseline[mask] = 0

        assert_allclose(baseline, recreated_poly)

    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_compare_to_statsmodels(self, conserve_memory):
        """
        Compares the output of loess to the output of statsmodels.lowess.

        The library statsmodels has a well-tested lowess implementation, so
        can compare the output of polynomial.loess to statsmodels to ensure
        that the pybaselines implementation is correct.

        Since pybaselines's loess is for calculating the baseline rather than
        smoothing, the following changes need to be made to match statsmodels:

        * statsmodels uses int(fraction * num_x) to determine the window size while
          pybaselines uses ceil(fraction * num_x), so need to specify total points
          instead of fraction.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * set tol to -1 so that it goes through all iterations.

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, iterations, delta=0.0).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # Use set values to not worry about rng generation changes causing issues.
        # Used the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x), kind='stable')
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = LOESS_X
        y = LOESS_Y

        # test several iterations to ensure weighting is correct
        for iterations in range(4):
            output = self.algorithm_base(x, check_finite=False, assume_sorted=True).loess(
                y, conserve_memory=conserve_memory, total_points=total_points,
                max_iter=iterations, tol=-1, scale=4.0469385011764905, symmetric_weights=True,
                delta=0.0
            )

            assert_allclose(
                output[0], STATSMODELS_LOESS_ITER[iterations],
                err_msg=f'failed on iteration {iterations}'
            )

    @pytest.mark.parametrize('delta', (0.01, 0.3))
    def test_compare_to_statsmodels_delta(self, delta):
        """
        Compares the output of loess to the output of statsmodels.lowess when using delta.

        The library statsmodels has a well-tested lowess implementation, so
        can compare the output of polynomial.loess to statsmodels to ensure
        that the pybaselines implementation is correct.

        Since pybaselines's loess is for calculating the baseline rather than
        smoothing, the following changes need to be made to match statsmodels:

        * statsmodels uses int(fraction * num_x) to determine the window size while
          pybaselines uses ceil(fraction * num_x), so need to specify total points
          instead of fraction.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * only test the first iteration, since just want to check which points are selected
          for fitting

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, 0, delta=delta * (x.max() - x.min())).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # use set values since minimum numpy version is < 1.17
        # once min numpy version is >= 1.17, can use the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x), kind='stable')
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = LOESS_X
        y = LOESS_Y

        output = self.algorithm_base(x, check_finite=False, assume_sorted=True).loess(
            y, total_points=total_points, max_iter=0, scale=4.0469385011764905,
            symmetric_weights=True, delta=delta * (x.max() - x.min())
        )

        assert_allclose(output[0], STATSMODELS_LOESS_DELTA[delta])

    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_input_weights(self, use_threshold):
        """Ensures the input weights are sorted correctly."""
        super().test_input_weights(use_threshold=use_threshold)

    @pytest.mark.threaded_test
    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_threading(self, conserve_memory):
        """Tests the different possible computation routes under threading."""
        delta = 0.05 * (self.x.max() - self.x.min())  # use a larger delta to speed up method
        super().test_threading(conserve_memory=conserve_memory, delta=delta)


class TestQuantReg(IterativePolynomialTester):
    """Class for testing quant_reg baseline."""

    func_name = 'quant_reg'
    required_kwargs = {'tol': 1e-9}
    required_repeated_kwargs = {'tol': 1e-3}

    @pytest.mark.parametrize('quantile', (0, 1, -0.1, 1.1))
    def test_outside_quantile_fails(self, quantile):
        """Ensures quantile values outside of (0, 1) raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('quantile', tuple(STATSMODELS_QUANTILES.keys()))
    def test_compare_to_statsmodels(self, quantile):
        """
        Compares the output of quant_reg to statsmodels's quantile regression implementation.

        The library statsmodels has a well-tested quantile regression implementation,
        so can compare the output of polynomial.quant_reg to statsmodels to ensure
        that the pybaselines implementation is correct.

        The outputs from statsmodels were created using::

            from statsmodels.regression.quantile_regression import QuantReg
            vander = np.polynomial.polynomial.polyvander(x, 1)
            fitter = QuantReg(y, vander).fit(quantile, max_iter=1000, p_tol=1e-6).predict()

        with statsmodels version 0.11.1.

        Could also compare with the "true" quantile regression result using linear
        programming such as detailed in:

        https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-
        linear-programming-problem

        but the comparison to statsmodels is good enough since it uses an iteratively
        reweighted least squares calculation for the quantile regression similar to the
        pybaselines implementation, and the linear programming requires a scipy version
        of at least 1.0 or 1.6 to get a fast, reliable result due to the older solvers not
        working as well.

        """
        x = np.linspace(-1000, 1000, 200)
        # Use set values to not worry about rng generation changes causing issues.
        # Used the following to create y:
        # y = x + np.random.default_rng(0).normal(0, 200, x.size)
        y = QUANTILE_Y

        output = self.algorithm_base(x, check_finite=False, assume_sorted=True).quant_reg(
            y, poly_order=1, quantile=quantile, tol=1e-9, eps=1e-12
        )

        assert_allclose(output[0], STATSMODELS_QUANTILES[quantile], rtol=1e-6)


class TestGoldindec(PolynomialTester):
    """Class for testing goldindec baseline."""

    func_name = 'goldindec'
    checked_keys = ('weights', 'tol_history', 'threshold')

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'a_truncated_quadratic',
            'asymmetric_huber',
            'asymmetric_indec',
            'indec',
            'huber',
            'truncated_quadratic'
        )
    )
    def test_unchanged_data(self, use_class, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('s_huber', 's_indec', 'symmetric_indec'))
    def test_symmetric_cost_function_fails(self, cost_function):
        """Ensures a symmetric cost function fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function='a_hub')

    @pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
    def test_weighting(self, weight_enum):
        """
        Tests that weighting is correctly applied by comparing to other algorithms.

        Weights were not included in the original goldindec method, so need to ensure
        that their usage in pybaselines is correct.

        For uniform weights, the reference baseline is simply the unweighted calculation,
        since they should be equal. For non-uniform weights, compare to the output of
        penalized_poly, whose weighting is correctly tested, using the output optimal
        threshold.

        """
        if weight_enum == 0:
            # all weights = 1
            weights = None
            uniform_weights = True
        elif weight_enum == 1:
            # same as all weights = 1, but would cause issues if weights were
            # incorrectly multiplied
            weights = np.full_like(self.y, 2)
            uniform_weights = True
        elif weight_enum == 2:
            # binary mask, only fitting the first half of the data
            weights = np.ones_like(self.y)
            weights[self.x < 0.5 * (np.max(self.x) + np.min(self.x))] = 0
            uniform_weights = False
        else:
            # weight array where the two endpoints have weighting >> 1
            weights = np.ones_like(self.y)
            fraction = max(1, ceil(self.y.shape[0] * 0.1))
            weights[:fraction] = 100
            weights[-fraction:] = 100
            uniform_weights = False

        poly_order = 2
        fit_baseline, params = self.class_func(self.y, poly_order=poly_order, weights=weights)
        if uniform_weights:
            reference_baseline = self.class_func(self.y, poly_order=poly_order)[0]
        else:
            reference_baseline = polynomial.penalized_poly(
                self.y, self.x, poly_order=poly_order, weights=weights,
                threshold=params['threshold'], cost_function='a_indec'
            )[0]

        assert_allclose(fit_baseline, reference_baseline)

    @pytest.mark.parametrize('exit_enum', (0, 1, 2, 3))
    def test_tol_history(self, exit_enum):
        """
        Ensures the 'tol_history' item in the parameter output is correct.

        Since the shape of 'tol_history' is dictated by the number of iterations
        completed for fitting each threshold value and for iterating between
        threshold values, need to ensure each exit criteria works independently.

        """
        if exit_enum == 0:
            # inner fitting does more iterations
            max_iter = 15
            tol = -1
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = -1

            expected_shape_0 = max_iter_2 + 2
            expected_shape_1 = max_iter

        if exit_enum == 1:
            # outer fitting does more iterations
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = -1

            expected_shape_0 = max_iter_2 + 2
            expected_shape_1 = max_iter_2

        if exit_enum == 2:
            # only one iteration completed; exits due to tol_2
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 1e6
            tol_3 = -1

            expected_shape_0 = 3
            expected_shape_1 = 1

        if exit_enum == 3:
            # only one iteration completed; exits due to tol_3
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = 1e6

            expected_shape_0 = 3
            expected_shape_1 = 1

        _, params = self.class_func(
            self.y, max_iter=max_iter, tol=tol, max_iter_2=max_iter_2,
            tol_2=tol_2, tol_3=tol_3
        )

        assert params['tol_history'].shape[0] == expected_shape_0
        assert params['tol_history'].shape[1] == expected_shape_1

    @pytest.mark.parametrize('alpha_factor', (-0.1, 0, 1.01))
    def test_wrong_alpha_factor_fails(self, alpha_factor):
        """Ensures an alpha factor outside of (0, 1] fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha_factor=alpha_factor)

    @pytest.mark.parametrize('peak_ratio', (-0.1, 0, 1, 1.01))
    def test_wrong_peak_ratio_fails(self, peak_ratio):
        """Ensures a peak ratio outside of (0, 1) fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, peak_ratio=peak_ratio)


@pytest.mark.parametrize(
    'residual', (np.arange(100), -np.arange(100), np.linspace(-100, 100, 100))
)
@pytest.mark.parametrize('scale', (-3, 0.01, 1, 3, 50))
@pytest.mark.parametrize('symmetric', (True, False))
def test_tukey_square(residual, scale, symmetric):
    """
    Tests the Tukey square (sqrt of Tukey's bisquare) weighting for loess.

    Note for future, a negative scale is included to ensure it has no effect since it is
    squared in the weighting.

    """
    weights = polynomial._tukey_square(residual, scale, symmetric)

    assert np.all(weights >= 0)
    assert np.all(weights <= 1)

    if not symmetric:
        assert np.all(weights[residual < 0] == 1)

    # ensure that skipping the second squaring part of Tukey's bisquare does not change
    # the weights
    assert_allclose(weights, np.sqrt(weights**2))


@pytest.mark.parametrize(
    'values', (np.arange(10), np.linspace(-10, 10), np.full(10, 1))
)
def test_median_absolute_value(values):
    """Tests the median absolute values function."""
    mav_calc = polynomial._median_absolute_value(values)
    mav_actual = np.median(np.abs(values)) / 0.6744897501960817

    assert_allclose(mav_calc, mav_actual)


def test_loess_solver():
    """Tests that the loess solver solves `Ax=b` given `A.T` and `b`."""
    x = np.linspace(-1.0, 1.0, 50)
    coefs = np.array([2.0, -1.0, 0.2])
    y = coefs[0] + coefs[1] * x + coefs[2] * x**2

    vander = np.polynomial.polynomial.polyvander(x, coefs.size - 1)

    solved_coefs = polynomial._loess_solver(vander.T, y)

    assert_allclose(solved_coefs, coefs)


def test_determine_fits_simple():
    """A simple test to ensure the inner workings of _determine_fits work."""
    x = np.arange(22, dtype=float)
    num_x = x.shape[0]
    total_points = 5
    delta = 2.1  # should skip every other x-value, excluding the endpoints

    windows, fits, skips = polynomial._determine_fits(x, num_x, total_points, delta)

    # always fit first point
    desired_windows = [[0, total_points]]
    desired_fits = [0]
    desired_skips = []
    left = 0
    right = total_points
    for i, x_val in enumerate(x[1:-1], 1):
        if i % 2:  # all odd indices are skipped in this test setup
            # should be a slice that includes the last fit index and next fit index
            desired_skips.append([i - 1, i + 2])
        else:
            desired_fits.append(i)
            while right < num_x and x_val - x[left] > x[right] - x_val:
                left += 1
                right += 1
            desired_windows.append([left, right])
    # always fit last point
    desired_fits.append(num_x - 1)
    desired_windows.append([num_x - total_points, num_x])

    assert_array_equal(windows, desired_windows)
    assert_array_equal(fits, desired_fits)
    assert_array_equal(skips, desired_skips)


@pytest.mark.parametrize('delta', (0.0, 0.01, 0.5, -1.0, 3.0, np.nan, np.inf, -np.inf))
@pytest.mark.parametrize('total_points', (2, 10, 25, 50))
def test_determine_fits(delta, total_points):
    """Tests various inputs for _determine_fits to ensure any float delta works."""
    x = np.linspace(-1, 1, 50)
    num_x = x.shape[0]

    windows, fits, skips = polynomial._determine_fits(x, num_x, total_points, delta)

    assert windows.shape[0] == fits.shape[0]
    assert windows.shape[1] == 2

    # always fit first and last x-values
    assert fits[0] == 0
    assert fits[-1] == num_x - 1
    assert_array_equal(windows[0], (0, total_points))
    assert_array_equal(windows[-1], (num_x - total_points, num_x))

    # each window should be separated by total_points indices
    windows_transpose = windows.T
    assert_array_equal(
        windows_transpose[1] - windows_transpose[0],
        np.full(windows.shape[0], total_points)
    )

    # ensure no repeated fit indices
    assert not (np.diff(fits) == 0).sum()

    if delta <= 0:  # no points should be skipped
        assert skips.shape[0] == 0
        assert windows.shape[0] == num_x
        assert fits.shape[0] == num_x

        assert_array_equal(fits, np.arange(num_x))


def test_fill_skips():
    """Tests the linear interpolation performed by _fill_skips."""
    x = np.arange(20)
    y_actual = 2 + 5 * x
    y_calc = y_actual.copy()
    # `skips` slices y[left:right] where y_slice[0] and y_slice[-1] are actual values
    # and inbetween will be calculated using interpolation; fill in the sections
    # [left+1:right-1] in y_calc with zeros, and then check that they are returned to
    # the correct value by _fill_skips
    skips = np.array([[0, 5], [8, 14], [16, x.shape[0]]], dtype=np.intp)
    for (left, right) in skips:
        y_calc[left + 1:right - 1] = 0

    output = polynomial._fill_skips(x, y_calc, skips)

    # should not output anything from the function
    assert output is None
    # y_calc should be same as y_actual after interpolating each section
    assert_allclose(y_calc, y_actual)


def test_fill_skips_no_skips():
    """Ensures _fill_skips does not affect the input array when there are no skipped points."""
    skips = np.array([], np.intp).reshape(0, 2)

    x = np.arange(10)
    y_calc = np.empty(x.shape[0])
    y_calc[0] = 5
    y_calc[-1] = 10

    y_calc_before = y_calc.copy()

    polynomial._fill_skips(x, y_calc, skips)

    # y_calc should be unchanged since skips is an empty array
    assert_array_equal(y_calc, y_calc_before)
