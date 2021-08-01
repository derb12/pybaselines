# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
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

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)


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

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)


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

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)


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

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)


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


class TestLoess(AlgorithmTester):
    """Class for testing LOESS baseline."""

    func = polynomial.loess

    @pytest.mark.parametrize('delta', (0, 0.01))
    @pytest.mark.parametrize('conserve_memory', (True, False))
    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_unchanged_data(self, data_fixture, use_threshold, conserve_memory, delta):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(
            data_fixture, y, x, y, x, use_threshold=use_threshold,
            conserve_memory=conserve_memory, delta=delta
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

    @pytest.mark.parametrize('poly_order', (1, 2))
    @pytest.mark.parametrize('delta', (0, 0.01))
    def test_output_coefs(self, poly_order, delta):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(
            self.y, self.x, return_coef=True, poly_order=poly_order, delta=delta
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
        * statsmodels uses the input iterations as number of robust fittings, while
          pybaselines uses iterations as total number of fits (intial + robust fittings),
          so add 1.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * set tol to -1 so that it goes through all iterations.

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, iterations).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # use set values since minimum numpy version is < 1.17
        # once min numpy version is >= 1.17, can use the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x))
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = np.array([
            0.08603252016391653, 0.4620121964065525, 0.5192309835763667, 0.8896887082266529,
            1.055121966462336, 1.2872212178963478, 1.634297372541321, 1.8399690787918987,
            2.6394198618724256, 2.8510910140793273, 3.3142319528623445, 3.610715163730215,
            3.9044742841509907, 4.244181877040637, 4.673559279481576, 4.721168691822169,
            5.518384072467553, 6.236471222720201, 6.267962530482449, 7.13636627051602,
            7.245838693824786, 7.520012836005392, 8.475599579967072, 9.383815765196083,
            9.415726735057563, 9.74653597091934, 10.111825144195999, 10.559428881564424,
            10.615794236187337, 11.2404676147093, 11.470274223088868, 12.053585858164322,
            12.218326883964336, 12.303073750654486, 12.709370879795477, 13.026797701038113,
            13.279110688808476, 13.358951239219458, 13.834856340638826, 14.147828458876802,
            14.452744299731492, 15.262967941601095, 15.626994858727365, 15.704690158414039,
            16.504492800213836, 16.628831939299175, 17.010505917383114, 17.078482794955864,
            17.95513917528035, 18.23167960456526, 18.670486089035595, 19.058024965549173,
            19.33289345358043, 19.484580245090765, 19.578001555572705, 19.76401547190605,
            20.010741575072398, 20.332058150420306, 20.43478083782305, 21.06828734519464,
            21.111341718376682, 21.536936621719185, 21.62819191149577, 22.341212411453007,
            22.666224692044207, 22.902685361628365, 22.917810368063478, 22.92280190156093,
            23.074481933138635, 23.80475373833605, 24.686185846584056, 24.72742260459408,
            25.017264774098162, 25.549638088547898, 25.63079532033328, 25.835635751138295,
            26.158287373224493, 26.93614976483979, 27.11756561187958, 27.132053628611267,
            27.535564205022077, 27.94408445848314, 27.958150040199374, 27.968793639376692,
            28.675062160988013, 29.127419326603402, 29.13588200930302, 29.281518641717938,
            29.343842478613347, 29.376166571460544, 29.597356268178427, 29.811944778549844,
            29.98934482165474, 30.071644682071696, 30.40829807484428, 30.5560353617595,
            30.813850946806603, 30.8251512961117, 31.261878704601713, 31.328274083621345
        ])
        y = np.array([
            -0.31643948442795794, 0.025294176131094415, 0.647017478746023, 1.0737896890796879,
            0.8206720006830988, 0.6377518395963726, 1.259897131794639, 0.5798730362015778,
            0.26741078101860366, 0.4727382381565013, -0.8468253620243291, -0.33619288971896866,
            -0.8654995424139683, -0.8595949238604239, -1.0219566797460828, -0.9393271400679253,
            -0.484142018099982, -0.27420802162579655, 0.4110724179260672, 0.9712041195512975,
            1.0738302517367937, 1.2942080292091869, 1.049213774810279, 0.29417434561985983,
            0.03172918534871679, -0.7442669944536149, -0.674770592505406, -1.1372411535695315,
            -1.3555687271138004, -0.8926273563547041, -1.059994748851263, -0.7995470629457199,
            -0.6539598102298176, -0.17974008110838563, 0.2501149777114302, 0.8410679994805202,
            0.6497348888787546, 1.0247191935429998, 1.3753305584133848, 1.3449928574031798,
            0.2410261733731892, 0.7990587775689757, 0.18276597078472304, 0.13040450954666363,
            -0.603565652310072, -0.6813004303259478, -0.8684109596218175, -1.0876855412294537,
            -1.350334701806057, -0.6119798501022928, -0.41923391123976983, 0.5310113576086335,
            0.37810702031798554, 0.6182397032157187, 0.410828769461069, 0.6390461696807992,
            0.9138160240699695, 0.5504920525015544, 1.0901014439969032, 0.7655100757551879,
            0.41489986158068176, -0.2807155537378368, 0.5089554617452788, -0.43223298454102355,
            -0.7839592638879793, -0.8612923921663656, -0.25467700223013867, -0.8175475502673442,
            -0.8575382314233819, -1.4167883371270387, 0.062340456353472684,
            -0.11906519352829137, 0.20486047721541026, 0.4192268707418591, 0.7527133733771633,
            0.7577128888236917, 1.038954288353462, 0.9274096611755345, 0.47334138166288053,
            1.2182396587983262, 0.09289095590039953, 0.2522979824352233, 0.24958510013390783,
            -0.012049601859495218, -0.20615208716141473, -0.8134120774853145,
            -0.8899122080685423, -0.6893786555970529, -1.019938175719467, -0.4753431711651658,
            -0.8640242335846221, -1.141749310181384, -1.5728985760169076, -1.3667810598063708,
            -0.5195189005816284, -0.7729527752716384, -0.6512918332086576, -0.06402954966432,
            -0.5382340490144448, -0.2632375970675537
        ])

        # outputs for iteration numbers from 0 to 3; test several iterations to ensure
        # weighting is correct
        statsmodels_outputs = {
            0: np.array([
                0.013949173555858115, 0.28450953728555345, 0.3235872359802879, 0.5654824453094635,
                0.6662153656116145, 0.785530170334511, 0.7506470516691395, 0.7066468110557571,
                0.22276406508876975, 0.057066368459986624, -0.2757522455273732, -0.4645446340925608,
                -0.6184296460333416, -0.7530368402016178, -0.7762575079967078, -0.7479540895316928,
                -0.40197196468678825, 0.15849673370691408, 0.18014175529804785, 0.7786503604739311,
                0.7896729843541994, 0.9213607881485303, 0.6117221206735213, 0.04238915039014286,
                0.011867565386790317, -0.3063316051973748, -0.6276671543213955, -0.9108500301537548,
                -0.9467508014226902, -0.9711786056041706, -0.8803946787628942, -0.5164455033883036,
                -0.33665555792536794, -0.2618288279042163, 0.1733249979935404, 0.5455634831830709,
                0.838881074083777, 0.8759154349258786, 0.9140394360646751, 0.8951455195299661,
                0.7947599612587716, 0.3821827601443787, 0.1512117349631967, 0.09301799334373848,
                -0.4965343116225852, -0.5767092464770565, -0.8284529823712354, -0.8333897829320087,
                -0.7961999810752187, -0.6688759468942046, -0.3043400334135521, 0.14139781590335151,
                0.40684222112537777, 0.49385608811180576, 0.5389406543303799, 0.633681218926093,
                0.7280361065264241, 0.7355179605022224, 0.7522593016658435, 0.4776899262477174,
                0.4520942101790715, 0.18673709685456896, 0.11439812612816182, -0.3744272253914193,
                -0.576991450149173, -0.7359246913680174, -0.7459521031614607, -0.7492664086445703,
                -0.8494030222551598, -0.6507205593587618, -0.2138275389166584, -0.1733819619391141,
                0.10855935515534255, 0.5649038887804986, 0.6211904489892349, 0.7611554486022242,
                0.7886392334805351, 0.7985530647362659, 0.6908910652230766, 0.6821904597960557,
                0.4422805681131109, 0.14751814767227792, 0.1382551476910366, 0.13141652432138526,
                -0.34179415758466986, -0.6686132853934371, -0.6748737335592727, -0.7823371105410748,
                -0.8143840666226682, -0.825596154630696, -1.0103668209266579, -1.1716761119973165,
                -1.0918251801003562, -1.1127686785924296, -0.8234247338878935, -0.6974499317651827,
                -0.5479975556623471, -0.5410878522967348, -0.29460927434031603, -0.2623235846170958
            ]),
            1: np.array([
                0.01346125764454193, 0.2692642398394642, 0.3061035256436901, 0.5332712747559366,
                0.6272399548736404, 0.7382692594538496, 0.7053387567902282, 0.6646796433786578,
                0.2323640059741657, 0.0732102907329617, -0.24208143437774257, -0.4344790148980033,
                -0.5944176245440754, -0.7386440408224565, -0.7732961317448773, -0.7468933864635734,
                -0.38864077929380175, 0.17376605338956985, 0.19575335549178777, 0.8004593165358889,
                0.8023627241804053, 0.9261251955073373, 0.5994439645248809, 0.05699878408812928,
                0.026778379675386994, -0.29169837149050243, -0.6026592561389593, -0.886440198981581,
                -0.9239793915493274, -0.9609694843339092, -0.8724345120389131, -0.5051580544040742,
                -0.3271780675113524, -0.2532521125138645, 0.17736628619186365, 0.551946966252646,
                0.837913844080486, 0.8785877440186318, 0.9393958885313628, 0.9188956576650167,
                0.7891363132059664, 0.3992718043014226, 0.1673797246537597, 0.10806424728859307,
                -0.5055534159783011, -0.5832066091639541, -0.8153769847974243, -0.8127875659982049,
                -0.7433780928306141, -0.6239392745287204, -0.2896337544128261, 0.12840721447996856,
                0.3924041730637772, 0.4844186732272839, 0.5310429518106545, 0.6279753242141739,
                0.7226446429902373, 0.7235517339825621, 0.7389667828447715, 0.48521139018302706,
                0.4591660736842548, 0.1957075636053834, 0.12443366754757555, -0.3866279810488643,
                -0.5977519177178532, -0.7518851267358441, -0.7616255982308762, -0.7648223824532712,
                -0.852675911156658, -0.5033274286373174, -0.1325593462844366, -0.0962563110744339,
                0.15101593034682986, 0.5636136209068374, 0.6200950996381333, 0.7600298538837871,
                0.7803653276682748, 0.7637445365292009, 0.6602361538819609, 0.6519445633670966,
                0.4237234581863734, 0.14624953450768802, 0.13757744072359518, 0.13118839776630803,
                -0.34175982603826516, -0.6719520156563255, -0.6783251218633565, -0.7877675374568502,
                -0.8205754954903806, -0.8316765329159181, -1.0041130525588309, -1.159110284262741,
                -1.0686241021543004, -1.0865289781985887, -0.8301676506585461, -0.7147444884905023,
                -0.5679481193833289, -0.5612212147658148, -0.3199989504639437, -0.28762875412878713
            ]),
            2: np.array([
                0.009632821670032841, 0.2625608627268248, 0.29896340674207683, 0.5232560468894007,
                0.6159304336895257, 0.7255391566074584, 0.6940970271682658, 0.6552320827338113,
                0.237006729183965, 0.07883508498982184, -0.23270924737933235, -0.4260654242619405,
                -0.5876180474849226, -0.7347025720277461, -0.7727618128545918, -0.7463492617833534,
                -0.3860056933483532, 0.1764240188412542, 0.1984457278388993, 0.8052960744507982,
                0.8065260859303459, 0.9288859197233666, 0.598362989014333, 0.059249987125926336,
                0.02912681367998666, -0.288564131875102, -0.5978613434024713, -0.8818149543395937,
                -0.9196946896640417, -0.9595401426176005, -0.8715597091873306, -0.5034937803298825,
                -0.32564302888101165, -0.2518252703610642, 0.17805050251945617, 0.5518797168689674,
                0.8378699835200704, 0.8791185038005354, 0.9449523497737788, 0.9251372451827247,
                0.7943227256588703, 0.4017342199761544, 0.16891164970863634, 0.10945830563044567,
                -0.5061528516633934, -0.5833965674913659, -0.8118362991601958, -0.8074048296684508,
                -0.7300984524620348, -0.6116517595281588, -0.28376513756146915, 0.12761290437844916,
                0.39103577604756684, 0.4833757160197309, 0.5300145368276965, 0.6268762416877547,
                0.7212053540247538, 0.7218010598361171, 0.7371642669851735, 0.4870840229211958,
                0.46112026509065546, 0.1986448394628882, 0.1274991389728427, -0.38643185770874733,
                -0.5993893750731988, -0.7513760032390301, -0.7608218644233942, -0.7639057159167137,
                -0.8408899914753452, -0.4414779953742434, -0.08149490003254517,
                -0.04942917198002014, 0.1729979788685168, 0.5640091213105332, 0.6203542740084158,
                0.7601187618196771, 0.7779580204561654, 0.7552101656857411, 0.6524288432383029,
                0.6442047380101015, 0.4177027955834176, 0.14424216043496363, 0.13577200849356333,
                0.12954031841513225, -0.34192682003191166, -0.672545738948012, -0.6789263071736452,
                -0.7884622058491338, -0.8212530125629236, -0.8322925612295267, -1.0029673626871698,
                -1.1572807374559146, -1.063641532624775, -1.081381934260377, -0.8303651121548383,
                -0.7167574250326637, -0.5708797172339388, -0.5642153599235998, -0.32535642053883235,
                -0.2932379764786771
            ]),
            3: np.array([
                0.007106331472409507, 0.259404964445307, 0.2957148378560415, 0.5194034836119471,
                0.6118138566010974, 0.7212223470091101, 0.6904598931449145, 0.6522574042502203,
                0.23875595379354567, 0.08088020256939216, -0.2294977703741831, -0.42317762492962857,
                -0.585268118753555, -0.733321795449864, -0.7725784968455178, -0.7461965061218885,
                -0.3853436325421424, 0.17713864942047153, 0.19916945554511944, 0.8065789237690831,
                0.8076768967841921, 0.9297017937123315, 0.598171636640104, 0.05989595535684206,
                0.02980300826380175, -0.28765441120426166, -0.5965191059465016, -0.8805269523851127,
                -0.9185021283550571, -0.9591390430363158, -0.8713205430033873, -0.5030803704283456,
                -0.3252804699266583, -0.25149179793725274, 0.17817845836393603, 0.5518385551773628,
                0.8377824917551533, 0.879484181963784, 0.9490546749668137, 0.9294451337103058,
                0.796727007213466, 0.4038053780544547, 0.17034305949441655, 0.11074127849421078,
                -0.5064491715658765, -0.5835580019853329, -0.8107264026696844, -0.8057097756794069,
                -0.725734631524126, -0.6075219493317787, -0.28170511937086584, 0.12756516512760932,
                0.3908011423820175, 0.48320473981083323, 0.5298383697871437, 0.6266453458882145,
                0.7208897297028004, 0.7213591144704607, 0.7367072332077464, 0.48790188023203035,
                0.46195031699963357, 0.19970635698784583, 0.1286095300811426, -0.38597719455088464,
                -0.5993971508213999, -0.7503800322752152, -0.75967609212089, -0.7627027202535341,
                -0.8337926944629892, -0.41910718421816456, -0.05942619548910799,
                -0.029552299992710543, 0.18159230858436773, 0.5642684398241131, 0.6205242852640489,
                0.7601508143598422, 0.777216093243909, 0.7524509125402858, 0.6499048318707901,
                0.6417024914571287, 0.4157578052225643, 0.1435747502329964, 0.1351686513197739,
                0.1289869115783057, -0.341974691255135, -0.6726754822580662, -0.6790581630565466,
                -0.7886240804504231, -0.8214152331220816, -0.832438899647899, -1.0025693005333978,
                -1.1566488665741346, -1.06210263238769, -1.0797871263797816, -0.8302873248767663,
                -0.7171663382187876, -0.5715459335050725, -0.5648992578038142, -0.3266929570620274,
                -0.29464213838044545
            ]),
        }
        for iterations in range(4):
            self._test_accuracy(
                statsmodels_outputs[iterations], y, x, conserve_memory=conserve_memory,
                total_points=total_points, max_iter=iterations + 1, tol=-1,
                scale=4.0469385011764905, symmetric_weights=True,
                assertion_kwargs={'err_msg': f'failed on iteration {iterations}'}
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
        * since x is scaled to (-1, 1) in pybaselines, use delta = delta * 2 rather than
          delta = delta * (x.max() - x.min()) for statsmodels.
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
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x))
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = np.array([
            0.08603252016391653, 0.4620121964065525, 0.5192309835763667, 0.8896887082266529,
            1.055121966462336, 1.2872212178963478, 1.634297372541321, 1.8399690787918987,
            2.6394198618724256, 2.8510910140793273, 3.3142319528623445, 3.610715163730215,
            3.9044742841509907, 4.244181877040637, 4.673559279481576, 4.721168691822169,
            5.518384072467553, 6.236471222720201, 6.267962530482449, 7.13636627051602,
            7.245838693824786, 7.520012836005392, 8.475599579967072, 9.383815765196083,
            9.415726735057563, 9.74653597091934, 10.111825144195999, 10.559428881564424,
            10.615794236187337, 11.2404676147093, 11.470274223088868, 12.053585858164322,
            12.218326883964336, 12.303073750654486, 12.709370879795477, 13.026797701038113,
            13.279110688808476, 13.358951239219458, 13.834856340638826, 14.147828458876802,
            14.452744299731492, 15.262967941601095, 15.626994858727365, 15.704690158414039,
            16.504492800213836, 16.628831939299175, 17.010505917383114, 17.078482794955864,
            17.95513917528035, 18.23167960456526, 18.670486089035595, 19.058024965549173,
            19.33289345358043, 19.484580245090765, 19.578001555572705, 19.76401547190605,
            20.010741575072398, 20.332058150420306, 20.43478083782305, 21.06828734519464,
            21.111341718376682, 21.536936621719185, 21.62819191149577, 22.341212411453007,
            22.666224692044207, 22.902685361628365, 22.917810368063478, 22.92280190156093,
            23.074481933138635, 23.80475373833605, 24.686185846584056, 24.72742260459408,
            25.017264774098162, 25.549638088547898, 25.63079532033328, 25.835635751138295,
            26.158287373224493, 26.93614976483979, 27.11756561187958, 27.132053628611267,
            27.535564205022077, 27.94408445848314, 27.958150040199374, 27.968793639376692,
            28.675062160988013, 29.127419326603402, 29.13588200930302, 29.281518641717938,
            29.343842478613347, 29.376166571460544, 29.597356268178427, 29.811944778549844,
            29.98934482165474, 30.071644682071696, 30.40829807484428, 30.5560353617595,
            30.813850946806603, 30.8251512961117, 31.261878704601713, 31.328274083621345
        ])
        y = np.array([
            -0.31643948442795794, 0.025294176131094415, 0.647017478746023, 1.0737896890796879,
            0.8206720006830988, 0.6377518395963726, 1.259897131794639, 0.5798730362015778,
            0.26741078101860366, 0.4727382381565013, -0.8468253620243291, -0.33619288971896866,
            -0.8654995424139683, -0.8595949238604239, -1.0219566797460828, -0.9393271400679253,
            -0.484142018099982, -0.27420802162579655, 0.4110724179260672, 0.9712041195512975,
            1.0738302517367937, 1.2942080292091869, 1.049213774810279, 0.29417434561985983,
            0.03172918534871679, -0.7442669944536149, -0.674770592505406, -1.1372411535695315,
            -1.3555687271138004, -0.8926273563547041, -1.059994748851263, -0.7995470629457199,
            -0.6539598102298176, -0.17974008110838563, 0.2501149777114302, 0.8410679994805202,
            0.6497348888787546, 1.0247191935429998, 1.3753305584133848, 1.3449928574031798,
            0.2410261733731892, 0.7990587775689757, 0.18276597078472304, 0.13040450954666363,
            -0.603565652310072, -0.6813004303259478, -0.8684109596218175, -1.0876855412294537,
            -1.350334701806057, -0.6119798501022928, -0.41923391123976983, 0.5310113576086335,
            0.37810702031798554, 0.6182397032157187, 0.410828769461069, 0.6390461696807992,
            0.9138160240699695, 0.5504920525015544, 1.0901014439969032, 0.7655100757551879,
            0.41489986158068176, -0.2807155537378368, 0.5089554617452788, -0.43223298454102355,
            -0.7839592638879793, -0.8612923921663656, -0.25467700223013867, -0.8175475502673442,
            -0.8575382314233819, -1.4167883371270387, 0.062340456353472684,
            -0.11906519352829137, 0.20486047721541026, 0.4192268707418591, 0.7527133733771633,
            0.7577128888236917, 1.038954288353462, 0.9274096611755345, 0.47334138166288053,
            1.2182396587983262, 0.09289095590039953, 0.2522979824352233, 0.24958510013390783,
            -0.012049601859495218, -0.20615208716141473, -0.8134120774853145,
            -0.8899122080685423, -0.6893786555970529, -1.019938175719467, -0.4753431711651658,
            -0.8640242335846221, -1.141749310181384, -1.5728985760169076, -1.3667810598063708,
            -0.5195189005816284, -0.7729527752716384, -0.6512918332086576, -0.06402954966432,
            -0.5382340490144448, -0.2632375970675537
        ])

        # outputs for delta values of 0.01 and 0.3
        statsmodels_outputs = {
            0.01: np.array([
                0.013949173555858115, 0.28450953728555345, 0.3235872359802879, 0.5654824453094635,
                0.6662153656116145, 0.785530170334511, 0.7506470516691395, 0.7066468110557571,
                0.22276406508876975, 0.057066368459986624, -0.2757522455273732, -0.4645446340925608,
                -0.6184296460333416, -0.7530368402016178, -0.7762575079967078, -0.7479540895316928,
                -0.40197196468678825, 0.15849673370691408, 0.18014175529804785, 0.7786503604739311,
                0.7896729843541994, 0.9213607881485303, 0.6117221206735213, 0.04238915039014286,
                0.011867565386790317, -0.3063316051973748, -0.6276671543213955, -0.9108500301537548,
                -0.9467508014226902, -0.9711786056041706, -0.8803946787628942, -0.5164455033883036,
                -0.3483178564457109, -0.2618288279042163, 0.1733249979935404, 0.5455634831830709,
                0.838881074083777, 0.8759154349258786, 0.9140394360646751, 0.8951455195299661,
                0.7947599612587716, 0.3821827601443787, 0.1512117349631967, 0.09301799334373848,
                -0.4965343116225852, -0.5767092464770565, -0.8284529823712354, -0.8333897829320087,
                -0.7961999810752187, -0.6688759468942046, -0.3043400334135521, 0.14139781590335151,
                0.40684222112537777, 0.488592221772197, 0.5389406543303799, 0.633681218926093,
                0.7280361065264241, 0.7355179605022224, 0.7522593016658435, 0.4776899262477174,
                0.4520942101790715, 0.18673709685456896, 0.11439812612816182, -0.3744272253914193,
                -0.576991450149173, -0.735759456532645, -0.7459149175759684, -0.7492664086445703,
                -0.8494030222551598, -0.6507205593587618, -0.2138275389166584, -0.1733819619391141,
                0.10855935515534255, 0.5649038887804986, 0.62059397094563, 0.7611554486022242,
                0.7886392334805351, 0.7985530647362659, 0.6907960244585422, 0.6821904597960557,
                0.4422805681131109, 0.14751814767227792, 0.1383523764562392, 0.13141652432138526,
                -0.34179415758466986, -0.6686132853934371, -0.6739540328352579, -0.7658644233832042,
                -0.8051966167642209, -0.825596154630696, -1.0103668209266579, -1.1716761119973165,
                -1.1314366622189056, -1.1127686785924296, -0.8234247338878935, -0.6974499317651827,
                -0.547653595497032, -0.5410878522967348, -0.29460927434031603, -0.2623235846170958
            ]),
            0.3: np.array([
                0.013949173555858115, 0.013865286308597562, 0.013852519856444532,
                0.013769864647370771, 0.013732953767566246, 0.013681168608621992,
                0.013603730212233364, 0.013557841473115433, 0.013379470859620783,
                0.013332243545472963, 0.013228909187519254, 0.013162758908687096,
                0.013097216419220518, 0.013021422070106203, 0.012925620912217519,
                0.01291499846957685, 0.012737126611117783, 0.012576909561808462,
                0.012569883333297362, 0.012376128181449276, 0.01235170308397037,
                0.012290530325146268, 0.012077323211743976, 0.01187468524881543,
                0.011867565386790317, 0.0005647977910163918, -0.011916046705488586,
                -0.02720933501195239, -0.02913517152040577, -0.05047840277060817,
                -0.05833021101810402, -0.07826023452543904, -0.08388894535407192,
                -0.08678449370892206, -0.1006664580789543, -0.1115119883732472, -0.1201327726591955,
                -0.12286068678064722, -0.1391209485141917, -0.14981427485735035, -0.160232342165757,
                -0.18791527399442004, -0.20035299095285303, -0.2030076082381135,
                -0.23033448549170488, -0.23458278402689878, -0.24762344856698976,
                -0.2499460162761096, -0.2798987571732841, -0.2893473210711461, -0.3043400334135521,
                -0.2861783834677158, -0.2732969267589717, -0.2661882659876923, -0.26181016293460463,
                -0.2530927933021881, -0.24153020335374314, -0.22647200012065624,
                -0.22165799666663405, -0.19196930187348327, -0.18995159861386657,
                -0.17000648827351655, -0.1657298937714677, -0.13231484947726038,
                -0.11708345044628218, -0.10600194051989659, -0.10529312112408705,
                -0.10505919753714474, -0.09795085356383719, -0.0637273428527906,
                -0.02241984476373135, -0.02048732235303341, -0.006904137952460115,
                0.018045043571014635, 0.02184840195280935, 0.03144805872145001, 0.04656882768146553,
                0.08302262707739037, 0.09152451245778169, 0.09220347992204611, 0.11111362876735556,
                0.13025855148037838, 0.13091772191951714, 0.13141652432138526, 0.040046697842723716,
                -0.01847466402451703, -0.019569479701766937, -0.03841046358467973,
                -0.04647328670134099, -0.050655048544259354, -0.07927031821799804,
                -0.10703159364935735, -0.12998180402942822, -0.1406289215710106,
                -0.18418170799692413, -0.20329445315945116, -0.23664800715263393,
                -0.23810993122762905, -0.29460927434031603, -0.2623235846170958
            ])
        }

        self._test_accuracy(
            statsmodels_outputs[delta], y, x, total_points=total_points, max_iter=1,
            scale=4.0469385011764905, symmetric_weights=True, delta=2 * delta
        )


@pytest.mark.parametrize('quantile', np.linspace(0, 1, 21))
def test_quantile_loss(quantile):
    """Ensures the quantile loss calculation is correct."""
    residual = np.linspace(-1, 1)
    eps = 1e-10
    calc_loss = polynomial._quantile_loss(residual, quantile, eps)

    numerator = np.where(residual > 0, quantile, 1 - quantile)
    denominator = np.sqrt(residual**2 + eps)

    expected_loss = numerator / denominator

    assert_allclose(calc_loss, expected_loss)


class TestQuantReg(AlgorithmTester):
    """Class for testing quant_reg baseline."""

    func = polynomial.quant_reg

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        kwargs = {'tol': 1e-9}
        self._test_algorithm_no_x(
            with_args=(self.y, self.x), with_kwargs=kwargs,
            without_args=(self.y,), without_kwargs=kwargs
        )

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

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)

    @pytest.mark.parametrize('quantile', (0, 1, -0.1, 1.1))
    def test_outside_quantile_fails(self, quantile):
        """Ensures quantile values outside of (0, 1) raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('quantile', [0.1, 0.5, 0.9])
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
        # TODO directly calculating y gives different results, so probably something to
        # do with the random seed not working correctly; once minimum numpy version
        # is >= 1.17, can use np.random.default_rng to get a consistent random
        # generator and recreate outputs without having to save y as well

        # np.random.seed(0)
        # y = x + np.random.normal(0, 200, x.size)
        y = np.array([
            -1073.8363675884887, -1037.825584258904, -759.9675783100146, -838.7965000866362,
            -831.7726897553559, -1273.1399525807599, -944.5637173420987, -1077.254423047168,
            -863.613070141101, -929.1778166220589, -717.4617058186675, -826.003593142644,
            -722.1313925028279, -962.6305530155303, -1048.1857335957104, -931.2561697962885,
            -842.6000626717856, -753.3153815321997, -367.23368724876417, -817.4966564627817,
            -990.1839749729272, -858.1410787579676, -871.6136672910278, -672.5479263508352,
            -1066.953372738171, -736.0913197528985, -707.3921597436083, -682.2070088403468,
            -838.0561786171932, -756.1270595150406, -983.3046441080642, -787.1061877277152,
            -786.9642550023385, -585.1316992904285, -889.5279436508147, -492.0015856881521,
            -339.29404587559554, -1042.1377085202944, -532.8387061056862, -472.658593998976,
            -725.4773548591895, -667.3940613582218, -604.4655627535719, -627.397371860245,
            -619.5915385330425, -882.9394547333319, -307.2221292544312, -311.71447254740957,
            -680.260791539333, -800.8225540027138, -393.2744618953779, -602.5947798922614,
            -448.99630200921086, -531.2023668461044, -318.97868194676664, -308.2863521733214,
            -582.305405340958, -703.8084694709709, -733.6731066026946, -284.95930005795583,
            -634.7567761799213, -488.2979442265716, -496.1472298006543, -377.34763010818057,
            -744.0398807672913, -308.9779489829428, -231.90521231858597, -308.9487484202134,
            -378.76014891255863, -287.0526300628262, -216.67314293227545, -840.95071208935,
            114.80055210240027, -188.3129937538719, -386.7631235125798, -324.4218308164146,
            -137.4325490527753, -249.3514410732049, -622.2172955663491, 206.8684215180951,
            -218.08803094398192, 18.10489410195396, -314.2893665418028, 141.44626512051633,
            -98.51015669390577, -23.959876320990134, -344.72906518918984, 116.60091723302267,
            22.385743459721596, 154.841607722045, -221.09489886298906, -181.63255937054942,
            385.4064551146779, -277.32979771013856, -82.46632204521185, 182.15214186727277,
            -15.630885854013663, 91.46510780995534, -94.96518273699702, 68.98605194136316,
            -256.2802447189225, 346.7015128080598, 1.4928191149896577, -100.85976140238506,
            178.50274705964927, -36.86757556816285, -201.5250611147215, -193.96661673740044,
            224.1817662164185, 63.562699305420026, 78.78732625726697, 331.12665064248876,
            -99.73702104783109, -10.457158613165973, 68.7526813798253, 174.6492123357766,
            157.39485547052763, 118.50195850694331, 173.60436782207674, 174.5188442392527,
            62.10927304341044, 53.481804299234966, 281.03392481111973, 57.99788792350765,
            14.760103940723894, 193.82295681003734, 234.79825505869422, 727.7266090073806,
            145.49210563277347, 485.13455705420057, 455.97033015750935, 78.79392353211719,
            481.28376130966575, 99.90728904676365, -185.1007792576396, 478.04782446986616,
            15.656054166727529, 467.0713144717358, 250.1324938193909, 728.89508386109,
            620.7370557425988, 326.4082663654011, 289.5681561862235, 194.3704490294173,
            359.05165444600453, 401.2153331239024, 394.39797453874854, 508.7277057278398,
            603.1414854754241, 567.4183285845646, 354.7089036609226, 230.02964493817694,
            800.5445605752683, 399.7986053010679, 417.27997348033296, 453.55108226339604,
            199.22528594856993, 482.2946464280837, 492.00853569087286, 722.0616094174463,
            747.7316308264924, 618.8446300785605, 814.5103783404487, 706.1839515341218,
            645.1047837096398, 690.4770909460768, 630.2110098264337, 599.4220569923052,
            634.8955036764834, 472.89019601755666, 764.6310546310987, 519.9682426382046,
            896.9694688951295, 688.8017513044939, 758.6427149229831, 857.5613251054377,
            897.5071141181133, 464.7697906351187, 747.5639883852963, 975.0307572905285,
            469.4240622442025, 896.5515724659475, 378.03288276277834, 634.6946111723769,
            856.9723415238167, 527.9534619476697, 672.2445861315117, 585.0299834319059,
            1219.0742226224165, 932.3430385346528, 1023.0057942640012, 875.0629698467179,
            858.9618914545906, 616.4036547321053, 891.3812711690207, 807.50055261255,
            1141.4340310896232, 1208.1198708189522, 1283.2654918585545, 1170.5103878922464
        ])
        statsmodels_outputs = {
            0.1: np.array([
                -1225.0484344585661, -1215.4121557841213, -1205.7758771096762, -1196.1395984352314,
                -1186.5033197607863, -1176.8670410863415, -1167.2307624118964, -1157.5944837374516,
                -1147.9582050630065, -1138.3219263885615, -1128.6856477141166, -1119.0493690396715,
                -1109.4130903652267, -1099.7768116907819, -1090.1405330163368, -1080.504254341892,
                -1070.867975667447, -1061.231696993002, -1051.595418318557, -1041.959139644112,
                -1032.322860969667, -1022.6865822952221, -1013.0503036207772, -1003.4140249463322,
                -993.7777462718873, -984.1414675974424, -974.5051889229974, -964.8689102485525,
                -955.2326315741075, -945.5963528996626, -935.9600742252177, -926.3237955507726,
                -916.6875168763278, -907.0512382018828, -897.4149595274379, -887.7786808529929,
                -878.1424021785479, -868.5061235041029, -858.8698448296581, -849.233566155213,
                -839.5972874807682, -829.9610088063232, -820.3247301318781, -810.6884514574332,
                -801.0521727829881, -791.4158941085433, -781.7796154340983, -772.1433367596534,
                -762.5070580852084, -752.8707794107636, -743.2345007363185, -733.5982220618736,
                -723.9619433874286, -714.3256647129838, -704.6893860385388, -695.0531073640939,
                -685.4168286896489, -675.780550015204, -666.1442713407589, -656.507992666314,
                -646.871713991869, -637.2354353174242, -627.5991566429791, -617.9628779685341,
                -608.3265992940892, -598.6903206196444, -589.0540419451994, -579.4177632707544,
                -569.7814845963095, -560.1452059218645, -550.5089272474196, -540.8726485729745,
                -531.2363698985296, -521.6000912240847, -511.96381254963967, -502.3275338751948,
                -492.6912552007498, -483.0549765263049, -473.4186978518599, -463.782419177415,
                -454.14614050297007, -444.5098618285251, -434.87358315408017, -425.2373044796352,
                -415.60102580519015, -405.96474713074525, -396.3284684563003, -386.69218978185535,
                -377.0559111074104, -367.41963243296544, -357.7833537585205, -348.1470750840756,
                -338.51079640963064, -328.8745177351857, -319.23823906074074, -309.6019603862958,
                -299.9656817118508, -290.3294030374058, -280.6931243629609, -271.0568456885159,
                -261.42056701407097, -251.78428833962604, -242.1480096651811, -232.51173099073614,
                -222.87545231629122, -213.23917364184626, -203.6028949674013, -193.9666162929564,
                -184.33033761851144, -174.69405894406648, -165.05778026962156, -155.4215015951766,
                -145.78522292073166, -136.14894424628673, -126.51266557184178, -116.87638689739684,
                -107.24010822295166, -97.60382954850672, -87.96755087406179, -78.33127219961685,
                -68.69499352517191, -59.05871485072694, -49.422436176282005, -39.78615750183707,
                -30.149878827392126, -20.513600152947163, -10.877321478502221, -1.241042804057283,
                8.395235870387657, 18.031514544832596, 27.667793219277534, 37.30407189372247,
                46.94035056816741, 56.57662924261235, 66.2129079170573, 75.84918659150223,
                85.48546526594717, 95.12174394039211, 104.7580226148371, 114.39430128928204,
                124.03057996372698, 133.66685863817213, 143.30313731261708, 152.939415987062,
                162.57569466150696, 172.21197333595188, 181.84825201039683, 191.48453068484176,
                201.1208093592867, 210.75708803373163, 220.39336670817664, 230.0296453826215,
                239.66592405706652, 249.3022027315114, 258.9384814059564, 268.5747600804013,
                278.21103875484624, 287.84731742929125, 297.48359610373615, 307.11987477818116,
                316.75615345262605, 326.392432127071, 336.0287108015159, 345.6649894759609,
                355.3012681504058, 364.93754682485076, 374.57382549929565, 384.2101041737409,
                393.84638284818584, 403.48266152263074, 413.11894019707574, 422.75521887152064,
                432.3914975459656, 442.0277762204105, 451.6640548948555, 461.3003335693004,
                470.93661224374534, 480.57289091819024, 490.20916959263525, 499.84544826708014,
                509.4817269415251, 519.1180056159701, 528.7542842904151, 538.39056296486,
                548.0268416393048, 557.6631203137499, 567.2993989881948, 576.9356776626397,
                586.5719563370847, 596.2082350115296, 605.8445136859746, 615.4807923604195,
                625.1170710348646, 634.7533497093096, 644.3896283837546, 654.0259070581994,
                663.6621857326445, 673.2984644070893, 682.9347430815343, 692.571021755979
            ]),
            0.5: np.array([
                -1011.3097618372178, -1001.4284486370647, -991.5471354369114, -981.6658222367582,
                -971.784509036605, -961.9031958364517, -952.0218826362985, -942.1405694361453,
                -932.259256235992, -922.3779430358387, -912.4966298356856, -902.6153166355323,
                -892.7340034353791, -882.8526902352259, -872.9713770350727, -863.0900638349194,
                -853.2087506347663, -843.327437434613, -833.4461242344596, -823.5648110343064,
                -813.6834978341532, -803.802184634, -793.9208714338467, -784.0395582336936,
                -774.1582450335403, -764.2769318333872, -754.3956186332339, -744.5143054330807,
                -734.6329922329274, -724.7516790327743, -714.870365832621, -704.9890526324677,
                -695.1077394323146, -685.2264262321613, -675.3451130320082, -665.4637998318549,
                -655.5824866317016, -645.7011734315483, -635.8198602313952, -625.9385470312419,
                -616.0572338310888, -606.1759206309355, -596.2946074307822, -586.4132942306289,
                -576.5319810304757, -566.6506678303225, -556.7693546301692, -546.8880414300161,
                -537.0067282298628, -527.1254150297096, -517.2441018295564, -507.3627886294031,
                -497.4814754292499, -487.60016222909667, -477.71884902894345, -467.83753582879024,
                -457.956222628637, -448.0749094284838, -438.1935962283305, -428.31228302817726,
                -418.43096982802405, -408.54965662787083, -398.66834342771756, -388.78703022756434,
                -378.90571702741113, -369.0244038272579, -359.1430906271047, -349.2617774269515,
                -339.38046422679827, -329.49915102664505, -319.61783782649184, -309.7365246263385,
                -299.8552114261853, -289.9738982260321, -280.0925850258788, -270.2112718257256,
                -260.3299586255724, -250.44864542541916, -240.56733222526591, -230.6860190251127,
                -220.80470582495948, -210.92339262480627, -201.04207942465305, -191.16076622449984,
                -181.27945302434648, -171.39813982419327, -161.51682662404005, -151.63551342388683,
                -141.7542002237336, -131.87288702358038, -121.99157382342716, -112.11026062327394,
                -102.22894742312072, -92.3476342229675, -82.46632102281427, -72.58500782266105,
                -62.70369462250772, -52.822381422354496, -42.941068222201274, -33.05975502204806,
                -23.178441821894836, -13.297128621741614, -3.415815421588392, 6.465497778564831,
                16.34681097871805, 26.228124178871273, 36.10943737902449, 45.99075057917771,
                55.87206377933094, 65.75337697948416, 75.63469017963737, 85.5160033797906,
                95.39731657994382, 105.27862978009703, 115.15994298025025, 125.04125618040347,
                134.92256938055692, 144.80388258071014, 154.68519578086335, 164.5665089810166,
                174.4478221811698, 184.32913538132306, 194.21044858147627, 204.0917617816295,
                213.9730749817827, 223.85438818193595, 233.73570138208916, 243.61701458224238,
                253.4983277823956, 263.37964098254884, 273.26095418270205, 283.14226738285527,
                293.0235805830085, 302.9048937831617, 312.7862069833149, 322.6675201834681,
                332.54883338362134, 342.43014658377456, 352.31145978392783, 362.19277298408105,
                372.07408618423426, 381.9553993843877, 391.8367125845409, 401.71802578469413,
                411.59933898484735, 421.48065218500057, 431.3619653851538, 441.243278585307,
                451.1245917854602, 461.0059049856134, 470.8872181857667, 480.76853138591986,
                490.64984458607313, 500.5311577862263, 510.41247098637956, 520.2937841865328,
                530.175097386686, 540.0564105868393, 549.9377237869925, 559.8190369871458,
                569.7003501872989, 579.5816633874522, 589.4629765876053, 599.3442897877586,
                609.2256029879118, 619.106916188065, 628.9882293882182, 638.8695425883717,
                648.750855788525, 658.6321689886781, 668.5134821888314, 678.3947953889846,
                688.2761085891378, 698.157421789291, 708.0387349894443, 717.9200481895974,
                727.8013613897507, 737.6826745899039, 747.5639877900571, 757.4453009902103,
                767.3266141903636, 777.2079273905168, 787.08924059067, 796.9705537908233,
                806.8518669909764, 816.7331801911297, 826.6144933912828, 836.4958065914361,
                846.3771197915893, 856.2584329917426, 866.1397461918957, 876.021059392049,
                885.9023725922024, 895.7836857923556, 905.6649989925089, 915.5463121926621,
                925.4276253928153, 935.3089385929685, 945.1902517931218, 955.0715649932747
            ]),
            0.9: np.array([
                -782.9534183838906, -772.9270578562677, -762.9006973286445, -752.8743368010215,
                -742.8479762733983, -732.8216157457753, -722.7952552181522, -712.7688946905291,
                -702.7425341629059, -692.7161736352829, -682.6898131076599, -672.6634525800367,
                -662.6370920524138, -652.6107315247906, -642.5843709971676, -632.5580104695445,
                -622.5316499419215, -612.5052894142983, -602.4789288866751, -592.4525683590522,
                -582.426207831429, -572.399847303806, -562.3734867761829, -552.3471262485599,
                -542.3207657209367, -532.2944051933138, -522.2680446656906, -512.2416841380676,
                -502.2153236104445, -492.1889630828215, -482.16260255519836, -472.13624202757524,
                -462.1098814999522, -452.0835209723291, -442.05716044470614, -432.030799917083,
                -422.0044393894599, -411.97807886183676, -401.95171833421375, -391.9253578065906,
                -381.8989972789676, -371.8726367513445, -361.84627622372136, -351.8199156960983,
                -341.79355516847517, -331.76719464085215, -321.740834113229, -311.714473585606,
                -301.6881130579829, -291.6617525303599, -281.63539200273675, -271.6090314751137,
                -261.58267094749067, -251.55631041986757, -241.5299498922445, -231.50358936462146,
                -221.4772288369984, -211.45086830937532, -201.42450778175214, -191.3981472541291,
                -181.37178672650603, -171.34542619888296, -161.31906567125984, -151.29270514363677,
                -141.26634461601373, -131.23998408839066, -121.21362356076759, -111.18726303314452,
                -101.16090250552146, -91.1345419778984, -81.10818145027534, -71.08182092265216,
                -61.055460395029094, -51.02909986740603, -41.00273933978291, -30.976378812159876,
                -20.950018284536814, -10.92365775691375, -0.8972972292906581, 9.129063298332404,
                19.15542382595547, 29.181784353578532, 39.2081448812016, 49.23450540882466,
                59.260865936447864, 69.28722646407093, 79.313586991694, 89.33994751931705,
                99.36630804694013, 109.39266857456319, 119.41902910218626, 129.4453896298093,
                139.4717501574324, 149.49811068505545, 159.52447121267852, 169.5508317403016,
                179.57719226792477, 189.60355279554784, 199.6299133231709, 209.65627385079398,
                219.68263437841705, 229.70899490604012, 239.7353554336632, 249.76171596128626,
                259.7880764889093, 269.8144370165324, 279.84079754415546, 289.86715807177853,
                299.8935185994016, 309.91987912702467, 319.94623965464774, 329.9726001822708,
                339.9989607098939, 350.02532123751695, 360.05168176513996, 370.07804229276303,
                380.1044028203864, 390.13076334800945, 400.15712387563246, 410.18348440325553,
                420.2098449308786, 430.23620545850173, 440.2625659861248, 450.2889265137478,
                460.3152870413709, 470.341647568994, 480.3680080966171, 490.39436862424014,
                500.42072915186316, 510.4470896794862, 520.4734502071093, 530.4998107347324,
                540.5261712623554, 550.5525317899785, 560.5788923176016, 570.6052528452246,
                580.6316133728477, 590.6579739004708, 600.6843344280938, 610.7106949557169,
                620.73705548334, 630.7634160109633, 640.7897765385864, 650.8161370662094,
                660.8424975938325, 670.8688581214556, 680.8952186490786, 690.9215791767017,
                700.9479397043248, 710.9743002319478, 721.0006607595709, 731.027021287194,
                741.053381814817, 751.07974234244, 761.1061028700632, 771.1324633976861,
                781.1588239253093, 791.1851844529324, 801.2115449805555, 811.2379055081785,
                821.2642660358016, 831.2906265634247, 841.3169870910476, 851.3433476186708,
                861.3697081462938, 871.396068673917, 881.4224292015399, 891.4487897291633,
                901.4751502567864, 911.5015107844094, 921.5278713120325, 931.5542318396555,
                941.5805923672787, 951.6069528949016, 961.6333134225248, 971.6596739501477,
                981.6860344777709, 991.7123950053939, 1001.7387555330171, 1011.76511606064,
                1021.7914765882631, 1031.8178371158863, 1041.8441976435092, 1051.8705581711324,
                1061.8969186987554, 1071.9232792263786, 1081.9496397540015, 1091.9760002816247,
                1102.0023608092476, 1112.0287213368708, 1122.0550818644938, 1132.081442392117,
                1142.1078029197402, 1152.1341634473633, 1162.1605239749863, 1172.1868845026095,
                1182.2132450302324, 1192.2396055578556, 1202.2659660854786, 1212.2923266131015
            ])
        }

        self._test_accuracy(
            statsmodels_outputs[quantile], y, x, poly_order=1, quantile=quantile,
            tol=1e-9, eps=1e-6, assertion_kwargs={'rtol': 1e-6}
        )
