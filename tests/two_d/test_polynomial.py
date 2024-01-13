# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines.two_d import polynomial

from ..conftest import BasePolyTester2D, InputWeightsMixin


class PolynomialTester(BasePolyTester2D, InputWeightsMixin):
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

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, new_instance, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            new_instance, use_original=use_original, mask_initial_peaks=mask_initial_peaks
        )


class TestIModPoly(IterativePolynomialTester):
    """Class for testing imodpoly baseline."""

    func_name = 'imodpoly'

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, new_instance, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            new_instance, use_original=use_original, mask_initial_peaks=mask_initial_peaks
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

    @pytest.mark.parametrize('new_instance', (True, False))
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
    def test_unchanged_data(self, new_instance, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(new_instance, cost_function=cost_function)

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

        poly_baseline = polynomial._Polynomial(self.x, self.z).poly(
            self.y, poly_order, weights=weights
        )[0]
        penalized_poly_1 = self.class_func(
            self.y, poly_order, cost_function='s_huber',
            threshold=1e10, weights=weights
        )[0]

        assert_allclose(poly_baseline, penalized_poly_1, 1e-10)

        modpoly_baseline = polynomial._Polynomial(self.x, self.z).modpoly(
            self.y, poly_order, tol=tol, weights=weights, use_original=True
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


class TestQuantReg(IterativePolynomialTester):
    """Class for testing quant_reg baseline."""

    func_name = 'quant_reg'
    required_kwargs = {'tol': 1e-9}

    @pytest.mark.parametrize('quantile', (0, 1, -0.1, 1.1))
    def test_outside_quantile_fails(self, quantile):
        """Ensures quantile values outside of (0, 1) raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, quantile=quantile)


class TestGoldindec(PolynomialTester):
    """Class for testing goldindec baseline."""

    func_name = 'goldindec'
    checked_keys = ('weights', 'tol_history', 'threshold')

    @pytest.mark.parametrize('new_instance', (True, False))
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
    def test_unchanged_data(self, new_instance, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(new_instance, cost_function=cost_function)

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
            reference_baseline = polynomial._Polynomial(self.x, self.z).penalized_poly(
                self.y, poly_order=poly_order, weights=weights,
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


