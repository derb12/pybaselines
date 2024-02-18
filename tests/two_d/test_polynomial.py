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
from ..data import STATSMODELS_QUANTILES_2D


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

    @pytest.mark.parametrize('quantile', tuple(STATSMODELS_QUANTILES_2D.keys()))
    def test_compare_to_statsmodels(self, quantile):
        """
        Compares the output of quant_reg to statsmodels's quantile regression implementation.

        The library statsmodels has a well-tested quantile regression implementation,
        so can compare the output of polynomial.quant_reg to statsmodels to ensure
        that the pybaselines implementation is correct.

        The outputs from statsmodels were created using::

            from statsmodels.regression.quantile_regression import QuantReg
            # map x and z to [-1, 1] to improve numerical stability for the Vandermonde
            # within statsmodels
            mapped_x = np.polynomial.polyutils.mapdomain(
                x, np.polynomial.polyutils.getdomain(x), np.array([-1., 1.])
            )
            mapped_z = np.polynomial.polyutils.mapdomain(
                z, np.polynomial.polyutils.getdomain(z), np.array([-1., 1.])
            )
            vander = np.polynomial.polynomial.polyvander2d(
                *np.meshgrid(mapped_x, mapped_z, indexing='ij'), 1
            ).reshape((-1, 4))
            fitter = QuantReg(y.ravel(), vander).fit(quantile, max_iter=1000, p_tol=1e-9).predict()

        with statsmodels version 0.13.2.

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
        x = np.linspace(-1000, 1000, 25)
        z = np.linspace(-200, 301, 31)

        X, Z = np.meshgrid(x, z, indexing='ij')
        y = (
            3 + 1e-2 * X - 5e-1 * Z + 1e-2 * X * Z
        ) + np.random.default_rng(0).normal(0, 200, X.shape)

        output = self.algorithm_base(x, z, check_finite=False, assume_sorted=True).quant_reg(
            y, poly_order=1, quantile=quantile, tol=1e-9, eps=1e-12
        )

        # use slightly high rtol since the number of data points is small for 2D to not bog
        # down the data file; for higher number of points, rtol and atol could be reduced
        assert_allclose(
            output[0].ravel(), STATSMODELS_QUANTILES_2D[quantile], rtol=1e-5, atol=1e-10
        )
