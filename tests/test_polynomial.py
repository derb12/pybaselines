# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pybaselines import polynomial

from .conftest import get_data, AlgorithmTester


class TestPoly(AlgorithmTester):
    """Class for testing regular polynomial baseline."""

    func = polynomial.poly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestModPoly(AlgorithmTester):
    """Class for testing ModPoly baseline."""

    func = polynomial.modpoly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIModPoly(AlgorithmTester):
    """Class for testing IModPoly baseline."""

    func = polynomial.imodpoly

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


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
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        with pytest.raises(ValueError):
            super()._call_func(self.y, self.x, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        with pytest.raises(KeyError):
            super()._call_func(self.y, self.x, cost_function='a_hub')

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

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
            weights[self.x < 0.5 * (max(self.x) + min(self.x))] = 0
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


class TestLoess(AlgorithmTester):
    """Class for testing LOESS baseline."""

    func = polynomial.loess

    @pytest.mark.parametrize('use_threshold', (False, True))
    def test_unchanged_data(self, data_fixture, use_threshold):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x, use_threshold=use_threshold)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_output(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
