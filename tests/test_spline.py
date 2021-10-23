# -*- coding: utf-8 -*-
"""Tests for pybaselines.splines.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import spline

from .conftest import AlgorithmTester, get_data


@pytest.mark.parametrize('rng_seed', (0, 1, 2, 3, 4))
@pytest.mark.parametrize('num_bins', (10, 100, 1000))
def test_mapped_histogram(rng_seed, num_bins):
    """Compares the output with numpy and the bin_mapping with a nieve version."""
    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    rng = np.random.RandomState(rng_seed)
    values = rng.normal(0, 20, 1000)
    np_histogram, np_bin_edges = np.histogram(values, num_bins, density=True)
    histogram, bin_edges, bin_mapping = spline._mapped_histogram(values, num_bins)

    assert_allclose(histogram, np_histogram)
    assert_allclose(bin_edges, np_bin_edges)

    expected_bin_mapping = np.zeros_like(values)
    for i, left_bin in enumerate(bin_edges[:-1]):
        mask = (values >= left_bin) & (values < bin_edges[i + 1])
        expected_bin_mapping[mask] = i
    expected_bin_mapping[values >= bin_edges[-1]] = num_bins - 1

    assert_array_equal(bin_mapping, expected_bin_mapping)


@pytest.mark.parametrize('num_bins', (10, 100, 1000))
def test_assign_weights(num_bins):
    """Ensures weights are correctly mapped from the posterior probability."""
    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    rng = np.random.RandomState(0)
    values = rng.normal(0, 20, 1000)
    histogram, bin_edges, bin_mapping = spline._mapped_histogram(values, num_bins)
    posterior_prob = rng.normal(5, 1, num_bins)
    weights = spline._assign_weights(bin_mapping, posterior_prob, values)

    expected_weights = np.zeros_like(values)
    for i, left_bin in enumerate(bin_edges[:-1]):
        mask = (values >= left_bin) & (values < bin_edges[i + 1])
        expected_weights[mask] = posterior_prob[i]
    expected_weights[values >= bin_edges[-1]] = posterior_prob[-1]

    assert_allclose(weights, expected_weights)


class TestMixtureModel(AlgorithmTester):
    """Class for testing mixture_model baseline."""

    func = spline.mixture_model

    @pytest.mark.parametrize('weight_bool', (True, False))
    def test_unchanged_data(self, data_fixture, weight_bool):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        if weight_bool:
            weights = np.ones_like(y)
        else:
            weights = None
        self._test_unchanged_data(data_fixture, y, None, y, weights=weights)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('weights', 'tol_history'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self._call_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


class TestIRSQR(AlgorithmTester):
    """Class for testing irsqr baseline."""

    func = spline.irsqr

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('weights', 'tol_history'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    @pytest.mark.parametrize('quantile', (-1, 2))
    def test_outside_p_fails(self, quantile):
        """Ensures quantile values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self._call_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


class TestCornerCutting(AlgorithmTester):
    """Class for testing corner_cutting baseline."""

    func = spline.corner_cutting

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=())

    def test_no_x(self):
        """Ensures that function output is similar when no x is input."""
        self._test_algorithm_no_x(
            with_args=(self.y, self.x), without_args=(self.y,), rtol=1e-3
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(
            array_args=(self.y,), list_args=(y_list,), assertion_kwargs={'rtol': 1e-5}
        )
