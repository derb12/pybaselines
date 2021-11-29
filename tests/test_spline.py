# -*- coding: utf-8 -*-
"""Tests for pybaselines.splines.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import spline, utils, whittaker

from .conftest import AlgorithmTester, get_data


@pytest.mark.parametrize('use_numba', (True, False))
def test_mapped_histogram_simple(use_numba):
    """Compares the output with numpy and the bin_mapping, testing corner cases."""
    num_bins = 10
    values = np.array([0, 0.01, 1, 1.5, 8, 9, 9.1, 10])
    expected_bin_edges = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    expected_bin_mapping = np.array([0, 0, 1, 1, 8, 9, 9, 9], dtype=np.intp)

    np_histogram, np_bin_edges = np.histogram(values, num_bins, density=True)
    assert_allclose(np_bin_edges, expected_bin_edges, rtol=0, atol=1e-12)

    with mock.patch.object(spline, '_HAS_NUMBA', use_numba):
        histogram, bin_edges, bin_mapping = spline._mapped_histogram(values, num_bins)

    assert_allclose(histogram, np_histogram)
    assert_allclose(bin_edges, np_bin_edges)
    assert_array_equal(bin_mapping, expected_bin_mapping)


@pytest.mark.parametrize('rng_seed', (0, 1))
@pytest.mark.parametrize('num_bins', (10, 100, 1000))
@pytest.mark.parametrize('use_numba', (True, False))
def test_mapped_histogram(rng_seed, num_bins, use_numba):
    """Compares the output with numpy and the bin_mapping with a nieve version."""
    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    rng = np.random.RandomState(rng_seed)
    values = rng.normal(0, 20, 1000)
    np_histogram, np_bin_edges = np.histogram(values, num_bins, density=True)
    with mock.patch.object(spline, '_HAS_NUMBA', use_numba):
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


@pytest.mark.parametrize('fraction_pos', (0, 0.4))
@pytest.mark.parametrize('fraction_neg', (0, 0.3))
def test_mixture_pdf(fraction_pos, fraction_neg):
    """Ensures the probability density function for the Gaussian-uniform mixture model is right."""
    x = np.linspace(-5, 10, 1000)
    actual_sigma = 0.5
    sigma = np.log10(actual_sigma)
    # the gaussian should be area-normalized, so set height accordingly
    height = 1 / (actual_sigma * np.sqrt(2 * np.pi))
    expected_gaussian = utils.gaussian(x, height, 0, actual_sigma)

    fraction_gaus = 1 - fraction_pos - fraction_neg
    if fraction_pos > 0:
        pos_uniform = np.zeros_like(x)
        pos_uniform[x >= 0] = 1 / abs(x.max())
    elif fraction_neg > 0:
        pos_uniform = None
    else:
        pos_uniform = 0

    if fraction_neg > 0:
        neg_uniform = np.zeros_like(x)
        neg_uniform[x <= 0] = 1 / abs(x.min())
    elif fraction_pos > 0:
        neg_uniform = None
    else:
        neg_uniform = 0

    output_pdf = spline._mixture_pdf(
        x, fraction_gaus, sigma, fraction_pos, pos_uniform, neg_uniform
    )

    # now ensure neg_uniform and pos_uniform are not None
    if pos_uniform is None:
        pos_uniform = 0
    if neg_uniform is None:
        neg_uniform = 0

    expected_pdf = (
        fraction_gaus * expected_gaussian
        + fraction_pos * pos_uniform
        + fraction_neg * neg_uniform
    )

    assert_allclose(expected_pdf, output_pdf, 1e-12, 1e-12)
    # ensure pdf has an area of 1, ie total probability is 100%; accuracy is limited
    # by number of x-values
    assert_allclose(1.0, np.trapz(output_pdf, x), 1e-3)


def compare_pspline_whittaker(pspline_class, whittaker_func, data, lam=1e5,
                              test_rtol=1e-6, test_atol=1e-12, **kwargs):
    """
    Compares the output of the penalized spline (P-spline) versions of Whittaker functions.

    The number of knots for the P-splines are set to ``len(data) + 1`` and the spline
    degree is set to 0; the result is that the spline basis becomes the identity matrix,
    and the P-spline version should give the same output as the Whittaker version if
    the weighting and linear systems were correctly set up.

    """
    # ensure the Whittaker functions use Scipy since that is what P-splines use
    with mock.patch.object(whittaker, '_HAS_PENTAPY', False):
        whittaker_output = whittaker_func(data, lam=lam, **kwargs)[0]

    spline_output = pspline_class._call_func(
        data, lam=lam, num_knots=len(data) + 1, spline_degree=0, **kwargs
    )[0]

    assert_allclose(spline_output, whittaker_output, rtol=test_rtol, atol=test_atol)


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

    @pytest.mark.parametrize('symmetric', (False, True))
    def test_output(self, symmetric):
        """Ensures that the output has the desired format."""
        if symmetric:
            # make data with both positive and negative peaks; roll so peaks are not overlapping
            y = np.roll(self.y, -50) - np.roll(self.y, 50)
            p = 0.5
        else:
            y = self.y
            p = 0.01
        self._test_output(y, y, p=p, symmetric=symmetric, checked_keys=('weights', 'tol_history'))

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


class TestPsplineAsLS(AlgorithmTester):
    """Class for testing pspline_asls baseline."""

    func = spline.pspline_asls

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

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self._call_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self._call_func(self.y, lam=lam, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self._call_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('p', (0.01, 0.1))
    def test_whittaker_comparison(self, lam, p):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.asls, self.y, lam=lam, p=p)
