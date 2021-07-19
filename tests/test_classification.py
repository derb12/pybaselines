# -*- coding: utf-8 -*-
"""Tests for pybaselines.classification.

@author: Donald Erb
Created on July 3, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import classification

from .conftest import AlgorithmTester, get_data


def _nieve_rolling_std(data, half_window, ddof=0):
    """
    A nieve approach for a rolling standard deviation.

    Used for ensuring faster, more complex approaches are correct.

    Parameters
    ----------
    data : numpy.ndarray
        The array for the calculation. Should be padded on the left and right
        edges by `half_window`.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int, optional
        The degrees of freedom for the calculation. Default is 0.

    Returns
    -------
    rolling_std : numpy.ndarray
        The array of the rolling standard deviation for each window.

    """
    num_y = data.shape[0]
    rolling_std = np.array([
        np.std(data[max(0, i - half_window):min(i + half_window + 1, num_y)], ddof=ddof)
        for i in range(num_y)
    ])

    return rolling_std


@pytest.mark.parametrize('y_scale', (1, 1e-9, 1e9))
@pytest.mark.parametrize('half_window', (1, 3, 10, 30))
@pytest.mark.parametrize('ddof', (0, 1))
def test_rolling_std(y_scale, half_window, ddof):
    """
    Test the rolling standard deviation calculation against a nieve implementation.

    Also tests different y-scales while using the same noise level, since some
    implementations have numerical instability when values are small/large compared
    to the standard deviation.

    """
    x = np.arange(100)
    y = y_scale * np.sin(x) + np.random.normal(0, 0.2, x.size)
    # only compare within [half_window:-half_window] since the calculation
    # can have slightly different values at the edges
    compare_slice = slice(half_window, -half_window)

    actual_rolled_std = _nieve_rolling_std(y, half_window, ddof)
    calc_rolled_std = classification._rolling_std(
        np.pad(y, half_window, 'reflect'), half_window, ddof
    )[half_window:-half_window]

    assert_allclose(calc_rolled_std[compare_slice], actual_rolled_std[compare_slice])


@pytest.mark.parametrize(
    'inputs',
    (
        [[0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0]],
        [[0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0]],
        [[0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1]]
    )
)
def test_remove_single_points(inputs):
    """
    Test that _remove_single_points fills holes in binary mask.

    Lone True values should be removed before lone False values, and
    the edges should convert to False unless there are two True values.

    """
    mask, expected_mask = np.asarray(inputs, bool)
    output_mask = classification._remove_single_points(mask)

    assert_array_equal(expected_mask, output_mask)


class TestGolotvin(AlgorithmTester):
    """Class for testing golotvin baseline."""

    func = classification.golotvin

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, 15, 6)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, self.x, 15, 6, checked_keys=('mask',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        x_list = self.x.tolist()
        self._test_algorithm_list(
            array_args=(self.y, self.x, 15, 6), list_args=(y_list, x_list, 15, 6)
        )


class TestDietrich(AlgorithmTester):
    """Class for testing dietrich baseline."""

    func = classification.dietrich

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures that the output has the desired format."""
        param_keys = ['mask']
        if return_coef:
            param_keys.append('coef')
        self._test_output(self.y, self.y, checked_keys=param_keys, return_coef=return_coef)

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)


class TestNoiseDistribution(AlgorithmTester):
    """Class for testing noise_distribution baseline."""

    func = classification.noise_distribution

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('mask',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
