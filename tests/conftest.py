# -*- coding: utf-8 -*-
"""Setup code for testing pybaselines.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pytest


def gaussian(x, height=1.0, center=0.0, sigma=1.0):
    """
    Generates a gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center : float, optional
        The center of the distribution. Default is 0.0.
    sigma : float, optional
        The standard deviation of the distribution. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The gaussian distribution evaluated with x.

    Notes
    -----
    This is the same code as in pybaselines.utils.gaussian, but
    this removes the dependence on pybaselines so that if an error
    with pybaselines occurs, this will be unaffected.

    """
    return height * np.exp(-0.5 * ((x - center)**2) / sigma**2)


def get_data():
    """Creates x- and y-data for testing."""
    x_data = np.linspace(1, 100, 1000)
    y_data = (
        500  # constant baseline
        + gaussian(x_data, 10, 25)
        + gaussian(x_data, 20, 50)
        + gaussian(x_data, 10, 75)
    )

    return x_data, y_data


@pytest.fixture(scope='session')
def data_fixture():
    """Test fixture for creating x- and y-data for testing."""
    return get_data()


def _raise_error(*args, **kwargs):
    raise NotImplementedError('must specify func for each subclass')


class AlgorithmTester:
    """
    Abstract class for testing baseline algorithms.

    Attributes
    ----------
    func : callable
        The baseline function to test.
    x, y : numpy.ndarray
        The x- and y-values to use for testing. Should only be used for
        tests where it is known that x and y are unchanged by the function.

    """

    func = _raise_error
    x, y = get_data()

    @classmethod
    def _test_output(cls, y, *args, **kwargs):
        """
        Ensures that the output is correct/consistent.

        Ensures that output has two elements, a numpy array and a param dictionary,
        and that the output baseline is the same shape as the input y-data.

        """
        output = cls.func(*args, **kwargs)

        assert len(output) == 2, 'algorithm output should have two items'
        assert isinstance(output[0], np.ndarray), 'output[0] should be a numpy ndarray'
        assert isinstance(output[1], dict), 'output[1] should be a dictionary'
        assert y.shape == output[0].shape, 'output[0] must have same shape as y-data'

    @classmethod
    def _test_unchanged_data(cls, static_data, y=None, x=None, *args, **kwargs):
        """
        Ensures that input data is unchanged by the function.

        Notes
        -----
        y- and/or x-values should appear in both y=y, x=x, and *args, since the
        actual input of the two values may be different for various functions (see
        example below).

        Examples
        --------
        >>> def test_unchanged_data(self, data_fixture):
        >>>     x, y = get_data()
        >>>     super()._test_unchanged_data(data_fixture, y, x, y, x, lam=100)

        """
        cls.func(*args, **kwargs)

        if y is not None:
            assert_array_equal(
                static_data[1], y, err_msg='the y-data was changed by the algorithm'
            )
        if x is not None:
            assert_array_equal(
                static_data[0], x, err_msg='the x-data was changed by the algorithm'
            )

    @classmethod
    def _test_algorithm_no_x(cls, with_args=(), with_kwargs=None,
                             without_args=(), without_kwargs=None):
        """
        Ensures that function output is same when no x is input.

        Maybe only valid for evenly spaced data, such as used for testing.
        """
        if with_kwargs is None:
            with_kwargs = {}
        if without_kwargs is None:
            without_kwargs = {}

        output_with = cls.func(*with_args, **with_kwargs)
        output_without = cls.func(*without_args, **without_kwargs)

        assert_array_almost_equal(
            output_with[0], output_without[0],
            err_msg='algorithm output is different with no x-values'
        )

    @classmethod
    def _test_algorithm_list(cls, array_args=(), list_args=(), **kwargs):
        """Ensures that function works the same for both array and list inputs."""
        output_array = cls.func(*array_args, **kwargs)
        output_list = cls.func(*list_args, **kwargs)

        assert_array_almost_equal(
            output_array[0], output_list[0],
            err_msg='algorithm output is different for arrays vs lists'
        )
