# -*- coding: utf-8 -*-
"""Setup code for testing pybaselines.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal
import pytest


try:
    import pentapy  # noqa
except ImportError:
    no_pentapy = pytest.mark.skipif(False, reason='pentapy is not installed')
    has_pentapy = pytest.mark.skipif(True, reason='pentapy is not installed')
else:
    no_pentapy = pytest.mark.skipif(True, reason='pentapy is installed')
    has_pentapy = pytest.mark.skipif(False, reason='pentapy is installed')

try:
    import numba  # noqa
except ImportError:
    if_has_numba = pytest.mark.skipif(True, reason='numba is not installed')
else:
    if_has_numba = pytest.mark.skipif(False, reason='numba is installed')


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


def get_data(include_noise=True, num_points=1000):
    """Creates x- and y-data for testing.

    Parameters
    ----------
    include_noise : bool, optional
        If True (default), will include noise with the y-data.
    num_points : int, optional
        The number of data points to use. Default is 1000.

    Returns
    -------
    x_data : numpy.ndarray
        The x-values.
    y_data : numpy.ndarray
        The y-values.

    """
    # use np.random.default_rng(0) once minimum numpy version is >= 1.17
    np.random.seed(0)
    x_data = np.linspace(1, 100, num_points)
    y_data = (
        500  # constant baseline
        + gaussian(x_data, 10, 25)
        + gaussian(x_data, 20, 50)
        + gaussian(x_data, 10, 75)
    )
    if include_noise:
        y_data += np.random.normal(0, 0.5, x_data.size)

    return x_data, y_data


@pytest.fixture()
def data_fixture():
    """Test fixture for creating x- and y-data for testing."""
    return get_data()


@pytest.fixture()
def no_noise_data_fixture():
    """Test fixture that creates x- and y-data without noise for testing."""
    return get_data(include_noise=False)


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
    x, y = get_data()  # TODO remove this and make it a per-function call

    @classmethod
    def _test_output(cls, y, *args, checked_keys=None, **kwargs):
        """
        Ensures that the output has the desired format.

        Ensures that output has two elements, a numpy array and a param dictionary,
        and that the output baseline is the same shape as the input y-data.

        Parameters
        ----------
        y : array-like
            The data to pass to the fitting function.
        *args : tuple
            Any arguments to pass to the fitting function.
        checked_keys : Iterable, optional
            The keys to ensure are present in the parameter dictionary output of the
            fitting function. If None (default), will not check the param dictionary.
            Used to track changes to the output params.
        **kwargs : dict
            Any keyword arguments to pass to the fitting function.

        """
        output = cls.func(*args, **kwargs)

        assert len(output) == 2, 'algorithm output should have two items'
        assert isinstance(output[0], np.ndarray), 'output[0] should be a numpy ndarray'
        assert isinstance(output[1], dict), 'output[1] should be a dictionary'
        assert y.shape == output[0].shape, 'output[0] must have same shape as y-data'

        # check all entries in output param dictionary
        if checked_keys is not None:
            for key in checked_keys:
                if key not in output[1]:
                    assert False, f'key "{key}" missing from param dictionary'
                output[1].pop(key)
            if output[1]:
                assert False, f'unchecked keys in param dictionary: {output[1]}'

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
        >>>     self._test_unchanged_data(data_fixture, y, x, y, x, lam=100)

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
        Ensures that function output is the same when no x is input.

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

    @classmethod
    def _call_func(cls, *args, **kwargs):
        """Class method to allow calling the class's function."""
        return cls.func(*args, **kwargs)

    @classmethod
    def _test_accuracy(cls, known_output, *args, assertion_kwargs=None, **kwargs):
        """
        Compares the output of the baseline function to a known output.

        Useful for ensuring results are consistent across versions, or for
        comparing to the output of a method from another library.

        Parameters
        ----------
        known_output : numpy.ndarray
            The output to compare against. Should be from an earlier version if testing
            for changes, or against the output of an established method.
        assertion_kwargs : dict, optional
            A dictionary of keyword arguments to pass to
            :func:`numpy.testing.assert_allclose`. Default is None.

        """
        if assertion_kwargs is None:
            assertion_kwargs = {}
        output = cls.func(*args, **kwargs)[0]

        assert_allclose(output, known_output, **assertion_kwargs)
