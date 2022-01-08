# -*- coding: utf-8 -*-
"""Setup code for testing pybaselines.

@author: Donald Erb
Created on March 20, 2021

"""

from functools import wraps
import inspect

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest


try:
    import pentapy  # noqa
except ImportError:
    _HAS_PENTAPY = False
else:
    _HAS_PENTAPY = True

has_pentapy = pytest.mark.skipif(not _HAS_PENTAPY, reason='pentapy is not installed')


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
    # TODO use np.random.default_rng(0) once minimum numpy version is >= 1.17
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


@pytest.fixture
def small_data():
    """A small array of data for testing."""
    return np.arange(10, dtype=float)


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

    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Sets the x and y attributes for each class."""
        self.x, self.y = get_data()

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
                             without_args=(), without_kwargs=None,
                             **assertion_kwargs):
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

        assert_allclose(
            output_with[0], output_without[0],
            err_msg='algorithm output is different with no x-values',
            **assertion_kwargs
        )

    @classmethod
    def _test_algorithm_list(cls, array_args=(), list_args=(), assertion_kwargs=None, **kwargs):
        """Ensures that function works the same for both array and list inputs."""
        output_array = cls.func(*array_args, **kwargs)
        output_list = cls.func(*list_args, **kwargs)

        if assertion_kwargs is None:
            assertion_kwargs = {}
        assert_allclose(
            output_array[0], output_list[0],
            err_msg='algorithm output is different for arrays vs lists', **assertion_kwargs
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


def dummy_wrapper(func):
    """A dummy wrapper to simulate using the _Algorithm._register wrapper function."""
    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)
    return inner


class DummyModule:
    """A dummy object to serve as a fake module."""

    @staticmethod
    def func(*args, data=None, x_data=None, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')


class DummyAlgorithm:
    """A dummy object to serve as a fake Algorithm subclass."""

    def __init__(self, *args, **kwargs):
        pass

    @dummy_wrapper
    def func(self, *args, data=None, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')


class BaseTester:
    """
    A base class for testing all algorithms.

    Ensure the functional and class-based algorithms are the same and that both do not
    modify the inputs. After that, only the class-based call is used to potentially save
    time from the setup.

    Attributes
    ----------
    kwargs : dict
        The keyword arguments that will be used as inputs for all default test cases.

    """

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'func'
    checked_keys = None
    required_kwargs = None

    @pytest.fixture(autouse=True)
    def setup_class(self):
        """Sets the x and y attributes for each class."""
        self.x, self.y = get_data()
        self.func = getattr(self.module, self.func_name)
        self.algorithm = self.algorithm_base(self.x)
        self.class_func = getattr(self.algorithm, self.func_name)
        self.kwargs = self.required_kwargs if self.required_kwargs is not None else {}
        self.param_keys = self.checked_keys if self.checked_keys is not None else []

    def test_ensure_wrapped(self):
        """Ensures the class method was wrapped using _Algorithm._register to control inputs."""
        assert hasattr(self.class_func, '__wrapped__')

    @pytest.mark.parametrize('use_class', (True, False))
    def test_unchanged_data(self, use_class, **kwargs):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        x2, y2 = get_data()
        if use_class:
            getattr(self.algorithm_base(x_data=x), self.func_name)(
                data=y, **self.kwargs, **kwargs
            )
        else:
            self.func(data=y, x_data=x, **self.kwargs, **kwargs)

        assert_array_equal(y2, y, err_msg='the y-data was changed by the algorithm')
        assert_array_equal(x2, x, err_msg='the x-data was changed by the algorithm')

    def test_repeated_fits(self):
        """Ensures the setup is properly reset when using class api."""
        first_output = self.class_func(data=self.y, **self.kwargs)
        second_output = self.class_func(data=self.y, **self.kwargs)

        assert_allclose(first_output[0], second_output[0], 1e-14)

    def test_functional_vs_class_output(self, **assertion_kwargs):
        """Ensures the functional and class-based functions perform the same."""
        class_output, class_params = self.class_func(data=self.y, **self.kwargs)
        func_output, func_params = self.func(data=self.y, x_data=self.x, **self.kwargs)

        assert_allclose(class_output, func_output, **assertion_kwargs)
        for key in class_params:
            assert key in func_params

    def test_functional_vs_class_parameters(self):
        """
        Ensures the args and kwargs for functional and class-based functions are the same.

        Also ensures that both api have a `data` argument. The only difference between
        the two signatures should be that the functional api has an `x_data` keyword.

        """
        class_parameters = inspect.signature(self.class_func).parameters
        functional_parameters = inspect.signature(self.func).parameters

        # should be the same except that functional signature has x_data
        assert len(class_parameters) == len(functional_parameters) - 1
        assert 'data' in class_parameters
        assert 'x_data' in functional_parameters
        for key in class_parameters:
            assert key in functional_parameters

    def test_list_input(self, **assertion_kwargs):
        """Ensures that function works the same for both array and list inputs."""
        output_array = self.class_func(data=self.y, **self.kwargs)
        output_list = self.class_func(data=self.y.tolist(), **self.kwargs)

        assert_allclose(
            output_array[0], output_list[0],
            err_msg='algorithm output is different for arrays vs lists', **assertion_kwargs
        )
        for key in output_array[1]:
            assert key in output_list[1]

    def test_no_x(self, **assertion_kwargs):
        """
        Ensures that function output is the same when no x is input.

        Usually only valid for evenly spaced data, such as used for testing.

        """
        output_with = self.class_func(data=self.y, **self.kwargs)
        output_without = getattr(self.algorithm_base(), self.func_name)(
            data=self.y, **self.kwargs
        )

        assert_allclose(
            output_with[0], output_without[0],
            err_msg='algorithm output is different with no x-values',
            **assertion_kwargs
        )

    def test_output(self, additional_keys=None, **kwargs):
        """
        Ensures that the output has the desired format.

        Ensures that output has two elements, a numpy array and a param dictionary,
        and that the output baseline is the same shape as the input y-data.

        Parameters
        ----------
        additional_keys : Iterable(str, ...), optional
            Additional keys to check for in the output parameter dictionary. Default is None.
        **kwargs
            Additional keyword arguments to pass to the function.

        """
        output = self.class_func(data=self.y, **self.kwargs, **kwargs)

        assert len(output) == 2, 'algorithm output should have two items'
        assert isinstance(output[0], np.ndarray), 'output[0] should be a numpy ndarray'
        assert isinstance(output[1], dict), 'output[1] should be a dictionary'
        assert self.y.shape == output[0].shape, 'output[0] must have same shape as y-data'

        if additional_keys is not None:
            total_keys = list(self.param_keys) + list(additional_keys)
        else:
            total_keys = self.param_keys
        # check all entries in output param dictionary
        for key in total_keys:
            if key not in output[1]:
                assert False, f'key "{key}" missing from param dictionary'
            output[1].pop(key)
        if output[1]:
            assert False, f'unchecked keys in param dictionary: {output[1]}'


class BasePolyTester(BaseTester):
    """
    A base class for testing polynomial algorithms.

    Checks that the polynomial coefficients are correctly returned and that they correspond
    to the polynomial used to create the baseline.

    """

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures the polynomial coefficients are output if `return_coef` is True."""
        if return_coef:
            additional_keys = ['coef']
        else:
            additional_keys = None
        super().test_output(additional_keys=additional_keys, return_coef=return_coef)

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self.class_func(data=self.y, **self.kwargs, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)
