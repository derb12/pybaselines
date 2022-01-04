# -*- coding: utf-8 -*-
"""Meta-tests for checking that AlgorithmTester from testing/conftest.py works as intended.

@author: Donald Erb
Created on March 22, 2021

"""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from .conftest import AlgorithmTester, BaseTester, get_data


def _change_y(data, x_data=None):
    """Changes the input data values, which is unwanted."""
    data[0] = 200000
    return data, {}


def _change_x(data, x_data):
    """Changes the input x-data values, which is unwanted."""
    x_data[0] = 200000
    return data, {}


def _different_output(data):
    """Has different behavior based on the input data type, which is unwanted."""
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data) * 20


def _single_output(data):
    """Does not include the parameter dictionary output, which is unwanted."""
    return data


def _output_list(data):
    """Returns a list rather than a numpy array, which is unwanted."""
    return [0, 1, 2, 3], {}


def _output_nondict(data):
    """The second output is not a dictionary, which is unwanted."""
    return data, []


def _output_wrong_shape(data):
    """The returned array has a different shape than the input data, which is unwanted."""
    return data[:-1], {}


def _good_output(data, x_data=None):
    """A model function that correctly gives a numpy array and dictionary output."""
    return data, {'param': 1, 'param2': 2}


class TestAlgorithmTesterPasses(AlgorithmTester):
    """Tests the various AlgorithmTester methods for a function with the correct output."""

    func = _good_output

    def test_unchanged_data(self, data_fixture):
        """Ensures test passes when input data is not changed."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        """Ensures test for no x-values passes."""
        self._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        """Ensures test for correct output passes."""
        self._test_output(self.y, self.y)

    def test_output_params(self):
        """Ensures the test passes with the correct parameter keys."""
        self._test_output(self.y, self.x, checked_keys=('param', 'param2'))

    def test_output_params_unchecked_param_fails(self):
        """Ensures test fails when not all parameter keys are given."""
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.x, checked_keys=('param',))

    def test_output_params_unknown_param_fails(self):
        """Ensures test fails when an unknown parameter key is given."""
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.x, checked_keys=('param', 'param2', 'param3'))

    def test_list_input(self):
        """Ensures test passes when output is the same, regardless of input data type."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_call_func(self):
        """Make sure call_func returns the output of the class's function."""
        out = self._call_func(self.y)
        assert_array_equal(self.y, out[0])

    def test_accuracy_passes(self):
        """Ensures test_accuracy passes when the arrays are equal."""
        self._test_accuracy(self.y, self.y)

    def test_accuracy_fails(self):
        """Ensures test_accuracy fails when the arrays are not close."""
        with pytest.raises(AssertionError):
            self._test_accuracy(self.y * 10, self.y)

    def test_accuracy_kwargs(self):
        """Ensures kwargs are correctly passed to test_accuracy."""
        offset_data = self.y + 1e-5
        self._test_accuracy(offset_data, self.y)
        # should fail when tolerances are changed
        with pytest.raises(AssertionError):
            self._test_accuracy(offset_data, self.y, assertion_kwargs={'atol': 1e-6, 'rtol': 0})


class TestAlgorithmTesterFailures(AlgorithmTester):
    """Tests the various AlgorithmTester methods for functions with incorrect output."""

    def test_changed_y_fails(self, data_fixture):
        """Ensures test fails when func changes the input data."""
        self.__class__.func = _change_y
        x, y = get_data()
        with pytest.raises(AssertionError):
            self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_changed_x_fails(self, data_fixture):
        """Ensures test fails when func changes the input x-data."""
        self.__class__.func = _change_x
        x, y = get_data()
        with pytest.raises(AssertionError):
            self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_list_input_fails(self):
        """Ensures test fails when func gives different outputs for different input types."""
        self.__class__.func = _different_output
        y_list = self.y.tolist()
        with pytest.raises(AssertionError):
            self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_one_output_fails(self):
        """Ensures test fails when func outputs only a single item."""
        self.__class__.func = _single_output
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.y)

    def test_nonarray_output_fails(self):
        """Ensures test fails when func does not output a numpy array as the first item."""
        self.__class__.func = _output_list
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.y)

    def test_nondict_output_fails(self):
        """Ensures test fails when func does not output a dictionary as the second item."""
        self.__class__.func = _output_nondict
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.y)

    def test_wrong_output_shape_fails(self):
        """Ensures test fails when func outputs an array with a different shape than the input."""
        self.__class__.func = _output_wrong_shape
        with pytest.raises(AssertionError):
            self._test_output(self.y, self.y)


class TestAlgorithmTesterNoFunc(AlgorithmTester):
    """Ensures the AlgorithmTester fails if the `func` attribute is not set."""

    def test_func_not_implemented(self, data_fixture):
        """Ensures NotImplementedError is raised from AlgorithmTester's default func."""
        with pytest.raises(NotImplementedError):
            self._test_unchanged_data(data_fixture)


class DummyModule:
    """A dummy object to serve as a fake module."""

    @staticmethod
    def func(data=None, x_data=None, **kwargs):
        """Dummy function."""
        return data, {'a': 1}


class DummyAlgorithm:
    """A dummy object to serve as a fake Algorithm subclass."""

    def __init__(self, *args, **kwargs):
        pass

    def func(self, data=None, **kwargs):
        """Dummy function."""
        return DummyModule.func(data=data, **kwargs)


class TestBaseTesterWorks(BaseTester):
    """Ensures a basic subclass of BaseTester works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'func'
    checked_keys = ['a']


class TestBaseTesterNoFunc(BaseTester):
    """Ensures the BaseTester fails if not setup correctly."""

    @pytest.mark.parametrize('use_class', (True, False))
    def test_unchanged_data(self, use_class):
        """Ensures that input data is unchanged by the function."""
        with pytest.raises(NotImplementedError):
            super().test_unchanged_data(use_class)

    def test_repeated_fits(self):
        """Ensures the setup is properly reset when using class api."""
        with pytest.raises(NotImplementedError):
            super().test_repeated_fits()

    def test_functional_vs_class_output(self):
        """Ensures the functional and class-based functions perform the same."""
        with pytest.raises(NotImplementedError):
            super().test_functional_vs_class_output()

    def test_functional_vs_class_parameters(self):
        """
        Ensures the args and kwargs for functional and class-based functions are the same.

        Only test that should actually pass if setup was done incorrectly.
        """
        super().test_functional_vs_class_parameters()

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        with pytest.raises(NotImplementedError):
            super().test_list_input()

    def test_no_x(self):
        """Ensures that function output is the same when no x is input."""
        with pytest.raises(NotImplementedError):
            super().test_no_x()

    def test_output(self):
        """Ensures that the output has the desired format."""
        with pytest.raises(NotImplementedError):
            super().test_output()
