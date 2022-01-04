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
    def func(*args, **kwargs):
        """Dummy function."""
        if 'data' in kwargs:
            output = kwargs['data']
        else:
            output = args[0]
        return output, {'a': 1}


class DummyAlgorithm:
    """A dummy object to serve as a fake Algorithm subclass."""

    def __init__(self, *args, **kwargs):
        pass

    def func(self, *args, **kwargs):
        """Dummy function."""
        return DummyModule.func(*args, **kwargs)


class TestBaseTesterWorks(BaseTester):
    """Ensures a basic subclass of BaseTester works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'func'
    checked_keys = ['a']


# use xfail rather than pytest.raises since raises does not seem to work for classes
@pytest.mark.xfail(raises=NotImplementedError)
class TestBaseTesterNoFunc(BaseTester):
    """Ensures the BaseTester fails if not setup correctly."""
