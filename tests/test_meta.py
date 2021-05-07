# -*- coding: utf-8 -*-
"""Meta-tests for checking that AlgorithmTester from testing/conftest.py actually works.

@author: Donald Erb
Created on March 22, 2021

"""

import numpy as np
import pytest

from .conftest import AlgorithmTester, get_data


def _change_y(data, x_data=None):
    data[0] = 200000
    return data, {}


def _change_x(data, x_data):
    x_data[0] = 200000
    return data, {}


def _different_output(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.asarray(data) * 20


def _single_output(data):
    return data


def _output_list(data):
    return [0, 1, 2, 3], {}


def _output_nondict(data):
    return data, []


def _output_wrong_shape(data):
    return data[:-1], {}


def _good_output(data, x_data=None):
    return data, {}


class TestAlgorithmTesterPasses(AlgorithmTester):

    func = _good_output

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_no_x(self):
        super()._test_algorithm_no_x(with_args=(self.y, self.x), without_args=(self.y,))

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAlgorithmTesterFailures(AlgorithmTester):

    def test_changed_y_fails(self, data_fixture):
        self.__class__.func = _change_y
        x, y = get_data()
        with pytest.raises(AssertionError):
            super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_changed_x_fails(self, data_fixture):
        self.__class__.func = _change_x
        x, y = get_data()
        with pytest.raises(AssertionError):
            super()._test_unchanged_data(data_fixture, y, x, y, x)

    def test_list_input_fails(self):
        self.__class__.func = _different_output
        y_list = self.y.tolist()
        with pytest.raises(AssertionError):
            super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_one_output_fails(self):
        self.__class__.func = _single_output
        with pytest.raises(AssertionError):
            super()._test_output(self.y, self.y)

    def test_nonarray_output_fails(self):
        self.__class__.func = _output_list
        with pytest.raises(AssertionError):
            super()._test_output(self.y, self.y)

    def test_nondict_output_fails(self):
        self.__class__.func = _output_nondict
        with pytest.raises(AssertionError):
            super()._test_output(self.y, self.y)

    def test_wrong_output_shape_fails(self):
        self.__class__.func = _output_wrong_shape
        with pytest.raises(AssertionError):
            super()._test_output(self.y, self.y)


class TestAlgorithmTesterNoFunc(AlgorithmTester):

    def test_func_not_implemented(self, data_fixture):
        with pytest.raises(NotImplementedError):
            super()._test_unchanged_data(data_fixture)
