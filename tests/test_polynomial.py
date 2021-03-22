# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from numpy.testing import assert_array_equal, assert_array_almost_equal

from pybaselines import polynomial

from .conftest import get_data


class TestPoly:
    """Class for testing regular polynomial baseline."""

    output = None
    func = polynomial.poly

    def test_algorithm(self, data_fixture):
        """Ensures that raw data is unchanged by the function."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data, x_data)

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_no_x(self, data_fixture):
        """
        Ensures that function output is same when no x is input.

        Maybe only valid for evenly spaced data, such as used for testing.
        """
        x_data, y_data = get_data()
        output = self.__class__.func(y_data)

        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_list(self, data_fixture):
        """Ensures that function also works with list inputs."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data.tolist(), x_data.tolist())

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    @classmethod
    def _compare_output(cls, values):
        """Ensures the output is the same for each evalutation."""
        if cls.output is not None:
            assert_array_almost_equal(cls.output, values)
        else:
            cls.output = values


class TestModPoly:
    """Class for testing ModPoly baseline."""

    output = None
    func = polynomial.modpoly

    def test_algorithm(self, data_fixture):
        """Ensures that raw data is unchanged by the function."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data, x_data)

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_no_x(self, data_fixture):
        """
        Ensures that function output is same when no x is input.

        Maybe only valid for evenly spaced data, such as used for testing.
        """
        x_data, y_data = get_data()
        output = self.__class__.func(y_data)

        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_list(self, data_fixture):
        """Ensures that function also works with list inputs."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data.tolist(), x_data.tolist())

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    @classmethod
    def _compare_output(cls, values):
        """Ensures the output is the same for each evalutation."""
        if cls.output is not None:
            assert_array_almost_equal(cls.output, values)
        else:
            cls.output = values


class TestIModPoly:
    """Class for testing IModPoly baseline."""

    output = None
    func = polynomial.imodpoly

    def test_algorithm(self, data_fixture):
        """Ensures that raw data is unchanged by the function."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data, x_data)

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_no_x(self, data_fixture):
        """
        Ensures that function output is same when no x is input.

        Maybe only valid for evenly spaced data, such as used for testing.
        """
        x_data, y_data = get_data()
        output = self.__class__.func(y_data)

        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    def test_algorithm_list(self, data_fixture):
        """Ensures that function also works with list inputs."""
        x_data, y_data = get_data()
        output = self.__class__.func(y_data.tolist(), x_data.tolist())

        assert_array_equal(data_fixture[0], x_data)
        assert_array_equal(data_fixture[1], y_data)
        self._compare_output(output[0])

    @classmethod
    def _compare_output(cls, values):
        """Ensures the output is the same for each evalutation."""
        if cls.output is not None:
            assert_array_almost_equal(cls.output, values)
        else:
            cls.output = values
