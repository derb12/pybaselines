# -*- coding: utf-8 -*-
"""Tests for pybaselines.morphological.

@author: Donald Erb
Created on March 20, 2021

"""

from pybaselines import morphological

from .conftest import get_data, AlgorithmTester


class TestMPLS(AlgorithmTester):
    """Class for testing mpls baseline."""

    func = morphological.mpls

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestMor(AlgorithmTester):
    """Class for testing mor baseline."""

    func = morphological.mor

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestIMor(AlgorithmTester):
    """Class for testing imor baseline."""

    func = morphological.imor

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestAMorMol(AlgorithmTester):
    """Class for testing amormol baseline."""

    func = morphological.amormol

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestMorMol(AlgorithmTester):
    """Class for testing mormol baseline."""

    func = morphological.mormol

    def test_unchanged_data(self, data_fixture):
        x, y = get_data()
        super()._test_unchanged_data(data_fixture, y, None, y)

    def test_output(self):
        super()._test_output(self.y, self.y)

    def test_list_input(self):
        y_list = self.y.tolist()
        super()._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
