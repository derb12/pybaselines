# -*- coding: utf-8 -*-
"""Tests for pybaselines.api.

@author: Donald Erb
Created on July 3, 2021

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import (
    api, classification, misc, morphological, optimizers, polynomial, smooth, spline, whittaker
)

from .conftest import get_data, check_param_keys


_ALL_CLASSES = (
    classification._Classification,
    misc._Misc,
    morphological._Morphological,
    optimizers._Optimizers,
    polynomial._Polynomial,
    smooth._Smooth,
    spline._Spline,
    whittaker._Whittaker
)


def get_public_methods(klass):
    """
    Gets all public methods from a class.

    Parameters
    ----------
    klass : class
        The class to use.

    Returns
    -------
    methods : list[str, ...]
        The list of all public methods of the input class.

    """
    methods = []
    for method in dir(klass):
        if (
            not (method.startswith('_')
            or method.startswith('pentapy_solver')
            or method.startswith('banded_solver')
            or method.startswith('get_method'))
        ):
            methods.append(method)
    return methods


# will be like [('asls', whittaker._Whittaker), ('modpoly', polynomial._Polynomial), ...]
_ALL_CLASSES_AND_METHODS = []
for klass in _ALL_CLASSES:
    for method in get_public_methods(klass):
        _ALL_CLASSES_AND_METHODS.append((method, klass))


class TestBaseline:
    """Class for testing the Baseline class."""

    algorithm_base = api.Baseline

    @classmethod
    def setup_class(cls):
        """Sets up the class for testing."""
        cls.x, cls.y = get_data()
        cls.algorithm = cls.algorithm_base(cls.x, check_finite=False, assume_sorted=True)

    @classmethod
    def teardown_class(cls):
        """
        Resets class attributes after testing.

        Probably not needed, but done anyway to catch changes in how pytest works.

        """
        cls.x = None
        cls.y = None
        cls.algorithm = None

    @pytest.mark.parametrize('method_and_class', _ALL_CLASSES_AND_METHODS)
    def test_all_methods(self, method_and_class):
        """Ensures all available methods work the same when accessing through Baseline class."""
        method, baseline_class = method_and_class
        # collab_pls needs 2D input data
        if method == 'collab_pls':
            fit_data = np.vstack((self.y, self.y))
        else:
            fit_data = self.y

        # need to handle some specific methods
        if method == 'optimize_extended_range':
            kwargs = {'method': 'modpoly'}
        elif method == 'interp_pts':
            kwargs = {'baseline_points': ((5, 10), (10, 20), (90, 100))}
        elif method == 'golotvin':
            # have to set kwargs for golotvin or else no baseline points are found
            kwargs = {'half_window': 15, 'num_std': 6}
        else:
            kwargs = {}

        api_baseline, api_params = getattr(self.algorithm, method)(fit_data, **kwargs)
        class_baseline, class_params = getattr(
            baseline_class(self.x, check_finite=False, assume_sorted=True), method
        )(fit_data, **kwargs)

        assert_allclose(api_baseline, class_baseline, rtol=1e-12, atol=1e-12)
        check_param_keys(api_params.keys(), class_params.keys())

    def test_method_availability(self):
        """Ensures all public algorithms are available through the Baseline class."""
        total_methods_list = get_public_methods(api.Baseline)
        total_methods = set(total_methods_list)

        # ensure no repeated methods
        assert len(total_methods) == len(total_methods_list)

        for klass in _ALL_CLASSES:
            assert issubclass(self.algorithm_base, klass)
            class_methods = set(get_public_methods(klass))
            # all individual class methods should be in Baseline
            assert len(class_methods - total_methods) == 0
            total_methods = total_methods - class_methods

        # no additional methods should be available
        assert len(total_methods) == 0

    def test_method_and_function_availability(self):
        """Ensures all Baseline methods are also available as functions."""
        total_methods = get_public_methods(api.Baseline)

        total_functions = []
        for module in (
            classification,
            misc,
            morphological,
            optimizers,
            polynomial,
            smooth,
            spline,
            whittaker
        ):
            total_functions.extend(get_public_methods(module))
        total_functions = set(total_functions)

        # note that total_functions also includes all imported functions from other
        # libraries, so just check that each method from Baseline is in total_functions
        for method in total_methods:
            assert method in total_functions

    def test_get_method(self):
        """Ensures the get_method helper function works as intended."""
        method = self.algorithm._get_method('asls')
        assert method == self.algorithm.asls

        # also ensure capitalization does not matter
        method2 = self.algorithm._get_method('AsLS')
        assert method2 == self.algorithm.asls

    def test_get_method_fails(self):
        """Ensures the get_method helper function fails when an incorrect name is given."""
        with pytest.raises(AttributeError):
            self.algorithm._get_method('aaaaaaaaaaaaa')
