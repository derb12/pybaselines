# -*- coding: utf-8 -*-
"""Tests for pybaselines.api.

@author: Donald Erb
Created on July 3, 2021

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines.two_d import (
    api, morphological, optimizers, polynomial, smooth, spline, whittaker
)

from ..conftest import get_data2d


_ALL_CLASSES = (
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
    list[str, ...]
        The list of all public methods of the input class.

    """
    return [method for method in dir(klass) if not method.startswith('_')]


# will be like [('asls', whittaker._Whittaker), ('modpoly', polynomial._Polynomial), ...]
_ALL_CLASSES_AND_METHODS = []
for klass in _ALL_CLASSES:
    for method in get_public_methods(klass):
        _ALL_CLASSES_AND_METHODS.append((method, klass))


class TestBaseline2D:
    """Class for testing the Baseline2D class."""

    algorithm_base = api.Baseline2D

    @classmethod
    def setup_class(cls):
        """Sets up the class for testing."""
        cls.x, cls.z, cls.y = get_data2d()
        cls.algorithm = cls.algorithm_base(cls.x, cls.z, check_finite=False, assume_sorted=True)

    @classmethod
    def teardown_class(cls):
        """
        Resets class attributes after testing.

        Probably not needed, but done anyway to catch changes in how pytest works.

        """
        cls.x = None
        cls.z = None
        cls.y = None
        cls.algorithm = None

    @pytest.mark.parametrize('method_and_class', _ALL_CLASSES_AND_METHODS)
    def test_all_methods(self, method_and_class):
        """Ensures all available methods work the same when accessing through Baseline class."""
        method, baseline_class = method_and_class
        # collab_pls needs 2D input data
        if method == 'collab_pls':
            fit_data = np.array((self.y, self.y))
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
            baseline_class(self.x, self.z, check_finite=False, assume_sorted=True), method
        )(fit_data, **kwargs)

        assert_allclose(api_baseline, class_baseline, rtol=1e-14, atol=1e-14)
        assert len(api_params.keys()) == len(class_params.keys())
        for key, value in api_params.items():
            assert key in class_params
            class_value = class_params[key]
            if isinstance(value, (int, float, np.ndarray, list, tuple)):
                assert_allclose(value, class_value, rtol=1e-14, atol=1e-14)
            else:
                assert value == class_value

    def test_method_availability(self):
        """Ensures all public algorithms are available through the Baseline class."""
        total_methods_list = get_public_methods(api.Baseline2D)
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
