# -*- coding: utf-8 -*-
"""Tests for pybaselines.api.

@author: Donald Erb
Created on July 3, 2021

"""

import inspect
import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import (
    api, classification, misc, morphological, optimizers, polynomial, smooth, spline, whittaker
)

from .base_tests import get_data, check_param_keys, ensure_deprecation


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


def pickle_and_check(fitter, banded_solver, polynomial, spline_basis, validated_x):
    """Pickles the Baseline object and ensures the loaded object matches its initial state."""
    try:
        bytestream = pickle.dumps(fitter)
        loaded_fitter = pickle.loads(bytestream)
    except Exception as e:
        raise AssertionError('pickle failed to save and reload the object') from e

    # ensure attributes are maintained
    assert loaded_fitter.banded_solver == banded_solver
    assert loaded_fitter._validated_x is validated_x

    if polynomial is None:
        assert loaded_fitter._polynomial is None
    else:
        assert_array_equal(loaded_fitter._polynomial.poly_order, polynomial.poly_order)
        if polynomial.vandermonde is not None:
            assert_allclose(
                loaded_fitter._polynomial.vandermonde, polynomial.vandermonde, rtol=1e-12,
                atol=1e-12
            )
        if polynomial._pseudo_inverse is not None:
            assert_allclose(
                loaded_fitter._polynomial.pseudo_inverse, polynomial.pseudo_inverse, rtol=1e-12,
                atol=1e-12
            )
    if spline_basis is None:
        assert loaded_fitter._spline_basis is None
    else:
        assert_allclose(
            loaded_fitter._spline_basis.x, fitter.x, rtol=1e-12,
            atol=1e-12
        )
        assert_array_equal(
            loaded_fitter._spline_basis.spline_degree, spline_basis.spline_degree
        )
        assert_array_equal(loaded_fitter._spline_basis.num_knots, spline_basis.num_knots)
        assert_allclose(
            loaded_fitter._spline_basis.knots, spline_basis.knots, rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            loaded_fitter._spline_basis.basis.toarray(), spline_basis.basis.toarray(),
            rtol=1e-12, atol=1e-12
        )


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

    @ensure_deprecation(1, 5)  # remove the warnings filters after version 1.5
    @pytest.mark.filterwarnings('ignore:"pspline_mpls" is deprecated')
    @pytest.mark.filterwarnings('ignore:"interp_pts" is deprecated')
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

    @pytest.mark.parametrize('input_x', (True, False))
    def test_ensure_pickleable(self, input_x):
        """Ensures that Baseline objects are able to be pickled for all baseline types.

        In order to be used with multiprocessing, objects must be able to be serialized with
        pickle; ensure this works for all algorithm setups.

        """
        x_data = self.x if input_x else None
        fitter = self.algorithm_base(x_data)
        x_validated = not input_x

        pickle_and_check(fitter, 2, None, None, x_validated)
        # call a polynomial method that does not require unique x to set polynomial attribute
        fitter.modpoly(self.y)
        pickle_and_check(fitter, 2, fitter._polynomial, None, x_validated)

        # call a polynomial method that requires unique x
        fitter.loess(self.y)
        pickle_and_check(fitter, 2, fitter._polynomial, None, True)

        # change banded solver
        fitter.banded_solver = 1
        pickle_and_check(fitter, 1, fitter._polynomial, None, True)

        # call a spline method to set the spline basis attribute
        fitter.mixture_model(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        # call other types of methods that don't change internal states
        fitter.arpls(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        fitter.mor(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        fitter.snip(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        fitter.std_distribution(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        fitter.beads(self.y)
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)

        fitter.optimize_extended_range(self.y, method='asls')
        pickle_and_check(fitter, 1, fitter._polynomial, fitter._spline_basis, True)


@ensure_deprecation(1, 4)  # remove the warnings filter once pspline_mpls is removed
@pytest.mark.filterwarnings('ignore:"pspline_mpls" is deprecated')
def test_tck(data_fixture):
    """Ensures all penalized spline methods return 'tck' in the output params."""
    methods = []
    for (method_name, method) in inspect.getmembers(api.Baseline):
        if (
            inspect.isfunction(method)
            and not method_name.startswith('_')
            and (
                'num_knots' in inspect.signature(method).parameters.keys()
                or 'spline_degree' in inspect.signature(method).parameters.keys()
            )
        ):
            methods.append(method_name)
    x, y = data_fixture
    fitter = api.Baseline(x)
    failures = []
    for method in methods:
        _, params = getattr(fitter, method)(y)
        if 'tck' not in params:
            failures.append(method)

    if failures:
        raise AssertionError(f'"tck" not in output params for {failures}')
