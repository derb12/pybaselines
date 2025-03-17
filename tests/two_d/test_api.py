# -*- coding: utf-8 -*-
"""Tests for pybaselines.api.

@author: Donald Erb
Created on July 3, 2021

"""

import pickle

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines.two_d import api, morphological, optimizers, polynomial, smooth, spline, whittaker

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


def pickle_and_check(fitter, banded_solver, polynomial, spline_basis, validated_x, validated_z):
    """Pickles the Baseline2D object and ensures the loaded object matches its initial state."""
    try:
        bytestream = pickle.dumps(fitter)
        loaded_fitter = pickle.loads(bytestream)
    except Exception as e:
        raise AssertionError('pickle failed to save and reload the object') from e

    # ensure attributes are maintained
    assert loaded_fitter.banded_solver == banded_solver
    assert loaded_fitter._validated_x is validated_x
    assert loaded_fitter._validated_z is validated_z

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
        assert_allclose(
            loaded_fitter._spline_basis.z, fitter.z, rtol=1e-12,
            atol=1e-12
        )
        assert_array_equal(
            loaded_fitter._spline_basis.spline_degree, spline_basis.spline_degree
        )
        assert_array_equal(loaded_fitter._spline_basis.num_knots, spline_basis.num_knots)
        assert_allclose(
            loaded_fitter._spline_basis.knots_r, spline_basis.knots_r, rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            loaded_fitter._spline_basis.knots_c, spline_basis.knots_c, rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            loaded_fitter._spline_basis.basis.toarray(), spline_basis.basis.toarray(),
            rtol=1e-12, atol=1e-12
        )


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

        assert_allclose(api_baseline, class_baseline, rtol=1e-12, atol=1e-12)
        assert len(api_params.keys()) == len(class_params.keys())
        for key, value in api_params.items():
            assert key in class_params
            class_value = class_params[key]
            if isinstance(value, (int, float, np.ndarray, list, tuple)):
                assert_allclose(value, class_value, rtol=1e-12, atol=1e-12)
            elif isinstance(value, dict):
                # do not check values of the internal dictionary since the nested structure
                # is no longer guaranteed to be the same shape for every value
                for internal_key in value.keys():
                    assert internal_key in class_value
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
    @pytest.mark.parametrize('input_z', (True, False))
    def test_ensure_pickleable(self, input_x, input_z):
        """Ensures that Baseline2D objects are able to be pickled for all baseline types.

        In order to be used with multiprocessing, objects must be able to be serialized with
        pickle; ensure this works for all algorithm setups.

        """
        x_data = self.x if input_x else None
        z_data = self.z if input_z else None
        fitter = self.algorithm_base(x_data, z_data)
        # no current 2D methods require unique x or z so these won't change state
        x_validated = not input_x
        z_validated = not input_z

        pickle_and_check(fitter, 2, None, None, x_validated, z_validated)
        # call a polynomial method that does not require unique x to set polynomial attribute
        fitter.modpoly(self.y)
        pickle_and_check(fitter, 2, fitter._polynomial, None, x_validated, z_validated)

        # change banded solver
        fitter.banded_solver = 1
        pickle_and_check(fitter, 1, fitter._polynomial, None, x_validated, z_validated)

        # call a spline method to set the spline basis attribute
        fitter.mixture_model(self.y)
        pickle_and_check(
            fitter, 1, fitter._polynomial, fitter._spline_basis, x_validated, z_validated
        )

        # call other types of methods that don't change internal states
        fitter.arpls(self.y)
        pickle_and_check(
            fitter, 1, fitter._polynomial, fitter._spline_basis, x_validated, z_validated
        )

        fitter.mor(self.y)
        pickle_and_check(
            fitter, 1, fitter._polynomial, fitter._spline_basis, x_validated, z_validated
        )

        fitter.noise_median(self.y)
        pickle_and_check(
            fitter, 1, fitter._polynomial, fitter._spline_basis, x_validated, z_validated
        )

        fitter.individual_axes(self.y, method='asls')
        pickle_and_check(
            fitter, 1, fitter._polynomial, fitter._spline_basis, x_validated, z_validated
        )
