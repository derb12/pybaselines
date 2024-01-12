# -*- coding: utf-8 -*-
"""Meta-tests for checking that AlgorithmTester from testing/conftest.py works as intended.

@author: Donald Erb
Created on March 22, 2021

"""

from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_allclose
import pytest

from .conftest import (
    BaseTester, BaseTester2D, BasePolyTester, InputWeightsMixin, dummy_wrapper, get_data,
    get_data2d
)


class DummyModule:
    """A dummy object to serve as a fake module."""

    @staticmethod
    def good_func(data=None, x_data=None, **kwargs):
        """Dummy function."""
        return np.asarray(data), {'a': 1}

    @staticmethod
    def good_func2(data=None, x_data=None, **kwargs):
        """Dummy function."""
        return np.asarray(data), {}

    @staticmethod
    def good_poly_func(data, x_data=None, return_coef=False, **kwargs):
        """A good polynomial algorithm."""
        if x_data is None:
            x = np.linspace(-1, 1, len(data))
            original_domain = np.array([-1, 1])
        else:
            original_domain = np.polynomial.polyutils.getdomain(x_data)
            x = np.polynomial.polyutils.mapdomain(
                x_data, original_domain, np.array([-1., 1.])
            )

        polynomial = np.polynomial.Polynomial.fit(data, x, 1)
        baseline = polynomial(x)
        params = {'a': 1}
        if return_coef:
            params['coef'] = polynomial.convert(window=original_domain).coef

        return baseline, params

    @staticmethod
    def bad_poly_func(data, x_data=None, return_coef=False, **kwargs):
        """A bad polynomial algorithm."""
        params = {'a': 1}
        if not return_coef:
            params['coef'] = np.zeros(5)

        return np.ones_like(data), params

    @staticmethod
    def good_weights_func(data, x_data=None, weights=None, **kwargs):
        """A good algorithm that can take weights."""
        return np.ones_like(data), {'a': 1, 'weights': np.ones_like(data)}

    @staticmethod
    def good_mask_func(data, x_data=None, weights=None, **kwargs):
        """A good algorithm that can take weights and outputs them as the 'mask' key."""
        return np.ones_like(data), {'a': 1, 'mask': np.ones_like(data)}

    @staticmethod
    def bad_weights_func(data, x_data=None, weights=None, **kwargs):
        """An algorithm that incorrectly uses weights."""
        return np.ones_like(data), {'a': 1, 'weights': np.arange(len(data))}

    @staticmethod
    def bad_weights_func_no_weights(data, x_data=None, weights=None, **kwargs):
        """An algorithm that does not include weights in the output parameters."""
        return np.ones_like(data), {'a': 1}

    @staticmethod
    def change_y(data, x_data=None):
        """Changes the input data values, which is unwanted."""
        data[0] = 200000
        return data, {}

    @staticmethod
    def change_x(data, x_data=None):
        """Changes the input x-data values, which is unwanted."""
        x_data[0] = 200000
        return data, {}

    @staticmethod
    def different_output(data, x_data=None):
        """Has different behavior based on the input data type, which is unwanted."""
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.asarray(data) * 20

    @staticmethod
    def single_output(data, x_data=None):
        """Does not include the parameter dictionary output, which is unwanted."""
        return data

    @staticmethod
    def output_list(data, x_data=None):
        """Returns a list rather than a numpy array, which is unwanted."""
        return [0, 1, 2, 3], {}

    @staticmethod
    def output_nondict(data, x_data=None):
        """The second output is not a dictionary, which is unwanted."""
        return data, []

    @staticmethod
    def output_wrong_shape(data, x_data=None):
        """The returned array has a different shape than the input data, which is unwanted."""
        return data[:-1], {}

    @staticmethod
    def no_wrapper(data, x_data=None):
        """A function without the correct wrapper."""
        return data, {}

    @staticmethod
    def repitition_changes(data, x_data=None):
        """
        Changes the output with repeated calls.

        Not actually used, only the class interface is used.

        """
        return data, {}

    @staticmethod
    def no_func(data=None, x_data=None, *args, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')

    @staticmethod
    def no_x(data=None, *args, **kwargs):
        """A module function without an x_data input."""
        return data, {}

    @staticmethod
    def different_kwargs(data=None, x_data=None, a=10, b=12):
        """A module function with different parameter names than the class function."""
        return data, {}

    @staticmethod
    def different_defaults(data=None, x_data=None, a=10, b=12):
        """A module function with different parameter defaults than the class function."""
        return data, {}

    @staticmethod
    def different_function_output(data=None, x_data=None, a=10, b=12):
        """A module function with different output than the class function."""
        return 10 * data, {}

    @staticmethod
    def different_output_params(data=None, x_data=None, a=10, b=12):
        """A module function with different output params than the class function."""
        return data, {'b': 10}

    @staticmethod
    def different_x_output(data=None, x_data=None):
        """Gives different output depending on the x-values."""
        if x_data is None:
            return data, {}
        else:
            return 10 * data, {}

    @staticmethod
    def different_x_ordering(data=None, x_data=None):
        """Gives different output depending on the x-value sorting."""
        return data[np.argsort(x_data)], {}


class DummyAlgorithm:
    """A dummy object to serve as a fake Algorithm and Algorithm2D subclass."""

    def __init__(self, x_data=None, z_data=None, *args, **kwargs):
        self.x = x_data
        self.z = z_data
        self.calls = 0

    @dummy_wrapper
    def good_func(self, data=None, **kwargs):
        """Dummy function."""
        return DummyModule.good_func(data=data, **kwargs)

    @dummy_wrapper
    def good_func2(self, data=None, **kwargs):
        """Dummy function."""
        return DummyModule.good_func2(data=data, **kwargs)

    @dummy_wrapper
    def good_poly_func(self, data, return_coef=False, **kwargs):
        """A good polynomial algorithm."""
        return DummyModule.good_poly_func(
            data=data, x_data=self.x, return_coef=return_coef, **kwargs
        )

    @dummy_wrapper
    def bad_poly_func(self, data, return_coef=False, **kwargs):
        """A bad polynomial algorithm."""
        return DummyModule.bad_poly_func(
            data=data, x_data=self.x, return_coef=return_coef, **kwargs
        )

    @dummy_wrapper
    def good_weights_func(self, data, weights=None, **kwargs):
        """A good algorithm that can take weights."""
        return DummyModule.good_weights_func(
            data=data, x_data=self.x, weights=weights, **kwargs
        )

    @dummy_wrapper
    def good_mask_func(self, data, weights=None, **kwargs):
        """A good algorithm that can take weights and outputs them as the 'mask' key."""
        return DummyModule.good_mask_func(
            data=data, x_data=self.x, weights=weights, **kwargs
        )

    @dummy_wrapper
    def bad_weights_func(self, data, weights=None, **kwargs):
        """An algorithm that incorrectly uses weights."""
        return DummyModule.bad_weights_func(
            data=data, x_data=self.x, weights=weights, **kwargs
        )

    @dummy_wrapper
    def bad_weights_func_no_weights(self, data, weights=None, **kwargs):
        """An algorithm that does not include weights in the output parameters."""
        return DummyModule.bad_weights_func_no_weights(
            data=data, x_data=self.x, weights=weights, **kwargs
        )

    @dummy_wrapper
    def change_y(self, data):
        """Changes the input data values, which is unwanted."""
        data[0] = 200000
        return data, {}

    @dummy_wrapper
    def change_x(self, data):
        """Changes the input x-data values, which is unwanted."""
        self.x[0] = self.x[0] + 5
        return data, {}

    @dummy_wrapper
    def change_z(self, data):
        """Changes the input x-data values, which is unwanted."""
        self.z[0] += 5
        return data, {}

    @dummy_wrapper
    def different_output(self, data):
        """Has different behavior based on the input data type, which is unwanted."""
        if isinstance(data, np.ndarray):
            return data
        else:
            return np.asarray(data) * 20

    @dummy_wrapper
    def single_output(self, data):
        """Does not include the parameter dictionary output, which is unwanted."""
        return data

    @dummy_wrapper
    def output_list(self, data):
        """Returns a list rather than a numpy array, which is unwanted."""
        return [0, 1, 2, 3], {}

    @dummy_wrapper
    def output_nondict(self, data):
        """The second output is not a dictionary, which is unwanted."""
        return data, []

    @dummy_wrapper
    def output_wrong_shape(self, data):
        """The returned array has a different shape than the input data, which is unwanted."""
        return data[:-1], {}

    def no_wrapper(self, data):
        """A function without the correct wrapper."""
        return data, {}

    @dummy_wrapper
    def repitition_changes(self, data):
        """Changes the output with repeated calls."""
        if self.calls == 0:
            output = (data, {})
        else:
            output = (10 * data, {})
        self.calls += 1
        return output

    @dummy_wrapper
    def no_func(self, data=None, *args, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')

    @dummy_wrapper
    def no_x(self, data=None, *args, **kwargs):
        """A module function without an x_data input."""
        return data, {}

    @dummy_wrapper
    def different_kwargs(self, data=None, a=10, c=12):
        """A module function with different parameter names than the module function."""
        return data, {}

    @dummy_wrapper
    def different_defaults(self, data=None, a=10, b=120):
        """A module function with different parameter defaults than the module function."""
        return data, {}

    @dummy_wrapper
    def different_function_output(self, data=None, a=10, b=12):
        """A module function with different output than the class function."""
        return data, {}

    @dummy_wrapper
    def different_output_params(self, data=None, a=10, b=12):
        """A module function with different output params than the class function."""
        return data, {'a': 10}

    @dummy_wrapper
    def different_x_output(self, data=None):
        """Gives different output depending on the x-values."""
        if self.x is None:
            return data, {}
        else:
            return 10 * data, {}

    @dummy_wrapper
    def different_x_ordering(self, data=None):
        """Gives different output depending on the x-value sorting."""
        return data[np.argsort(self.x)], {}

    @dummy_wrapper
    def different_z_ordering(self, data=None):
        """Gives different output depending on the z-value sorting."""
        return data[(..., np.argsort(self.z))], {}

    @dummy_wrapper
    def different_xz_output(self, data=None):
        """Gives different output depending on the x-values and z-values."""
        if self.x is None or self.z is None:
            return data, {}
        else:
            return 10 * data, {}

    @dummy_wrapper
    def different_xz_ordering(self, data=None):
        """Gives different output depending on the x-value and z-value sorting."""
        return data[np.argsort(self.x)[:, None], np.argsort(self.z)[None, :]], {}


class TestBaseTesterWorks(BaseTester):
    """Ensures a basic subclass of BaseTester works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_func'
    checked_keys = ['a']
    required_kwargs = {'key': 1}

    def test_setup(self):
        """Ensures the `setup_class` class method is done correctly."""
        expected_x, expected_y = get_data()
        assert_allclose(self.x, expected_x, rtol=1e-14, atol=1e-14)
        assert_allclose(self.y, expected_y, rtol=1e-14, atol=1e-14)
        assert callable(self.func)
        assert issubclass(self.algorithm_base, DummyAlgorithm)
        assert isinstance(self.algorithm, DummyAlgorithm)
        assert callable(self.class_func)
        assert self.kwargs == {'key': 1}
        assert self.param_keys == ['a']
        assert not self.two_d

    def test_reverse_array(self):
        """Ensures the reverse_array funcion works correctly."""
        assert_allclose(self.reverse_array(self.y), self.y[..., ::-1])


class TestBaseTesterWorks2d(BaseTester):
    """
    Ensures a basic subclass of BaseTester works for a two dimensional algorithm.

    Note: this is for one dimensional algorithms that take two dimensional data, not
    for two dimensional algorithms.
    """

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_func2'
    two_d = True

    def test_setup(self):
        """Ensures the `setup_class` class method is done correctly."""
        expected_x, expected_y = get_data()
        assert_allclose(self.x, expected_x, rtol=1e-14, atol=1e-14)
        assert_allclose(self.y, np.vstack((expected_y, expected_y)), rtol=1e-14, atol=1e-14)
        assert self.kwargs == {}
        assert self.param_keys == []
        assert self.two_d


    def test_reverse_array(self):
        """Ensures the reverse_array funcion works correctly."""
        assert_allclose(self.reverse_array(self.y), self.y[..., ::-1])


class TestBaseTesterFailures(BaseTester):
    """Tests the various BaseTester methods for functions with incorrect output."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'no_func'

    @contextmanager
    def set_func(self, func_name, checked_keys=None):
        """Temporarily sets a new function for the class."""
        original_name = self.func_name
        original_keys = self.param_keys
        try:
            self.__class__.func_name = func_name
            self.__class__.checked_keys = checked_keys
            self.__class__.setup_class()
            yield self
        finally:
            self.__class__.func_name = original_name
            self.__class__.checked_keys = original_keys
            self.__class__.setup_class()

    def test_ensure_wrapped(self):
        """Ensures no wrapper fails."""
        with self.set_func('no_wrapper'):
            with pytest.raises(AssertionError):
                super().test_ensure_wrapped()

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('func', ('change_x', 'change_y'))
    def test_unchanged_data(self, use_class, func):
        """Ensures changing the x and y data fails."""
        with self.set_func(func):
            with pytest.raises(AssertionError):
                super().test_unchanged_data(use_class)

    def test_repeated_fits(self):
        """Ensures no wrapper fails."""
        with self.set_func('repitition_changes'):
            with pytest.raises(AssertionError):
                super().test_repeated_fits()

    def test_list_input(self):
        """Ensures test fails when func gives different outputs for different input types."""
        with self.set_func('different_output'):
            with pytest.raises(AssertionError):
                super().test_list_input()

    def test_functional_vs_class_parameters(self):
        """Ensures test fails when class and functional parameters are different."""
        with self.set_func('no_x'):
            with pytest.raises(AssertionError):
                super().test_functional_vs_class_parameters()

        with self.set_func('different_kwargs'):
            with pytest.raises(AssertionError):
                super().test_functional_vs_class_parameters()

        with self.set_func('different_defaults'):
            with pytest.raises(AssertionError):
                super().test_functional_vs_class_parameters()

    def test_functional_vs_class_output(self):
        """Ensures test fails when functional and class outputs are different."""
        with self.set_func('different_function_output'):
            with pytest.raises(AssertionError):
                super().test_functional_vs_class_output()

        with self.set_func('different_output_params'):
            with pytest.raises(AssertionError):
                super().test_functional_vs_class_output()

    def test_no_x(self):
        """Ensures failure occurs when output changes when no x is given."""
        with self.set_func('different_x_output'):
            with pytest.raises(AssertionError):
                super().test_no_x()

    def test_output(self):
        """Ensures failure occurs when the output is not correct."""
        with self.set_func('single_output'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_list'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_nondict'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_wrong_shape'):
            with pytest.raises(AssertionError):
                super().test_output()

        # also ensure keys are checked
        with self.set_func('good_func'):
            with pytest.raises(AssertionError):
                super().test_output()
            with pytest.raises(AssertionError):
                super().test_output(additional_keys=['b', 'c'])

        with self.set_func('good_func', checked_keys=('a', 'b')):
            with pytest.raises(AssertionError):
                super().test_output()

    def test_x_ordering(self):
        """Ensures failure when output is dependent on x-value sorting."""
        with self.set_func('different_x_ordering'):
            with pytest.raises(AssertionError):
                super().test_x_ordering()


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

    def test_x_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        with pytest.raises(NotImplementedError):
            super().test_x_ordering()


class TestBasePolyTesterWorks(BasePolyTester):
    """Ensures a basic subclass of BaseTester works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_poly_func'
    checked_keys = ['a']


class TestBasePolyTesterFailures(BasePolyTester):
    """Tests the various BasePolyTester methods for functions with incorrect output."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'bad_poly_func'
    checked_keys = ['a']

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures failure if the coefficients are not correctly returned."""
        with pytest.raises(AssertionError):
            super().test_output(return_coef=return_coef)

    def test_output_coefs(self):
        """Ensures failure if the coefficients cannot recreate the output baseline."""
        with pytest.raises(AssertionError):
            super().test_output_coefs()


class TestInputWeightsMixinWorks(BaseTester, InputWeightsMixin):
    """Ensures a basic subclass of InputWeightsMixin works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_weights_func'
    checked_keys = ['a', 'weights']
    required_kwargs = {'key': 1}

    @contextmanager
    def set_func(self, func_name, checked_keys=None, weight_key=None):
        """Temporarily sets a new function for the class."""
        original_name = self.func_name
        original_keys = self.param_keys
        original_weight_key = self.weight_keys
        try:
            self.__class__.func_name = func_name
            self.__class__.checked_keys = checked_keys
            self.__class__.weight_keys = weight_key
            self.__class__.setup_class()
            yield self
        finally:
            self.__class__.func_name = original_name
            self.__class__.checked_keys = original_keys
            self.__class__.weight_keys = original_weight_key
            self.__class__.setup_class()

    def test_input_weights(self):
        """Ensures weight testing works for different weight keys in the parameter dictionary."""
        super().test_input_weights()
        with self.set_func('good_mask_func', weight_key=('mask',), checked_keys=('a', 'mask')):
            super().test_input_weights()


class TestInputWeightsMixinFails(BaseTester, InputWeightsMixin):
    """Tests the various BasePolyTester methods for functions with incorrect output."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'bad_weights_func'
    checked_keys = ['a', 'weights']
    required_kwargs = {'key': 1}

    @contextmanager
    def set_func(self, func_name, checked_keys=None, weight_key=('weights',)):
        """Temporarily sets a new function for the class."""
        original_name = self.func_name
        original_keys = self.param_keys
        original_weight_key = self.weight_keys
        try:
            self.__class__.func_name = func_name
            self.__class__.checked_keys = checked_keys
            self.__class__.weight_keys = weight_key
            self.__class__.setup_class()
            yield self
        finally:
            self.__class__.func_name = original_name
            self.__class__.checked_keys = original_keys
            self.__class__.weight_keys = original_weight_key
            self.__class__.setup_class()

    def test_input_weights(self):
        """Ensures weight testing works for different weight keys in the parameter dictionary."""
        with pytest.raises(AssertionError):
            super().test_input_weights()

    def test_has_no_weights(self):
        """Ensures failure occurs if the weight key is not present in the parameter dictionary."""
        with self.set_func('bad_weights_func_no_weights', checked_keys=('a',)):
            with pytest.raises(AssertionError):
                super().test_input_weights()


class TestBaseTester2DWorks(BaseTester2D):
    """Ensures a basic subclass of BaseTester2D works."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_func'
    checked_keys = ['a']
    required_kwargs = {'key': 1}

    def test_setup(self):
        """Ensures the `setup_class` class method is done correctly."""
        expected_x, expected_z, expected_y = get_data2d()
        assert_allclose(self.x, expected_x, rtol=1e-14, atol=1e-14)
        assert_allclose(self.z, expected_z, rtol=1e-14, atol=1e-14)
        assert_allclose(self.y, expected_y, rtol=1e-14, atol=1e-14)
        assert issubclass(self.algorithm_base, DummyAlgorithm)
        assert isinstance(self.algorithm, DummyAlgorithm)
        assert callable(self.class_func)
        assert self.kwargs == {'key': 1}
        assert self.param_keys == ['a']
        assert not self.three_d

    def test_reverse_array(self):
        """Ensures the reverse_array funcion works correctly."""
        assert_allclose(self.reverse_array(self.y), self.y[..., ::-1, ::-1])


class TestBaseTester2DWorks3d(BaseTester2D):
    """
    Ensures a basic subclass of BaseTester works for a two dimensional algorithm.

    Note: this is for two dimensional algorithms that take three dimensional data, not
    for three dimensional algorithms.
    """

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'good_func2'
    three_d = True

    def test_setup(self):
        """Ensures the `setup_class` class method is done correctly."""
        expected_x, expected_z, expected_y = get_data2d()
        assert_allclose(self.x, expected_x, rtol=1e-14, atol=1e-14)
        assert_allclose(self.z, expected_z, rtol=1e-14, atol=1e-14)
        assert_allclose(self.y, np.array((expected_y, expected_y)), rtol=1e-14, atol=1e-14)
        assert self.kwargs == {}
        assert self.param_keys == []
        assert self.three_d

    def test_reverse_array(self):
        """Ensures the reverse_array funcion works correctly."""
        assert_allclose(self.reverse_array(self.y), self.y[..., ::-1, ::-1])


class TestBaseTester2DFailures(BaseTester2D):
    """Tests the various BaseTester2D methods for functions with incorrect output."""

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'no_func'

    @contextmanager
    def set_func(self, func_name, checked_keys=None):
        """Temporarily sets a new function for the class."""
        original_name = self.func_name
        original_keys = self.param_keys
        try:
            self.__class__.func_name = func_name
            self.__class__.checked_keys = checked_keys
            self.__class__.setup_class()
            yield self
        finally:
            self.__class__.func_name = original_name
            self.__class__.checked_keys = original_keys
            self.__class__.setup_class()

    def test_ensure_wrapped(self):
        """Ensures no wrapper fails."""
        with self.set_func('no_wrapper'):
            with pytest.raises(AssertionError):
                super().test_ensure_wrapped()

    @pytest.mark.parametrize('new_instance', (True, False))
    @pytest.mark.parametrize('func', ('change_x', 'change_y', 'change_z'))
    def test_unchanged_data(self, new_instance, func):
        """Ensures changing the x and y data fails."""
        with self.set_func(func):
            with pytest.raises(AssertionError):
                super().test_unchanged_data(new_instance)

    def test_repeated_fits(self):
        """Ensures no wrapper fails."""
        with self.set_func('repitition_changes'):
            with pytest.raises(AssertionError):
                super().test_repeated_fits()

    def test_list_input(self):
        """Ensures test fails when func gives different outputs for different input types."""
        with self.set_func('different_output'):
            with pytest.raises(AssertionError):
                super().test_list_input()

    @pytest.mark.parametrize('has_x', (True, False))
    @pytest.mark.parametrize('has_z', (True, False))
    def test_no_xz(self, has_x, has_z):
        """Ensures failure occurs when output changes when no x or z is given."""
        if has_x and has_z:
            return  # the one test case that would not produce any difference, so just skip
        with self.set_func('different_xz_output'):
            with pytest.raises(AssertionError):
                super().test_no_xz(has_x, has_z)

    def test_output(self):
        """Ensures failure occurs when the output is not correct."""
        with self.set_func('single_output'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_list'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_nondict'):
            with pytest.raises(AssertionError):
                super().test_output()

        with self.set_func('output_wrong_shape'):
            with pytest.raises(AssertionError):
                super().test_output()

        # also ensure keys are checked
        with self.set_func('good_func'):
            with pytest.raises(AssertionError):
                super().test_output()
            with pytest.raises(AssertionError):
                super().test_output(additional_keys=['b', 'c'])

        with self.set_func('good_func', checked_keys=('a', 'b')):
            with pytest.raises(AssertionError):
                super().test_output()

    @pytest.mark.parametrize('func',
        ('different_x_ordering', 'different_z_ordering', 'different_xz_ordering')
    )
    def test_xz_ordering(self, func):
        """Ensures failure when output is dependent on x-value sorting."""
        with self.set_func(func):
            with pytest.raises(AssertionError):
                super().test_xz_ordering()


class TestBaseTester2DNoFunc(BaseTester2D):
    """Ensures the BaseTester2D fails if not setup correctly."""

    @pytest.mark.parametrize('new_instance', (True, False))
    def test_unchanged_data(self, new_instance):
        """Ensures that input data is unchanged by the function."""
        with pytest.raises(NotImplementedError):
            super().test_unchanged_data(new_instance)

    def test_repeated_fits(self):
        """Ensures the setup is properly reset when using class api."""
        with pytest.raises(NotImplementedError):
            super().test_repeated_fits()

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        with pytest.raises(NotImplementedError):
            super().test_list_input()

    @pytest.mark.parametrize('has_x', (True, False))
    @pytest.mark.parametrize('has_z', (True, False))
    def test_no_xz(self, has_x, has_z):
        """Ensures that function output is the same when no x or z is input."""
        if has_x and has_z:
            return  # the one test case that would not produce any difference, so just skip
        with pytest.raises(NotImplementedError):
            super().test_no_xz(has_x, has_z)

    def test_output(self):
        """Ensures that the output has the desired format."""
        with pytest.raises(NotImplementedError):
            super().test_output()

    def test_xz_ordering(self):
        """Ensures arrays are correctly sorted within the function."""
        with pytest.raises(NotImplementedError):
            super().test_xz_ordering()
