# -*- coding: utf-8 -*-
"""Tests for pybaselines._nd.optimizers.

Created on March 30, 2026
@author: Donald Erb

"""

import inspect

import pytest

from pybaselines import Baseline, Baseline2D
from pybaselines._nd import optimizers


@pytest.mark.parametrize(
    'method_and_outputs', (
        ('collab_pls', 'collab_pls', 'optimizers'),
        ('COLLAB_pls', 'collab_pls', 'optimizers'),
        ('modpoly', 'modpoly', 'polynomial'),
        ('asls', 'asls', 'whittaker'),
        ('AsLS', 'asls', 'whittaker')
    )
)
@pytest.mark.parametrize('ensure_new', (True, False))
@pytest.mark.parametrize('two_d', (True, False))
def test_optimizer_helper_method(method_and_outputs, ensure_new, two_d):
    """Ensures _OptimizerHelper recognizes the correct methods."""
    method, expected_method, expected_module = method_and_outputs

    if two_d:
        current_baseline = Baseline2D()
    else:
        current_baseline = Baseline()
    optimizer_helper = optimizers._OptimizerHelper(
        method, current_fitter=current_baseline, ensure_new=ensure_new
    )

    assert optimizer_helper.method == expected_method
    assert optimizer_helper.module == expected_module
    assert callable(optimizer_helper.method_call)
    assert optimizer_helper.method_call.__name__ == expected_method

    if ensure_new:
        assert optimizer_helper.fitter is not current_baseline
        if two_d:
            assert isinstance(optimizer_helper.fitter, Baseline2D)
        else:
            assert isinstance(optimizer_helper.fitter, Baseline)
    else:
        assert optimizer_helper.fitter is current_baseline


def test_optimizer_helper_params():
    """Tests basic method parameter handling for _OptimizerHelper."""

    class Dummy:
        def func(self, data, a, b, c):
            return data, {}

        def func2(self, data, a, b):
            return data, {}

        def func3(self, data, a):
            return data, {}

        def func4(self, data, b):
            return data, {}

        def _spawn_fitter(self, method, ensure_new):
            if ensure_new:
                return self.__class__
            else:
                return self

    fitter = Dummy()

    helper = optimizers._OptimizerHelper('func', fitter)
    assert helper.method == 'func'
    assert callable(helper.method_call)
    assert helper.method_call.__name__ == 'func'
    assert helper.module == 'test_optimizers'
    assert helper.method_param is None
    assert isinstance(helper.method_signature, inspect.Signature)

    method_params = {'func': 'a', 'func2': 'b', None: ('a', 'b')}
    expected = {'func': 'a', 'func2': 'b', 'func3': 'a', 'func4': 'b'}
    for method, expected_param in expected.items():
        helper = optimizers._OptimizerHelper(method, fitter, method_param=method_params)
        assert helper.method == method
        assert callable(helper.method_call)
        assert helper.method_call.__name__ == method
        assert helper.module == 'test_optimizers'
        assert helper.method_param == expected_param
        assert isinstance(helper.method_signature, inspect.Signature)


def test_optimizer_helper_failures():
    """Tests errors raised for _OptimizerHelper."""

    class Dummy:
        def func(self, data, a, b, c):
            return data, {}

        def func2(self, data, a, b):
            return data, {}

        def func3(self, data, a):
            return data, {}

        def func4(self, data, b):
            return data, {}

        def _spawn_fitter(self, method, ensure_new):
            if ensure_new:
                return self.__class__
            else:
                return self

    fitter = Dummy()
    with pytest.raises(
        TypeError,
        match='expected one parameter key for func, but instead got b and a'
    ):
        optimizers._OptimizerHelper('func', fitter, method_param={None: ('b', 'a')})
    with pytest.raises(
        ValueError,
        match=(
            'func2 is not a supported method because it is missing the required '
            'parameter: c or d'
        )
    ):
        optimizers._OptimizerHelper('func2', fitter, method_param={None: ('c', 'd')})
    with pytest.raises(
        ValueError,
        match=(
            'func2 is not a supported method because it is missing the required '
            'parameter: c'
        )
    ):
        optimizers._OptimizerHelper('func2', fitter, method_param={None: 'c'})

    with pytest.raises(
        ValueError,
        match=(
            'func3 is not a supported method because it is missing the required '
            'parameters: c, d'
        )
    ):
        optimizers._OptimizerHelper(
            'func3', fitter, method_param=None, needed_params=['a', 'c', 'd']
        )

    # internal issue, didn't set the default key
    with pytest.raises(KeyError):
        optimizers._OptimizerHelper('func3', fitter, method_param={'func': 'a'})
