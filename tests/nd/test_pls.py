# -*- coding: utf-8 -*-
"""Tests for pybaselines._nd._pls.

@author: Donald Erb
Created on March 30, 2026

"""

import inspect

import pytest

from pybaselines._nd import _pls


def get_module_methods(klass):
    """Gets all methods of a class defined in the same module as the class."""
    methods = []
    class_module = inspect.getmodule(klass)
    for (method_name, method) in inspect.getmembers(klass):
        if (
            inspect.isfunction(method)
            and inspect.getmodule(method) is class_module
        ):
            methods.append(method_name)

    return methods


@pytest.mark.parametrize('method', get_module_methods(_pls._PLSNDMixin))
def test_spline_degree_none(method):
    """Ensures the default `spline_degree` is None for all PLS methods to ensure logic flow.

    Penalized least squares methods should have default `spline_degree=None` to
    do Whittaker smoothing as the default behavior.

    """
    params = inspect.signature(getattr(_pls._PLSNDMixin, method)).parameters
    assert params['spline_degree'].default is None
