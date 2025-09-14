# -*- coding: utf-8 -*-
"""Setup code for testing pybaselines.

@author: Donald Erb
Created on March 20, 2021

"""

import sys

import numpy as np
import pytest

from .base_tests import get_data, get_data2d


pytest_plugins = []
try:
    # ignore if scipy-doctest is unavailable since it's not required for running non-doctest
    # tests; if running doctests without scipy-doctest, the doctests will fail due to expecting no
    # output from matplotlib's plotting routines
    import scipy_doctest  # noqa: F401
    pytest_plugins.append('scipy_doctest')
except ImportError:
    pass


def pytest_addoption(parser):
    """Adds additional pytest command line options."""
    if hasattr(sys, '_is_gil_enabled'):  # sys._is_gil_enabled added in Python 3.13
        gil_enabled = sys._is_gil_enabled()
    else:
        gil_enabled = True

    # potential combos:
    # gil enabled but overriden with input = 1 -> run
    # gil enabled and input is 0 or not set -> skip
    # gil disabled and input is 1 or not set -> run
    # gil disabled but overriden with input = 0 -> skip
    # set default variable so that default is to test if gil is disabled
    parser.addoption(
        "--test_threading",
        action="store",
        default=int(not gil_enabled),
        type=int,
        help='Set to 0 to skip threaded tests for pybaselines or 1 to run.',
    )


def pytest_collection_modifyitems(config, items):
    """Skips tests based on command line inputs."""
    if not config.getvalue('--test_threading'):
        skip_marker = pytest.mark.skip(reason='threaded tests are slow to run')
        for item in items:
            if 'threaded_test' in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def small_data():
    """A small array of data for testing."""
    return np.arange(10, dtype=float)


@pytest.fixture
def small_data2d():
    """A small array of data for testing."""
    return np.arange(60, dtype=float).reshape(6, 10)


@pytest.fixture()
def data_fixture():
    """Test fixture for creating x- and y-data for testing."""
    return get_data()


@pytest.fixture()
def data_fixture2d():
    """Test fixture for creating x-, z-, and y-data for testing."""
    return get_data2d()


@pytest.fixture()
def no_noise_data_fixture():
    """Test fixture that creates x- and y-data without noise for testing."""
    return get_data(include_noise=False)


@pytest.fixture()
def no_noise_data_fixture2d():
    """
    Test fixture that creates x-, z-, and y-data without noise for testing.

    Reduces the number of data points since this is used for testing that numerical
    issues are avoided for large iterations in spline and Whittaker functions, which
    can otherwise be time consuming.
    """
    return get_data2d(include_noise=False, num_points=(20, 31))
