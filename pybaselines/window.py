# -*- coding: utf-8 -*-
"""Smoothing-based techniques for fitting baselines to experimental data.

Created on March 7, 2021
@author: Donald Erb

"""

import warnings
from functools import wraps

from . import smooth


def _wrap_and_warn(func):
    """A temporary wrapper to emit warnings when using functions from pybaselines.windows."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            ('pybaselines.window is deprecated and will be removed in version 1.0; '
             'use pybaselines.smooth instead'), DeprecationWarning, stacklevel=2
        )
        return func(*args, **kwargs)

    return wrapper


@_wrap_and_warn
def noise_median(*args, **kwargs):  # noqa
    return smooth.noise_median(*args, **kwargs)


@_wrap_and_warn
def snip(*args, **kwargs):  # noqa
    return smooth.snip(*args, **kwargs)


@_wrap_and_warn
def swima(*args, **kwargs):  # noqa
    return smooth.swima(*args, **kwargs)


@_wrap_and_warn
def ipsa(*args, **kwargs):  # noqa
    return smooth.ipsa(*args, **kwargs)


@_wrap_and_warn
def ria(*args, **kwargs):  # noqa
    return smooth.ria(*args, **kwargs)
