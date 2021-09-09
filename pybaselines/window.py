# -*- coding: utf-8 -*-
"""Smoothing-based techniques for fitting baselines to experimental data.

Created on March 7, 2021
@author: Donald Erb

"""

# import warnings

from . import smooth


def __getattr__(func):
    # TODO: in a later version, begin emitting the warning
    # warnings.warn(
    #    'pybaselines.window is deprecated and will be removed in version 1.0; '
    #    'use pybaselines.smooth instead', DeprecationWarning, stacklevel=2
    # )
    return getattr(smooth, func)
