# -*- coding: utf-8 -*-
"""
=======================================================================================
pybaselines - A library of algorithms for the baseline correction of experimental data.
=======================================================================================

pybaselines provides different techniques for fitting baselines to experimental data.

@author: Donald Erb
Created on March 5, 2021

"""

try:
    from ._version import __version__
except ImportError:
    # in case of local use without installing first
    __version__ = '1.2.1.post1.dev0'

# import utils first since it is imported by other modules; likewise, import
# optimizers and api last since they import the other modules
from . import (
    utils, classification, misc, morphological, polynomial, spline, whittaker, smooth,
    optimizers, api
)

from .api import Baseline
from .two_d.api import Baseline2D
