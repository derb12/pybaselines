# -*- coding: utf-8 -*-
"""Techniques that rely on classifying peak and/or baseline segments for fitting baselines.

Created on January 14, 2024
@author: Donald Erb

"""

from ._algorithm_setup import _Algorithm2D


class _Classification(_Algorithm2D):
    """A base class for all classification algorithms."""
