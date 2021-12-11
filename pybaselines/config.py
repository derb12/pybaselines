# -*- coding: utf-8 -*-
"""Configuration settings for pybaselines.

Created on December 9, 2021
@author: Donald Erb

"""

# Note: the triple quotes are for including the attributes within the documentation
PENTAPY_SOLVER = 2
"""An integer designating the solver to use if pentapy is installed.
pentapy's solver can be used for solving pentadiagonal linear systems, such
as those used for the Whittaker-smoothing-based algorithms. Should be 2 (default)
or 1. See :func:`pentapy.core.solve` for more details.

.. versionchanged:: 1.0.0
    Was previously in the pybaselines.utils module.

"""
