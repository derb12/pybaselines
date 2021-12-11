# -*- coding: utf-8 -*-
"""Helper functions for working with banded linear systems.

Created on December 8, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded
from scipy.sparse import identity, diags

from . import config
from ._compat import _HAS_PENTAPY, _pentapy_solve


def _pentapy_solver(ab, y, check_output=False):
    """
    Convenience function for calling pentapy's solver with defaults already set.

    Solves the linear system :math:`A @ x = y` for `x`, given the matrix `A` in
    banded format, `ab`. The default settings of :func`:pentapy.solve` are
    already set for the fastest configuration.

    Parameters
    ----------
    ab : array-like
        The matrix `A` in row-wise banded format (see :func:`pentapy.solve`).
    y : array-like
        The right hand side of the equation.

    Returns
    -------
    numpy.ndarray
        The solution to the linear system.

    """
    output = _pentapy_solve(ab, y, is_flat=True, index_row_wise=True, solver=config.PENTAPY_SOLVER)
    if check_output and not np.isfinite(output.dot(output)):
        raise np.linalg.LinAlgError('non-finite value encountered in pentapy solver output')

    return output
