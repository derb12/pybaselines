# -*- coding: utf-8 -*-
"""Code to help use optional dependencies and handle changes within dependency versions.

Created on June 24, 2021
@author: Donald Erb

"""

from functools import wraps

from scipy import integrate


try:
    from pentapy import solve as _pentapy_solve
    _HAS_PENTAPY = True
except ImportError:
    _HAS_PENTAPY = False

    def _pentapy_solve(*args, **kwargs):
        """Dummy function in case pentapy is not installed."""
        raise NotImplementedError('must have pentapy installed to use its solver')

try:
    from numba import jit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def prange(*args):
        """Dummy function that acts exactly like `range` if numba is not installed."""
        return range(*args)

    def jit(func=None, *jit_args, **jit_kwargs):
        """Dummy decorator that does nothing if numba is not installed."""
        # First argument in jit would be the function signature.
        # Signatures can be given as strings: "float64(float64, float64)"
        # or literals or sequences of literals/strings: float64(float64, float64)
        # or (float64,) if no return or [int64(int64, int64), float64(float64, float64)];
        # none of which are callable.
        if func is None or not callable(func):
            # ignore jit_args and jit_kwargs since they are not used by this dummy decorator
            return jit

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper


# scipy.integrate.trapezoid was introduced in v1.6.0, while
# scipy.integrate.trapz will be deprecated in v1.14.0.
# Use scipy instead of numpy since numpy.trapz will be deprecated
# in v2.0.0
if hasattr(integrate, 'trapezoid'):
    trapezoid = integrate.trapezoid
else:
    trapezoid = integrate.trapz
