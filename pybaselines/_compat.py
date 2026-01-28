# -*- coding: utf-8 -*-
"""Code to help use optional dependencies and handle changes within dependency versions.

Created on June 24, 2021
@author: Donald Erb

"""

from functools import lru_cache, wraps

import numpy as np
import scipy
from scipy import integrate, sparse


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
# in v2.0.0 -> the deprecation was stopped (delayed?), but rely
# on scipy since there is no potential deprecation there
if hasattr(integrate, 'trapezoid'):
    trapezoid = integrate.trapezoid
else:
    trapezoid = integrate.trapz


@lru_cache(maxsize=None)
def _use_sparse_arrays():
    """
    Checks that the installed SciPy version is new enough to use sparse arrays.

    This check is wrapped into a function just in case it fails so that pybaselines
    can still be imported without error. The result is cached so it only has to
    be done once.

    Returns
    -------
    bool
        True if the installed SciPy version is above 1.12; False otherwise.

    Notes
    -----
    SciPy introduced its sparse arrays in version 1.8, but the interface and helper
    functions were not stable until version 1.12; a warning will be emitted in SciPy
    1.13 when using the matrix interface, so want to use the sparse array interface
    as early as possible.

    """
    try:
        _scipy_version = [int(val) for val in scipy.__version__.lstrip('v').split('.')[:2]]
    except Exception:
        # in case in the far future SciPy stops using semantic versioning; probably
        # bigger problems than this check at that point so just return True
        return True

    return _scipy_version[0] > 1 or (_scipy_version[0] == 1 and _scipy_version[1] >= 12)


@lru_cache(maxsize=None)
def _np_ge_2():
    """
    Checks that the installed NumPy version is version 2.0 or later.

    This check is wrapped into a function just in case it fails so that pybaselines
    can still be imported without error. The result is cached so it only has to
    be done once.

    Returns
    -------
    bool
        True if the installed NumPy version is 2.0 or later; False otherwise.

    """
    try:
        _numpy_version = int(np.__version__.lstrip('v').split('.')[0])
    except Exception:
        # in case in the far future NumPy stops using semantic versioning; probably
        # bigger problems than this check at that point so just return True
        return True

    return _numpy_version >= 2


def dia_object(*args, dtype=None, **kwargs):
    """
    Handles creation of a sparse diagonal object.

    Parameters
    ----------
    *args
        Any arguments to pass to the creation functions.
    dtype : type or numpy.dtype, optional
        The desired data type for the matrix or array.
    **kwargs
        Additional keyword arguments to pass to the creation functions.

    Returns
    -------
    scipy.sparse.dia_matrix or scipy.sparse.dia_array
        A sparse diagonal matrix if the intalled scipy version is older than 1.12,
        otherwise a sparse diagonal array.

    """
    if _use_sparse_arrays():
        return sparse.dia_array(*args, dtype=dtype, **kwargs)
    else:
        return sparse.dia_matrix(*args, dtype=dtype, **kwargs)


def csr_object(*args, dtype=None, **kwargs):
    """
    Handles creation of a sparse csr object.

    Parameters
    ----------
    *args
        Any arguments to pass to the creation functions.
    dtype : type or numpy.dtype, optional
        The desired data type for the matrix or array.
    **kwargs
        Additional keyword arguments to pass to the creation functions.

    Returns
    -------
    scipy.sparse.csr_matrix or scipy.sparse.csr_array
        A sparse csr matrix if the intalled scipy version is older than 1.12,
        otherwise a sparse csr array.

    """
    if _use_sparse_arrays():
        return sparse.csr_array(*args, dtype=dtype, **kwargs)
    else:
        return sparse.csr_matrix(*args, dtype=dtype, **kwargs)


def identity(size, format=None, **kwargs):
    """
    Handles creation of a sparse square identity matrix.

    Parameters
    ----------
    size : int
        The length of the rows and columns of the sparse matrix.
    format : str, optional
        The sparse format to use for the identiy matrix. Default is None, which
        will use the default of the underlying functions.
    **kwargs
        Additional keyword arguments to pass to the creation functions.

    Returns
    -------
    scipy.sparse.spmatrix or scipy.sparse.sparray
        The sparse identity matrix.

    Notes
    -----
    This function will need to be updated in the future to prefer sparse.identity again
    once the sparse matrices are removed.

    """
    if _use_sparse_arrays():
        return sparse.eye_array(size, size, format=format, **kwargs)
    else:
        return sparse.identity(size, format=format, **kwargs)


def diags(data, offsets=0, dtype=None, **kwargs):
    """
    Handles creation of a sparse diagonal matrix.

    Parameters
    ----------
    data : array-like
        The data to be put in the diagonals.
    offsets : int or Sequence[int], optional
        The offsets for `data`. Default is 0, which is the main diagonal.
    dtype : type or numpy.dtype, optional
        The desired data type for the matrix or array.
    **kwargs
        Additional keyword arguments to pass to the creation functions.

    Returns
    -------
    scipy.sparse.spmatrix or scipy.sparse.sparray
        The sparse identiy matrix.

    Notes
    -----
    This function will need to be updated in the future to prefer sparse.diags again
    once the sparse matrices are removed.

    """
    if _use_sparse_arrays():
        return sparse.diags_array(data, offsets=offsets, dtype=dtype, **kwargs)
    else:
        return sparse.diags(data, offsets=offsets, dtype=dtype, **kwargs)


@lru_cache(maxsize=None)
def _allows_1d_slice():
    """
    Determines whether 1d slicing is allowed for SciPy's sparse arrays.

    Returns
    -------
    can_1d_slice : bool
        Whether 1d slicing is allowed for the SciPy version being used.

    Notes
    -----
    An equivalent function would be checking that the SciPy version is at least 1.15.0 or
    that the output of `diags` is a sparse matrix.

    """
    try:
        diags([1]).tocsc()[:, 0]
        can_1d_slice = True
    except NotImplementedError:
        can_1d_slice = False

    return can_1d_slice


def _sparse_col_index(matrix, index):
    """
    Indexes a column within a sparse matrix or array.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix or scipy.sparse.sparray, shape (M, N)
        The sparse object to index.
    index : int
        The column to select.

    Returns
    -------
    output : numpy.ndarray, shape (M,)
        The selected column from the sparse object.

    """
    if not matrix.format == 'csc':
        matrix = matrix.tocsc()

    if sparse.isspmatrix(matrix) or not _allows_1d_slice():
        # for sparse matrices, both matrix[:, [i]] and matrix[:, i] produce the
        # same 2d output
        output = matrix[:, [index]].toarray().ravel()
    else:
        # when allowed, want to use non-fancy indexing since it should be faster
        # for sparse arrays
        output = matrix[:, index].toarray()

    return output
