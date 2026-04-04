# -*- coding: utf-8 -*-
"""Setup code for the ND Mixin classes.

Created on March 23, 2026
@author: Donald Erb

"""

from functools import partial, wraps


def _handle_io(func=None, *, sort_keys=(), ensure_dims=True, reshape_keys=(),
               skip_sorting=False, require_unique=False):
    """
    Wraps a baseline method to validate inputs and correct outputs.

    Allows passing through keywords to the underlying fitting object's `_handle_io` method.

    Parameters
    ----------
    func : callable, optional
        The method that is being decorated. Default is None, which returns a partial function.
    sort_keys : tuple, optional
        The keys within the output parameter dictionary that will need sorting to match the
        sort order of the object's `x` and potentially `z` attributes. Default is ().
    ensure_dims : bool, optional
        If True (default), will raise an error if the shape of `array` is not a one
        dimensional array with shape (N,) or a two dimensional array with shape (N, 1) or
        (1, N) if `self` is an `_Algorithm`, or if the shape of `array`
        is not a two dimensional array with shape (M, N) or a three dimensional array with
        shape (M, N, 1), (M, 1, N), or (1, M, N) if `self` is an `_Algorithm2D`.
    reshape_keys : tuple, optional
        If `self` is `_Algorithm2D`, the keys within the output parameter dictionary that
        will need reshaped to match the shape of the data. Ignored for `_Algorithm` `self`.
        Default is ().
    skip_sorting : bool, optional
        If True, will skip sorting the output baseline. The keys in `sort_keys` will
        still be sorted. Default is False.
    require_unique : bool, optional
        If True, will check ``self.x`` and potentially ``self.z`` to ensure all values are
        unique and will raise an error if non-unique values are present. Default is False,
        which skips the check.

    Returns
    -------
    callable
        The wrapped method.

    Notes
    -----
    Within the inner function, `self` can be either `pybaselines._algorithm_setup._Algorithm`
    or `pybaselines.two_d._algorithm_setup._Algorithm2D`.

    """
    if func is None:
        return partial(
            _handle_io, sort_keys=sort_keys, ensure_dims=ensure_dims, reshape_keys=reshape_keys,
            skip_sorting=skip_sorting, require_unique=require_unique
        )

    @wraps(func)
    def inner(self, *args, **kwargs):
        return self._handle_io(
            func, sort_keys=sort_keys, ensure_dims=ensure_dims, reshape_keys=reshape_keys,
            skip_sorting=skip_sorting, require_unique=require_unique
        )(self, *args, **kwargs)
    return inner
