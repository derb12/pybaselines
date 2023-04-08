# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on April 8, 2023
@author: Donald Erb

"""

from contextlib import contextmanager
from functools import partial, wraps

import numpy as np
from scipy.ndimage import grey_opening

from .._validation import (
    _check_array, _check_half_window, _check_sized_array, _yx_arrays
)
from ..utils import _inverted_sort, pad_edges, relative_difference


class _Algorithm2D:
    """
    A base class for all 2D algorithm types.

    Contains setup methods for all algorithm types to make more complex algorithms
    easier to set up.

    Attributes
    ----------
    poly_order : int
        The last polynomial order used for a polynomial algorithm. Initially is -1, denoting
        that no polynomial fitting has been performed.
    pspline : PSpline or None
        The PSpline object for setting up and solving penalized spline algorithms. Is None
        if no penalized spline setup has been performed (typically done in :meth:`._setup_spline`).
    vandermonde : numpy.ndarray or None
        The Vandermonde matrix for solving polynomial equations. Is None if no polynomial
        setup has been performed (typically done in :meth:`._setup_polynomial`).
    whittaker_system : PenalizedSystem or None
        The PenalizedSystem object for setting up and solving Whittaker-smoothing-based
        algorithms. Is None if no Whittaker setup has been performed (typically done in
        :meth:`_setup_whittaker`).
    x : numpy.ndarray or None
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same length as the input `data` and has min and max
        values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).

    """

    def __init__(self, x_data=None, z_data=None, check_finite=True, output_dtype=None):
        """
        Initializes the algorithm object.

        Parameters
        ----------
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 during the first function call with length equal to the
            input data length.
        z_data : array-like, shape (N,), optional
            The z-values of the measured data. Default is None, which will create an
            array from -1 to 1 during the first function call with length equal to the
            input data length.
        check_finite : bool, optional
            If True (default), will raise an error if any values in input data are not finite.
            Setting to False will skip the check. Note that errors may occur if
            `check_finite` is False and the input data contains non-finite values.
        output_dtype : type or numpy.dtype, optional
            The dtype to cast the output array. Default is None, which uses the typing
            of the input data.

        Notes
        -----
        Unlike `_Algorithm`, `_2DAlgorithm` does not sort input data.

        """
        if x_data is None:
            self.x = None
            self.x_domain = np.array([-1., 1.])
            self._len = None
        else:
            self.x = _check_array(x_data, check_finite=check_finite)
            self._len = len(self.x)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)

        if z_data is None:
            self.z = None
            self.z_domain = np.array([-1., 1.])
            self._len = None
        else:
            self.z = _check_array(z_data, check_finite=check_finite)
            self._len = len(self.z)
            self.z_domain = np.polynomial.polyutils.getdomain(self.z)

        self.whittaker_system = None
        self.vandermonde = None
        self.poly_order = -1
        self.pspline = None
        self._check_finite = check_finite
        self._dtype = output_dtype

    def _return_results(self, baseline, params, dtype, sort_keys=(), axis=-1):
        """
        Re-orders the input baseline and parameters based on the x ordering.

        If `self._sort_order` is None, then no reordering is performed.

        Parameters
        ----------
        baseline : numpy.ndarray, shape (N,)
            The baseline output by the baseline function.
        params : dict
            The parameter dictionary output by the baseline function.
        dtype : [type]
            The desired output dtype for the baseline.
        sort_keys : Iterable, optional
            An iterable of keys corresponding to the values in `params` that need
            re-ordering. Default is ().
        axis : int, optional
            The axis of the input which defines each unique set of data. Default is -1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The input `baseline` after re-ordering and setting to the desired dtype.
        params : dict
            The input `params` after re-ordering the values for `sort_keys`.

        """
        baseline = baseline.astype(dtype, copy=False)

        return baseline, params

    @classmethod
    def _register(cls, func=None, *, dtype=None, order=None, ensure_1d=True, axis=-1):
        """
        Wraps a baseline function to validate inputs and correct outputs.

        The input data is converted to a numpy array, validated to ensure the length is
        consistent, and ordered to match the input x ordering. The outputs are corrected
        to ensure proper inverted sort ordering and dtype.

        Parameters
        ----------
        func : Callable, optional
            The function that is being decorated. Default is None, which returns a partial function.
        dtype : type or numpy.dtype, optional
            The dtype to cast the output array. Default is None, which uses the typing of `array`.
        order : {None, 'C', 'F'}, optional
            The order for the output array. Default is None, which will use the default array
            ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
        ensure_1d : bool, optional
            If True (default), will raise an error if the shape of `array` is not a one dimensional
            array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
        axis : int, optional
            The axis of the input on which to check its length. Default is -1.

        Returns
        -------
        numpy.ndarray
            The calculated baseline.
        dict
            A dictionary of parameters output by the baseline function.

        """
        if func is None:
            return partial(
                cls._register, dtype=dtype, order=order, ensure_1d=ensure_1d, axis=axis
            )

        @wraps(func)
        def inner(self, data=None, *args, **kwargs):
            if self.x is None:
                if data is None:
                    raise TypeError('"data" and "x_data" cannot both be None')
                reset_x = False
                input_y = True
                y, self.x = _yx_arrays(
                    data, check_finite=self._check_finite, dtype=dtype, order=order,
                    ensure_1d=False, axis=axis
                )
                self._len = y.shape[axis]
            else:
                reset_x = True
                if data is not None:
                    input_y = True
                    y = _check_sized_array(
                        data, self._len, check_finite=self._check_finite, dtype=dtype, order=order,
                        ensure_1d=False, axis=axis, name='data'
                    )
                else:
                    y = data
                    input_y = False
                # update self.x just to ensure dtype and order are correct
                x_dtype = self.x.dtype
                self.x = _check_array(
                    self.x, dtype=dtype, order=order, check_finite=False, ensure_1d=False
                )

            if input_y and self._dtype is None:
                output_dtype = y.dtype
            else:
                output_dtype = self._dtype

            baseline, params = func(self, y, *args, **kwargs)
            if reset_x:
                self.x = np.array(self.x, dtype=x_dtype, copy=False)

            return self._return_results(baseline, params, output_dtype, axis)

        return inner

    @contextmanager
    def _override_x(self, new_x, new_sort_order=None):
        """
        Temporarily sets the x-values for the object to a different array.

        Useful when fitting extensions of the x attribute.

        Parameters
        ----------
        new_x : numpy.ndarray
            The x values to temporarily use.
        new_sort_order : [type], optional
            The sort order for the new x values. Default is None, which will not sort.

        Yields
        ------
        pybaselines._algorithm_setup._Algorithm
            The _Algorithm object with the new x attribute.

        """
        old_x = self.x
        old_len = self._len
        old_x_domain = self.x_domain
        old_sort_order = self._sort_order
        old_inverted_order = self._inverted_order
        # also have to reset any sized attributes to force recalculation for new x
        old_poly_order = self.poly_order
        old_vandermonde = self.vandermonde
        old_whittaker_system = self.whittaker_system
        old_pspline = self.pspline

        try:
            self.x = _check_array(new_x, check_finite=self._check_finite)
            self._len = len(self.x)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)
            self._sort_order = new_sort_order
            if self._sort_order is not None:
                self._inverted_order = _inverted_sort(self._sort_order)
            else:
                self._inverted_order = None

            self.vandermonde = None
            self.poly_order = -1
            self.whittaker_system = None
            self.pspline = None

            yield self

        finally:
            self.x = old_x
            self._len = old_len
            self.x_domain = old_x_domain
            self._sort_order = old_sort_order
            self._inverted_order = old_inverted_order
            self.vandermonde = old_vandermonde
            self.poly_order = old_poly_order
            self.whittaker_system = old_whittaker_system
            self.pspline = old_pspline

    def _setup_morphology(self, y, half_window=None, **window_kwargs):
        """
        Sets the starting parameters for morphology-based methods.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using pybaselines.morphological.optimize_window.
        **window_kwargs
            Keyword arguments to pass to :func:`.optimize_window`.
            Possible items are:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 3.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable half-window size. If None (default), will be
                    set to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        output_half_window : int
            The accepted half window size.

        Notes
        -----
        Ensures that window size is odd since morphological operations operate in
        the range [-output_half_window, ..., output_half_window].

        Half windows are dealt with rather than full window sizes to clarify their
        usage. SciPy morphology operations deal with full window sizes.

        """
        if half_window is not None:
            output_half_window = _check_half_window(half_window)
        else:
            output_half_window = _optimize_window(y, **window_kwargs)

        return y, output_half_window

    def _setup_smooth(self, y, half_window=0, allow_zero=True, **pad_kwargs):
        """
        Sets the starting parameters for doing smoothing-based algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
        half_window : int, optional
            The half-window used for the smoothing functions. Used
            to pad the left and right edges of the data to reduce edge
            effects. Default is 0, which provides no padding.
        allow_zero : bool, optional
            If True (default), allows `half_window` to be 0; otherwise, `half_window`
            must be at least 1.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing.

        Returns
        -------
        numpy.ndarray, shape (``N + 2 * half_window``,)
            The padded array of data.

        """
        hw = _check_half_window(half_window, allow_zero)
        return pad_edges(y, hw, **pad_kwargs)

    def _setup_misc(self, y):
        """
        Sets the starting parameters for doing miscellaneous algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.

        Notes
        -----
        Since the miscellaneous functions are not related, the only use of this
        function is for aliasing the input `data` to `y`.

        """
        return y


# TODO maybe just make a way to merge the 1D and 2D versions
def _optimize_window(data, increment=1, max_hits=3, window_tol=1e-6,
                     max_half_window=None, min_half_window=None):
    """
    Optimizes the morphological half-window size.

    Parameters
    ----------
    data : array-like, shape (N,)
        The measured data values.
    increment : int, optional
        The step size for iterating half windows. Default is 1.
    max_hits : int, optional
        The number of consecutive half windows that must produce the same
        morphological opening before accepting the half window as the optimum
        value. Default is 3.
    window_tol : float, optional
        The tolerance value for considering two morphological openings as
        equivalent. Default is 1e-6.
    max_half_window : int, optional
        The maximum allowable half-window size. If None (default), will be set
        to (len(data) - 1) / 2.
    min_half_window : int, optional
        The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    half_window : int
        The optimized half window size.

    Notes
    -----
    May only provide good results for some morphological algorithms, so use with
    caution.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

    """
    y = np.asarray(data)
    if max_half_window is None:
        max_half_window = (y.shape[0] - 1) // 2
    if min_half_window is None:
        min_half_window = 1

    # TODO would it be better to allow padding the data?
    opening = grey_opening(y, [2 * min_half_window + 1, 2 * min_half_window + 1])
    hits = 0
    best_half_window = min_half_window
    for half_window in range(min_half_window + increment, max_half_window, increment):
        new_opening = grey_opening(y, [half_window * 2 + 1, half_window * 2 + 1])
        if relative_difference(opening, new_opening) < window_tol:
            if hits == 0:
                # keep just the first window that fits tolerance
                best_half_window = half_window - increment
            hits += 1
            if hits >= max_hits:
                half_window = best_half_window
                break
        elif hits:
            hits = 0
        opening = new_opening

    return max(half_window, 1)  # ensure half window is at least 1
