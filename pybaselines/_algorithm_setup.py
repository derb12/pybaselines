# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on March 31, 2021
@author: Donald Erb

TODO: non-finite values (nan or inf) could be replaced for algorithms that use weighting
by setting their values to arbitrary value (eg. 0) within the output y, set their weights
to 0, and then back-fill after the calculation; something to consider, rather than just
raising an exception when encountering a non-finite value; could also interpolate rather
than just filling back in the nan or inf value. Could accomplish by setting check_finite
to something like 'mask'.

"""

from functools import partial, wraps
import warnings

import numpy as np
from scipy.linalg import solveh_banded

from ._banded_utils import _pentapy_solver, PenalizedSystem, diff_penalty_diagonals
from ._compat import _HAS_PENTAPY
from ._spline_utils import _spline_basis, _spline_knots, PSpline
from ._validation import (
    _check_array, _check_half_window, _check_lam, _check_optional_array, _check_sized_array,
    _yx_arrays
)
from .utils import ParameterWarning, _inverted_sort, optimize_window, pad_edges


class _Algorithm:
    """
    A base class for all algorithm types.

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
    x : numpy.ndarray
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same length as the input `data` and has min and max
        values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x` is None during initialization, then
        set to numpy.ndarray([-1, 1]).

    """

    def __init__(self, x_data=None, check_finite=True, assume_sorted=False,
                 output_dtype=None):
        """
        Initializes the algorithm object.

        Parameters
        ----------
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 during the first function call with length equal to the
            input data length.
        check_finite : bool, optional
            If True, will raise an error if any values if `array` are not finite.
            Default is False, which skips the check. Note that errors may occur if
            `check_finite` is False and the input data contains non-finite values.
        assume_sorted : bool, optional
            If False (default), will sort the input `x_data` values. Otherwise, the
            input is assumed to be sorted. Note that some functions may raise an error
            if `x_data` is not sorted.
        output_dtype : type or np.dtype, optional
            The dtype to cast the output array. Default is None, which uses the typing
            of the input data.

        """
        if x_data is None:
            self.x = None
            self.x_domain = np.array([-1., 1.])
            self._len = None
        else:
            self.x = _check_array(x_data, check_finite=check_finite)
            self._len = len(self.x)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)

        if x_data is None or assume_sorted:
            self._sort_order = None
            self._inverted_order = None
        else:
            self._sort_order = self.x.argsort(kind='mergesort')
            self.x = self.x[self._sort_order]
            self._inverted_order = _inverted_sort(self._sort_order)

        self.whittaker_system = None
        self.vandermonde = None
        self.poly_order = -1
        self.pspline = None
        self._check_finite = check_finite
        self._dtype = output_dtype

    def _return_results(self, baseline, params, dtype, sort_keys=()):
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

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The input `baseline` after re-ordering and setting to the desired dtype.
        params : dict
            The input `params` after re-ordering the values for `sort_keys`.

        """
        if self._sort_order is not None:
            for key in sort_keys:
                if key in params:  # some parameters are conditionally output
                    params[key] = params[key][self._inverted_order]
            baseline = baseline[self._inverted_order]

        baseline = baseline.astype(dtype, copy=False)

        return baseline, params

    @classmethod
    def _register(cls, func=None, *, sort_keys=(), dtype=None, order=None, ensure_1d=True, axis=-1):
        """
        Wraps a baseline function to validate inputs and correct outputs.

        The input data is converted to a numpy array, validated to ensure the length is
        consistent, and ordered to match the input x ordering. The outputs are corrected
        to ensure proper inverted sort ordering and dtype.

        Parameters
        ----------
        func : Callable, optional
            The function that is being decorated. Default is None, which returns a partial function.
        sort_keys : tuple, optional
            The keys within the output parameter dictionary that will need sorting to match the
            sort order of :attr:`.x`. Default is ().
        dtype : type or np.dtype, optional
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
                cls._register, sort_keys=sort_keys, dtype=dtype, order=order,
                ensure_1d=ensure_1d, axis=axis
            )

        @wraps(func)
        def inner(self, data, *args, **kwargs):
            if self.x is None:
                if data is None:
                    raise ValueError('"data" and "x_data" cannot both be None')
                reset_x = False
                y, self.x = _yx_arrays(
                    data, check_finite=self._check_finite, dtype=dtype, order=order,
                    ensure_1d=ensure_1d, axis=axis
                )
                self._len = y.shape[axis]
            else:
                reset_x = True
                y = _check_sized_array(
                    data, self._len, check_finite=self._check_finite, dtype=dtype, order=order,
                    ensure_1d=ensure_1d, axis=axis, name='data'
                )
                # update self.x just to ensure dtype and order are correct
                x_dtype = self.x.dtype
                self.x = _check_array(
                    self.x, dtype=dtype, order=order, check_finite=False, ensure_1d=False
                )
            if self._sort_order is not None:
                y = y[self._sort_order]
            if self._dtype is None:
                output_dtype = y.dtype
            else:
                output_dtype = self._dtype

            baseline, params = func(self, y, *args, **kwargs)
            if reset_x:
                self.x = np.array(self.x, dtype=x_dtype, copy=False)

            return self._return_results(baseline, params, output_dtype, sort_keys)

        return inner

    def _setup_whittaker(self, y, lam, diff_order=2, weights=None, copy_weights=False,
                         allow_lower=True, reverse_diags=None):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
        lam : float
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on the penalized least square method
            and the differential order.
        diff_order : int, optional
            The integer differential order; must be greater than 0. Default is 2.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            shape (N,) and all values set to 1.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        allow_lower : boolean, optional
            If True (default), will allow using only the lower non-zero diagonals of
            the squared difference matrix. If False, will include all non-zero diagonals.
        reverse_diags : {None, False, True}, optional
            If True, will reverse the order of the diagonals of the squared difference
            matrix. If False, will never reverse the diagonals. If None (default), will
            only reverse the diagonals if using pentapy's solver.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,), optional
            The weighting array.

        Raises
        ------
        ValueError
            Raised is `diff_order` is less than 1.

        Warns
        -----
        ParameterWarning
            Raised if `diff_order` is greater than 3.

        """
        if diff_order < 1:
            raise ValueError(
                'the difference order must be > 0 for Whittaker-smoothing-based methods'
            )
        elif diff_order > 3:
            warnings.warn(
                ('difference orders greater than 3 can have numerical issues;'
                 ' consider using a difference order of 2 or 1 instead'),
                ParameterWarning, stacklevel=2
            )
        weight_array = _check_optional_array(
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite
        )
        if self.whittaker_system is not None:
            self.whittaker_system.reset_diagonals(lam, diff_order, allow_lower, reverse_diags)
        else:
            self.whittaker_system = PenalizedSystem(
                self._len, lam, diff_order, allow_lower, reverse_diags
            )

        return y, weight_array

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=False,
                          calc_pinv=False, copy_weights=False):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 with N points.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        poly_order : int, optional
            The polynomial order. Default is 2.
        calc_vander : bool, optional
            If True, will calculate and the Vandermonde matrix. Default is False.
        calc_pinv : bool, optional
            If True, and if `return_vander` is True, will calculate and return the
            pseudo-inverse of the Vandermonde matrix. Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for fitting a polynomial to the data.
        pseudo_inverse : numpy.ndarray
            Only returned if `calc_pinv` is True. The pseudo-inverse of the
            Vandermonde matrix, calculated with singular value decomposition (SVD).

        Raises
        ------
        ValueError
            Raised if `calc_pinv` is True and `calc_vander` is False.

        Notes
        -----
        If x_data is given, its domain is reduced from ``[min(x_data), max(x_data)]``
        to [-1., 1.] to improve the numerical stability of calculations; since the
        Vandermonde matrix goes from ``x**0`` to ``x^**poly_order``, large values of
        x would otherwise cause difficulty when doing least squares minimization.

        """
        weight_array = _check_optional_array(
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite
        )
        if calc_vander:
            if self.vandermonde is None or poly_order > self.poly_order:
                mapped_x = np.polynomial.polyutils.mapdomain(
                    self.x, self.x_domain, np.array([-1., 1.])
                )
                self.vandermonde = np.polynomial.polynomial.polyvander(mapped_x, poly_order)
            elif poly_order < self.poly_order:
                self.vandermonde = self.vandermonde[:, :poly_order + 1]
        self.poly_order = poly_order

        if not calc_pinv:
            return y, weight_array
        elif not calc_vander:
            raise ValueError('if calc_pinv is True, then calc_vander must also be True')

        if weights is None:
            pseudo_inverse = np.linalg.pinv(self.vandermonde)
        else:
            pseudo_inverse = np.linalg.pinv(np.sqrt(weight_array)[:, None] * self.vandermonde)

        return y, weight_array, pseudo_inverse

    def _setup_splines(self, y, weights=None, spline_degree=3, num_knots=10,
                       penalized=True, diff_order=3, lam=1, make_basis=True, allow_lower=True,
                       reverse_diags=None, copy_weights=False):
        """
        Sets the starting parameters for doing spline fitting.

        Parameters
        ----------
        y : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        x_data : array-like, shape (N,), optional
            The x-values of the measured data. Default is None, which will create an
            array from -1 to 1 with N points.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        num_knots : int, optional
            The number of interior knots for the splines. Default is 10.
        penalized : bool, optional
            Whether the basis matrix should be for a penalized spline or a regular
            B-spline. Default is True, which creates the basis for a penalized spline.
        diff_order : int, optional
            The integer differential order for the spline penalty; must be greater than 0.
            Default is 3. Only used if `penalized` is True.
        lam : float, optional
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on the number of knots and the difference order.
            Default is 1.
        make_basis : bool, optional
            If True (default), will create the matrix containing the spline basis functions.
        allow_lower : boolean, optional
            If True (default), will include only the lower non-zero diagonals of
            the squared difference matrix. If False, will include all non-zero diagonals.
        reverse_diags : boolean, optional
            If True, will reverse the order of the diagonals of the penalty matrix.
            Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for fitting the spline to the data.

        Warns
        -----
        ParameterWarning
            Raised if `diff_order` is greater than 4.

        Notes
        -----
        `degree` is used instead of `order` like for polynomials since the order of a spline
        is defined by convention as ``degree + 1``.

        """
        weight_array = _check_optional_array(
            self._len, weights, dtype=float, order='C', copy_input=copy_weights,
            check_finite=self._check_finite
        )

        if make_basis:
            if diff_order > 4:
                warnings.warn(
                    ('differential orders greater than 4 can have numerical issues;'
                     ' consider using a differential order of 2 or 3 instead'),
                    ParameterWarning, stacklevel=2
                )

            if self.pspline is None or not self.pspline.same_basis(num_knots, spline_degree):
                self.pspline = PSpline(
                    self.x, num_knots, spline_degree, self._check_finite, lam, diff_order,
                    allow_lower, reverse_diags
                )
            else:
                self.pspline.reset_penalty_diagonals(
                    lam, diff_order, allow_lower, reverse_diags
                )

        return y, weight_array


def _setup_whittaker(data, lam, diff_order=2, weights=None, copy_weights=False,
                     lower_only=True, reverse_diags=False):
    """
    Sets the starting parameters for doing penalized least squares.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float
        The smoothing parameter, lambda. Typical values are between 10 and
        1e8, but it strongly depends on the penalized least square method
        and the differential order.
    diff_order : int, optional
        The integer differential order; must be greater than 0. Default is 2.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        shape (N,) and all values set to 1.
    copy_weights : boolean, optional
        If True, will copy the array of input weights. Only needed if the
        algorithm changes the weights in-place. Default is False.
    lower_only : boolean, optional
        If True (default), will include only the lower non-zero diagonals of
        the squared difference matrix. If False, will include all non-zero diagonals.
    reverse_diags : boolean, optional
        If True, will reverse the order of the diagonals of the squared difference
        matrix. Default is False.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    numpy.ndarray
        The array containing the diagonal data of the product of `lam` and the
        squared finite-difference matrix of order `diff_order`. Has a shape of
        (`diff_order` + 1, N) if `lower_only` is True, otherwise
        (``diff_order * 2 + 1``, N).
    weight_array : numpy.ndarray, shape (N,), optional
        The weighting array.

    Raises
    ------
    ValueError
        Raised is `diff_order` is less than 1 or if `weights` and `data` do not
        have the same shape.

    Warns
    -----
    ParameterWarning
        Raised if `diff_order` is greater than 3.

    """
    y = _check_array(data, check_finite=True)
    if diff_order < 1:
        raise ValueError(
            'the differential order must be > 0 for Whittaker-smoothing-based methods'
        )
    elif diff_order > 3:
        warnings.warn(
            ('differential orders greater than 3 can have numerical issues;'
             ' consider using a differential order of 2 or 1 instead'),
            ParameterWarning
        )
    num_y = len(y)
    if weights is None:
        weight_array = np.ones(num_y)
    else:
        weight_array = _check_sized_array(
            weights, num_y, check_finite=True, ensure_1d=True
        )
        if copy_weights:
            weight_array = weight_array.copy()

    diagonal_data = diff_penalty_diagonals(num_y, diff_order, lower_only)
    if reverse_diags:
        diagonal_data = diagonal_data[::-1]

    return y, _check_lam(lam) * diagonal_data, weight_array


def _get_vander(x, poly_order=2, weights=None, calc_pinv=True):
    """
    Calculates the Vandermonde matrix and its pseudo-inverse.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The x-values for the polynomial with N data points.
    poly_order : int, optional
        The polynomial order. Default is 2.
    weights : np.ndarray, shape (N,), optional
        The weighting array. If None (default), will ignore. Otherwise,
        will multiply the Vandermonde by the weighting array before calculating
        the pseudo-inverse.
    calc_pinv : bool, optional
        If True (default), will calculate and return the pseudo-inverse of the
        Vandermonde, after applying weights.

    Returns
    -------
    vander : numpy.ndarray, shape (N, poly_order + 1)
        The Vandermonde matrix for the polynomial.
    pseudo_inverse : numpy.ndarray, shape (poly_order + 1, N)
        The pseudo-inverse of the Vandermonde, with weights applied if input.
        Calculated using singular value decomposition (SVD).

    Notes
    -----
    If weights are supplied, they should be the square-root of the total weights.

    """
    vander = np.polynomial.polynomial.polyvander(x, poly_order)
    if not calc_pinv:
        return vander

    if weights is not None:
        pseudo_inverse = np.linalg.pinv(weights[:, np.newaxis] * vander)
    else:
        pseudo_inverse = np.linalg.pinv(vander)

    return vander, pseudo_inverse


def _setup_polynomial(data, x_data=None, weights=None, poly_order=2, return_vander=False,
                      return_pinv=False, copy_weights=False):
    """
    Sets the starting parameters for doing polynomial fitting.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    poly_order : int, optional
        The polynomial order. Default is 2.
    return_vander : bool, optional
        If True, will calculate and return the Vandermonde matrix. Default is False.
    return_pinv : bool, optional
        If True, and if `return_vander` is True, will calculate and return the
        pseudo-inverse of the Vandermonde matrix. Default is False.
    copy_weights : boolean, optional
        If True, will copy the array of input weights. Only needed if the
        algorithm changes the weights in-place. Default is False.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    x : numpy.ndarray, shape (N,)
        The x-values for fitting the polynomial, converted to fit within
        the domain [-1., 1.].
    weight_array : numpy.ndarray, shape (N,)
        The weight array for fitting a polynomial to the data.
    original_domain : numpy.ndarray, shape (2,)
        The minimum and maximum values of the original x_data values. Can
        be used to convert the coefficents found during least squares
        minimization using the normalized x into usable polynomial coefficients
        for the original x_data.
    vander : numpy.ndarray
        Only returned if return_vander is True. The Vandermonde matrix for the
        normalized x values.
    pseudo_inverse : numpy.ndarray
        Only returned if return_pinv is True. The pseudo-inverse of the
        Vandermonde matrix, calculated with singular value decomposition (SVD).

    Notes
    -----
    If x_data is given, its domain is reduced from [min(x_data), max(x_data)]
    to [-1., 1.] to improve the numerical stability of calculations; since the
    Vandermonde matrix goes from x^0 to x^poly_order, large values of x would
    otherwise cause difficulty when doing least squares minimization.

    """
    y, x = _yx_arrays(data, x_data, check_finite=True)
    len_y = len(y)
    if weights is None:
        weight_array = np.ones(len_y)
    else:
        weight_array = _check_sized_array(
            weights, len_y, check_finite=True, ensure_1d=True
        )
        if copy_weights:
            weight_array = weight_array.copy()
    if x_data is None:
        original_domain = np.array([-1., 1.])
    else:
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1., 1.]))

    output = [y, x, weight_array, original_domain]
    if return_vander:
        vander_output = _get_vander(x, poly_order, np.sqrt(weight_array), return_pinv)
        if return_pinv:
            output.extend(vander_output)
        else:
            output.append(vander_output)

    return output


def _setup_morphology(data, half_window=None, **window_kwargs):
    """
    Sets the starting parameters for morphology-based methods.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
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
    y = _check_array(data, ensure_1d=True)
    if half_window is not None:
        output_half_window = _check_half_window(half_window)
    else:
        output_half_window = optimize_window(y, **window_kwargs)

    return y, output_half_window


def _setup_smooth(data, half_window=0, allow_zero=True, **pad_kwargs):
    """
    Sets the starting parameters for doing smoothing-based algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
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
    numpy.ndarray, shape (N + 2 * half_window)
        The padded array of data.

    """
    y = _check_array(data, ensure_1d=True)
    hw = _check_half_window(half_window, allow_zero)
    return pad_edges(y, hw, **pad_kwargs)


def _setup_classification(data, x_data=None, weights=None):
    """
    Sets the starting parameters for doing classification algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    x : numpy.ndarray, shape (N,)
        The x-values for fitting the polynomial, converted to fit within
        the domain [-1., 1.].
    weight_array : numpy.ndarray, shape (N,)
        The weight array for the data, with boolean dtype.
    original_domain : numpy.ndarray, shape (2,)
        The minimum and maximum values of the original x_data values. Can
        be used to convert the coefficents found during least squares
        minimization using the normalized x into usable polynomial coefficients
        for the original x_data.

    """
    y, x = _yx_arrays(data, x_data, check_finite=True)
    len_y = len(y)
    if weights is None:
        weight_array = np.ones(len_y)
    else:
        weight_array = _check_sized_array(
            weights, len_y, dtype=bool, check_finite=True, ensure_1d=True
        )
    # TODO should remove the x-scaling here since most methods don't need it; can
    # make a separate function for it, which _setup_polynomial could also use
    if x_data is None:
        original_domain = np.array([-1., 1.])
    else:
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1., 1.]))

    return y, x, weight_array, original_domain


def _setup_splines(data, x_data=None, weights=None, spline_degree=3, num_knots=10,
                   penalized=True, diff_order=3, lam=1, make_basis=True, lower_only=True,
                   reverse_diags=False):
    """
    Sets the starting parameters for doing spline fitting.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    num_knots : int, optional
        The number of interior knots for the splines. Default is 10.
    penalized : bool, optional
        Whether the basis matrix should be for a penalized spline or a regular
        B-spline. Default is True, which creates the basis for a penalized spline.
    diff_order : int, optional
        The integer differential order for the spline penalty; must be greater than 0.
        Default is 3. Only used if `penalized` is True.
    lam : float, optional
        The smoothing parameter, lambda. Typical values are between 10 and
        1e8, but it strongly depends on the number of knots and the difference order.
        Default is 1.
    make_basis : bool, optional
        If True (default), will create the matrix containing the spline basis functions.
    lower_only : boolean, optional
        If True (default), will include only the lower non-zero diagonals of
        the squared difference matrix. If False, will include all non-zero diagonals.
    reverse_diags : boolean, optional
        If True, will reverse the order of the diagonals of the penalty matrix.
        Default is False.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    x : numpy.ndarray, shape (N,)
        The x-values for fitting the spline.
    weight_array : numpy.ndarray, shape (N,)
        The weight array for fitting the spline to the data.
    basis : scipy.sparse.csr.csr_matrix
        The spline basis matrix. Only returned if `make_basis` is True.
    knots : numpy.ndarray, shape (``num_knots + 2 * spline_degree``,)
        The array of knots for the spline, properly padded on each side. Only
        return if `make_basis` if True.
    penalty_diagonals : numpy.ndarray, shape (`diff_order` + 1 or ``diff_order * 2 + 1``, N)
        The finite difference penalty matrix, in LAPACK's banded format (see
        :func:`scipy.linalg.solveh_banded` and :func:`scipy.linalg.solve_banded`).
        Only returned if both `penalized` and `make_basis` are True. Has a shape of
        (`diff_order` + 1, N) if `lower_only` is True, otherwise
        (``diff_order * 2 + 1``, N).

    Raises
    ------
    ValueError
        Raised if `diff_order` is less than 1, if `weights` and `data` do not have
        the same shape, if the number of spline basis functions
        (`num_knots` + `spline_degree` - 1) is <= `diff_order`, or if `spline_degree`
        is less than 0.

    Warns
    -----
    ParameterWarning
        Raised if `diff_order` is greater than 4.

    Notes
    -----
    `degree` is used instead of `order` like for polynomials since the order of a spline
    is defined by convention as `degree` + 1.

    """
    y, x = _yx_arrays(data, x_data, check_finite=True, dtype=float, order='C')
    len_y = len(y)
    if weights is None:
        weight_array = np.ones(len_y)
    else:
        weight_array = _check_sized_array(
            weights, len_y, check_finite=True, dtype=float, order='C', ensure_1d=True
        )
    if not make_basis:
        return y, x, weight_array

    if spline_degree < 0:
        raise ValueError('spline degree must be >= 0')

    knots = _spline_knots(x, num_knots, spline_degree, penalized)
    basis = _spline_basis(x, knots, spline_degree)
    if not penalized:
        return y, x, weight_array, basis, knots

    num_bases = basis.shape[1]  # number of basis functions
    if diff_order < 1:
        raise ValueError(
            'the difference order must be > 0 for spline methods'
        )
    elif diff_order >= num_bases:
        raise ValueError((
            'the difference order must be less than the number of basis '
            'functions, which is the number of knots + spline degree - 1'
        ))
    elif diff_order > 4:
        warnings.warn(
            ('differential orders greater than 4 can have numerical issues;'
             ' consider using a differential order of 2 or 3 instead'),
            ParameterWarning
        )

    penalty_diagonals = _check_lam(lam) * diff_penalty_diagonals(
        num_bases, diff_order, lower_only, padding=spline_degree - diff_order
    )
    if reverse_diags:
        penalty_diagonals = penalty_diagonals[::-1]

    return y, x, weight_array, basis, knots, penalty_diagonals


def _whittaker_smooth(data, lam=1e6, diff_order=2, weights=None):
    """
    Performs Whittaker smoothing on the input data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points. Must not
        contain missing data (NaN) or Inf.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother fits.
        Default is 1e6.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be an array
        with size equal to N and all values set to 1.

    Returns
    -------
    smooth_y : numpy.ndarray, shape (N,)
        The smoothed data.
    weight_array : numpy.ndarray, shape (N,)
        The weights used for fitting the data.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    using_pentapy = _HAS_PENTAPY and diff_order == 2
    y, diagonals, weight_array = _setup_whittaker(
        data, lam, diff_order, weights, False, not using_pentapy, using_pentapy
    )
    main_diag_idx = diff_order if using_pentapy else 0
    diagonals[main_diag_idx] = diagonals[main_diag_idx] + weight_array
    if using_pentapy:
        smooth_y = _pentapy_solver(diagonals, weight_array * y)
    else:
        smooth_y = solveh_banded(
            diagonals, weight_array * y, overwrite_ab=True, overwrite_b=True, check_finite=False,
            lower=True
        )

    return smooth_y, weight_array


def _get_function(method, modules):
    """
    Tries to retrieve the indicated function from a list of modules.

    Parameters
    ----------
    method : str
        The string name of the desired function. Case does not matter.
    modules : Sequence
        A sequence of modules in which to look for the method.

    Returns
    -------
    func : Callable
        The corresponding function.
    func_module : str
        The module that `func` belongs to.

    Raises
    ------
    AttributeError
        Raised if no matching function is found within the modules.

    """
    function_string = method.lower()
    for module in modules:
        if hasattr(module, function_string):
            func = getattr(module, function_string)
            func_module = module.__name__.split('.')[-1]
            break
    else:  # in case no break
        raise AttributeError(f'unknown method {method}')

    return func, func_module


def _setup_optimizer(data, method, modules, method_kwargs=None, copy_kwargs=True,
                     ensure_1d=True, **kwargs):
    """
    Sets the starting parameters for doing optimizer algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    method : str
        The string name of the desired function, like 'asls'. Case does not matter.
    modules : Sequence(module, ...)
        The modules to search for the indicated `method` function.
    method_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the fitting function. Default
        is None, which uses an emtpy dictionary.
    copy_kwargs : bool, optional
        If True (default), will copy the input `method_kwargs` so that the input
        dictionary is not modified within the function.
    ensure_1d : bool, optional
        If True (default), will ensure the input `data` is one dimensional.
    **kwargs
        Deprecated in version 0.8.0 and will be removed in version 0.10 or 1.0. Pass any
        keyword arguments for the fitting function in the `method_kwargs` dictionary.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    fit_func : Callable
        The function for fitting the baseline.
    func_module : str
        The string name of the module that contained `fit_func`.
    method_kws : dict
        A dictionary of keyword arguments to pass to `fit_func`.

    Warns
    -----
    DeprecationWarning
        Passed if `kwargs` is not empty.

    """
    y = _check_array(data, ensure_1d=ensure_1d)
    fit_func, func_module = _get_function(method, modules)
    if method_kwargs is None:
        method_kws = {}
    elif copy_kwargs:
        method_kws = method_kwargs.copy()
    else:
        method_kws = method_kwargs

    if kwargs:  # TODO remove in version 0.10 or 1.0
        warnings.warn(
            ('Passing additional keyword arguments directly to optimizer functions is '
             'deprecated and will be removed in version 0.10.0 or version 1.0. Place all '
             'keyword arguments into the method_kwargs dictionary instead.'),
            DeprecationWarning, stacklevel=2
        )
        method_kws.update(kwargs)

    return y, fit_func, func_module, method_kws
