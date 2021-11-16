# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on March 31, 2021
@author: Donald Erb

TODO: non-finite values (nan or inf) could be replaced for algorithms that use weighting
by setting their values to arbitrary value (eg. 0) within the output y, set their weights
to 0, and then back-fill after the calculation; something to consider, rather than just
raising an exception when encountering a non-finite value; could also interpolate rather
than just filling back in the nan or inf value.

"""

import operator
import warnings

import numpy as np
from scipy.linalg import solveh_banded

from ._compat import _HAS_PENTAPY
from ._spline_utils import _spline_basis, _spline_knots
from .utils import (
    _check_scalar, _pentapy_solver, ParameterWarning, difference_matrix, optimize_window, pad_edges
)


def _yx_arrays(data, x_data=None, x_min=-1., x_max=1., check_finite=False):
    """
    Converts input data into numpy arrays and provides x data if none is given.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1. to 1. with N points.
    x_min : float, optional
        The minimum x-value if `x_data` is None. Default is -1.
    x_max : float, optional
        The maximum x-value if `x_data` is None. Default is 1.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        A numpy array of the y-values of the measured data.
    x : numpy.ndarray, shape (N,)
        A numpy array of the x-values of the measured data, or a created array.

    Notes
    -----
    Does not change the scale/domain of the input `x_data` if it is given, only
    converts it to an array.

    """
    if check_finite:
        y = np.asarray_chkfinite(data)
    else:
        y = np.asarray(data)
    if x_data is None:
        x = np.linspace(x_min, x_max, y.shape[0])
    else:
        x = np.asarray(x_data)

    return y, x


def _diff_2_diags(data_size, lower_only=True):
    """
    Creates the the diagonals of the square of a second-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    lower_only : bool, optional
        If True (default), will return only the lower diagonals of the
        matrix. If False, will include all diagonals of the matrix.

    Returns
    -------
    output : numpy.ndarray
        The array containing the diagonal data. Has a shape of (3, `data_size`)
        if `lower_only` is True, otherwise (5, `data_size`).

    Notes
    -----
    Equivalent to calling::

        from pybaselines.utils import difference_matrix
        diff_matrix = difference_matrix(data_size, 2)
        output = (diff_matrix.T @ diff_matrix).todia().data[::-1]
        if lower_only:
            output = output[2:]

    but is several orders of magnitude times faster.

    The data is output in the banded format required by SciPy's solve_banded
    and solveh_banded.

    """
    output = np.ones((3 if lower_only else 5, data_size))

    output[-1, -1] = output[-1, -2] = output[-2, -1] = 0
    output[-2, 0] = output[-2, -2] = -2
    output[-2, 1:-2] = -4
    output[-3, 1] = output[-3, -2] = 5
    output[-3, 2:-2] = 6

    if not lower_only:
        output[0, 0] = output[1, 0] = output[0, 1] = 0
        output[1, 1] = output[1, -1] = -2
        output[1, 2:-1] = -4

    return output


def _diff_1_diags(data_size, lower_only=True):
    """
    Creates the the diagonals of the square of a first-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    lower_only : bool, optional
        If True (default), will return only the lower diagonals of the
        matrix. If False, will include all diagonals of the matrix.

    Returns
    -------
    output : numpy.ndarray
        The array containing the diagonal data. Has a shape of (2, `data_size`)
        if `lower_only` is True, otherwise (3, `data_size`).

    Notes
    -----
    Equivalent to calling::

        from pybaselines.utils import difference_matrix
        diff_matrix = difference_matrix(data_size, 1)
        output = (diff_matrix.T @ diff_matrix).todia().data[::-1]
        if lower_only:
            output = output[1:]

    but is several orders of magnitude times faster.

    The data is output in the banded format required by SciPy's solve_banded
    and solveh_banded.

    """
    output = np.full((2 if lower_only else 3, data_size), -1.)

    output[-1, -1] = 0
    output[-2, 0] = output[-2, -1] = 1
    output[-2, 1:-1] = 2

    if not lower_only:
        output[0, 0] = 0

    return output


def _diff_3_diags(data_size, lower_only=True):
    """
    Creates the the diagonals of the square of a third-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    lower_only : bool, optional
        If True (default), will return only the lower diagonals of the
        matrix. If False, will include all diagonals of the matrix.

    Returns
    -------
    output : numpy.ndarray
        The array containing the diagonal data. Has a shape of (4, `data_size`)
        if `lower_only` is True, otherwise (7, `data_size`).

    Notes
    -----
    Equivalent to calling::

        from pybaselines.utils import difference_matrix
        diff_matrix = difference_matrix(data_size, 3)
        output = (diff_matrix.T @ diff_matrix).todia().data[::-1]
        if lower_only:
            output = output[3:]

    but is several orders of magnitude times faster.

    The data is output in the banded format required by SciPy's solve_banded
    and solveh_banded.

    """
    output = np.full((4 if lower_only else 7, data_size), -1.)

    for row in range(-1, -4, -1):
        output[row, -4 - row:] = 0

    output[-2, 0] = output[-2, -3] = 3
    output[-2, 1:-3] = 6
    output[-3, 0] = output[-3, -2] = -3
    output[-3, 1] = output[-3, -3] = -12
    output[-3, 2:-3] = -15
    output[-4, 0] = output[-4, -1] = 1
    output[-4, 1] = output[-4, -2] = 10
    output[-4, 2] = output[-4, -3] = 19
    output[-4, 3:-3] = 20

    if not lower_only:
        for row in range(3):
            output[row, :3 - row] = 0

        output[1, 2] = output[1, -1] = 3
        output[1, 3:-1] = 6
        output[2, 1] = output[2, -1] = -3
        output[2, 2] = output[2, -2] = -12
        output[2, 3:-2] = -15

    return output


def diff_penalty_diagonals(data_size, diff_order=2, lower_only=True, padding=0):
    """
    Creates the diagonals of the finite difference penalty matrix.

    If `D` is the finite difference matrix, then the finite difference penalty
    matrix is defined as ``D.T @ D``. The penalty matrix is banded and symmetric, so
    the non-zero diagonal bands can be computed efficiently.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : int, optional
        The integer differential order; must be >= 0. Default is 2.
    lower_only : bool, optional
        If True (default), will return only the lower diagonals of the
        matrix. If False, will include all diagonals of the matrix.
    padding : int, optional
        The number of extra layers of zeros to add to the bottom and top, if
        `lower_only` is True. Useful if working with other diagonal arrays with
        a different number of rows. Default is 0, which adds no extra layers.
        Negative `padding` is treated as equivalent to 0.

    Returns
    -------
    diagonals : numpy.ndarray
        The diagonals of the finite difference penalty matrix.

    Raises
    ------
    ValueError
        Raised if `diff_order` is negative or if `data_size` less than 1.

    Notes
    -----
    Equivalent to calling::

        from pybaselines.utils import difference_matrix
        diff_matrix = difference_matrix(data_size, diff_order)
        output = (diff_matrix.T @ diff_matrix).todia().data[::-1]
        if lower_only:
            output = output[diff_order:]

    but is several orders of magnitude times faster.

    The data is output in the banded format required by SciPy's solve_banded
    and solveh_banded functions.

    """
    if diff_order < 0:
        raise ValueError('the difference order must be >= 0')
    elif data_size <= 0:
        raise ValueError('data size must be > 0')

    # the fast, hard-coded values require that data_size > 2 * diff_order + 1,
    # otherwise, the band structure has to actually be calculated
    if diff_order == 0:
        diagonals = np.ones((1, data_size))
    elif data_size < 2 * diff_order + 1 or diff_order > 3:
        diff_matrix = difference_matrix(data_size, diff_order, 'csc')
        # scipy's diag_matrix stores the diagonals in opposite order of
        # the typical LAPACK banded structure
        diagonals = (diff_matrix.T @ diff_matrix).todia().data[::-1]
        if lower_only:
            diagonals = diagonals[diff_order:]
    else:
        diag_func = {1: _diff_1_diags, 2: _diff_2_diags, 3: _diff_3_diags}[diff_order]
        diagonals = diag_func(data_size, lower_only)

    if padding > 0:
        pad_layers = np.zeros((padding, data_size))
        if lower_only:
            diagonals = np.concatenate((diagonals, pad_layers))
        else:
            diagonals = np.concatenate((pad_layers, diagonals, pad_layers))

    return diagonals


def _check_scalar_variable(value, allow_zero=False, variable_name='lam', **asarray_kwargs):
    """
    Ensures the input is a scalar value.

    Parameters
    ----------
    value : float or array-like
        The value to check.
    allow_zero : bool, optional
        If False (default), only allows `value` > 0. If True, allows `value` >= 0.
    variable_name : str, optional
        The name displayed if an error occurs. Default is 'lam'.
    **asarray_kwargs : dict
        Additional keyword arguments to pass to :func:`numpy.asarray`.

    Returns
    -------
    output : float
        The verified scalar value.

    Raises
    ------
    ValueError
        Raised if `value` is less than or equal to 0 if `allow_zero` is False or
        less than 0 if `allow_zero` is True.

    """
    output = _check_scalar(value, 1, **asarray_kwargs)[0]
    if allow_zero:
        operation = operator.lt
        text = 'greater than or equal to'
    else:
        operation = operator.le
        text = 'greater than'
    if np.any(operation(output, 0)):
        raise ValueError(f'{variable_name} must be {text} 0')

    # use an empty tuple to get the single scalar value; that way, if the input
    # is a single item in an array, it is converted to a single scalar
    return output[()]


def _check_lam(lam, allow_zero=False):
    """
    Ensures the regularization parameter `lam` is a scalar greater than 0.

    Parameters
    ----------
    lam : float or array-like
        The regularization parameter, lambda, used in Whittaker smoothing and
        penalized splines.
    allow_zero : bool
        If False (default), only allows `lam` values > 0. If True, allows `lam` >= 0.

    Returns
    -------
    float
        The scalar `lam` value.

    Raises
    ------
    ValueError
        Raised if `lam` is less than or equal to 0.

    Notes
    -----
    Array-like `lam` values could be permitted, but they require using the full
    banded penalty matrix. Many functions use only half of the penalty matrix due
    to its symmetry; that symmetry is broken when using an array for `lam`, so allowing
    an array `lam` would change how the system is solved. Further, array-like `lam`
    values with large changes in scale cause some instability and/or discontinuities
    when using Whittaker smoothing or penalized splines. Thus, it is easier and better
    to only allow scalar `lam` values.

    TODO will maybe change this in the future to allow array-like `lam`, and the
    solver will be determined based on that; however, until then, want to ensure users
    don't unknowingly use an array-like `lam` when it doesn't work.

    """
    return _check_scalar_variable(lam, allow_zero)


def _check_half_window(half_window, allow_zero=False):
    """
    Ensures the half-window is an integer and has an appropriate value.

    Parameters
    ----------
    half_window : int, optional
        The half-window used for the smoothing functions. Used
        to pad the left and right edges of the data to reduce edge
        effects. Default is 0, which provides no padding.
    allow_zero : bool, optional
        If True, allows `half_window` to be 0; otherwise, `half_window`
        must be at least 1. Default is False.

    Returns
    -------
    output_half_window : int
        The verified half-window value.

    Raises
    ------
    TypeError
        Raised if the integer converted `half_window` is not equal to the input
        `half_window`.

    """
    output_half_window = _check_scalar_variable(
        half_window, allow_zero, 'half_window', dtype=np.intp
    )
    if output_half_window != half_window:
        raise TypeError('half_window must be an integer')

    return output_half_window


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
        (`diff_order` * 2 + 1, N).
    weight_array : numpy.ndarray, shape (N,), optional
        The weighting array.

    Raises
    ------
    ValueError
        Raised is `diff_order` is less than 1 or if `weights` and `data` do not
        have the same shape.

    Warns
    -----
    UserWarning
        Raised if `diff_order` is greater than 3.

    """
    y = np.asarray_chkfinite(data)
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
    num_y = y.shape[0]
    if weights is None:
        weight_array = np.ones(num_y)
    else:
        weight_array = np.asarray(weights)
        if copy_weights:
            weight_array = weight_array.copy()

        if weight_array.shape != y.shape:
            raise ValueError('weights must have the same shape as the input data')

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
    y, x = _yx_arrays(data, x_data)
    if x_data is None:
        original_domain = np.array([-1., 1.])
    else:
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1., 1.]))

    if weights is None:
        weight_array = np.ones(len(y))
    else:
        weight_array = np.asarray(weights)
        if copy_weights:
            weight_array = weight_array.copy()

        if weight_array.shape != y.shape:
            raise ValueError('weights must have the same shape as the input data')

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
    y = np.asarray(data)
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
    hw = _check_half_window(half_window, allow_zero)
    return pad_edges(data, hw, **pad_kwargs)


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
    y, x = _yx_arrays(data, x_data)
    # TODO should remove the x-scaling here since most methods don't need it; can
    # make a separate function for it, which _setup_polynomial could also use
    if x_data is None:
        original_domain = np.array([-1., 1.])
    else:
        original_domain = np.polynomial.polyutils.getdomain(x)
        x = np.polynomial.polyutils.mapdomain(x, original_domain, np.array([-1., 1.]))
    if weights is not None:
        weight_array = np.asarray(weights, bool)
    else:
        weight_array = np.ones(y.shape[0], bool)

    return y, x, weight_array, original_domain


def _setup_splines(data, x_data=None, weights=None, spline_degree=3, num_knots=10,
                   penalized=True, diff_order=3, lam=1, make_basis=True):
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
    penalty_diagonals : numpy.ndarray, shape (`diff_order` + 1, N)
        The finite difference penalty matrix, in LAPACK's lower banded format (see
        :func:`scipy.linalg.solveh_banded`).
        The penalty matrix for the spline. Only returned if both `penalized`
        and `make_basis` are True.

    Raises
    ------
    ValueError
        Raised if `diff_order` is less than 1, if `weights` and `data` do not have
        the same shape, if `num_knots` is less than 2, if the number of spline
        basis functions (`num_knots` + `spline_degree` - 1) is <= `diff_order`, or
        if `spline_degree` is less than 0.

    Warns
    -----
    UserWarning
        Raised if `diff_order` is greater than 4.

    Notes
    -----
    `degree` is used instead of `order` like for polynomials since the order of a spline
    is defined by convention as `degree` + 1.

    """
    y, x = _yx_arrays(data, x_data, check_finite=True)
    if weights is not None:
        weight_array = np.asarray(weights)
        if weight_array.shape != y.shape:
            raise ValueError('weights must have the same shape as the input data')
    else:
        weight_array = np.ones(y.shape[0])
    if not make_basis:
        return y, x, weight_array

    if num_knots < 2:  # num_knots == 2 means the only knots are the two endpoints
        raise ValueError('the number of knots must be at least 2')
    elif spline_degree < 0:
        raise ValueError('spline degree must be >= 0')
    # explicitly cast x and y as floats since most scipy functions do so anyway, so
    # can just do it once
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
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
        num_bases, diff_order, padding=spline_degree - diff_order
    )

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
