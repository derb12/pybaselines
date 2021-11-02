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

import warnings

import numpy as np
from scipy.interpolate import splev
from scipy.linalg import solveh_banded
from scipy.sparse import csr_matrix

from ._compat import _HAS_PENTAPY, _pentapy_solve
from .utils import _pentapy_solver, ParameterWarning, difference_matrix, optimize_window, pad_edges


def _yx_arrays(data, x_data=None, x_min=-1., x_max=1.):
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
    y = np.asarray(data)
    if x_data is None:
        x = np.linspace(x_min, x_max, y.shape[0])
    else:
        x = np.asarray(x_data)

    return y, x


def _diff_2_diags(data_size, upper_only=True):
    """
    Creates the the diagonals of the square of a second-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    upper_only : bool, optional
        If True (default), will return only the upper diagonals of the
        matrix. If False, will include all diagonals of the matrix.

    Returns
    -------
    output : numpy.ndarray
        The array containing the diagonal data. Has a shape of (3, `data_size`)
        if `upper_only` is True, otherwise (5, `data_size`).

    Notes
    -----
    Equivalent to calling:

        diff_matrix = difference_matrix(data_size, 2)
        diag_matrix = (diff_matrix.T * diff_matrix).todia()
        if upper_only:
            output = diag_matrix.data[2:][::-1]
        else:
            output = diag_matrix.data[::-1]

    but is several orders of magnitude times faster. The data is reversed
    in order to fit the format required by SciPy's solve_banded and solveh_banded.

    """
    output = np.ones((3 if upper_only else 5, data_size))
    output[0, 0] = output[1, 0] = output[0, 1] = 0
    output[1, 1] = output[1, -1] = -2
    output[2, 1] = output[2, -2] = 5
    output[1, 2:-1] = -4
    output[2, 2:-2] = 6

    if upper_only:
        return output

    output[-1, -1] = output[-1, -2] = output[-2, -1] = 0
    output[-2, 0] = output[-2, -2] = -2
    output[-3, 1] = output[-3, -2] = 5
    output[-2, 1:-2] = -4

    return output


def _diff_1_diags(data_size, upper_only=True, add_zeros=False):
    """
    Creates the the diagonals of the square of a first-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    upper_only : bool, optional
        If True (default), will return only the upper diagonals of the
        matrix. If False, will include all diagonals of the matrix.
    add_zeros : bool, optional
        If True, will stack a row of zeros on top of the output, and on the bottom
        if `upper_only` is False, so that the output array can be added to the output
        of :func:`_diff_2_diags`.

    Returns
    -------
    output : numpy.ndarray
        The array containing the diagonal data. The number of rows depends on
        `upper_only` and `add_zeros`.

    Notes
    -----
    Equivalent to calling:

        diff_matrix = difference_matrix(data_size, 1)
        diag_matrix = (diff_matrix.T * diff_matrix).todia()
        if upper_only:
            output = diag_matrix.data[1:][::-1]
        else:
            output = diag_matrix.data[::-1]

    but is several orders of magnitude times faster. The data is reversed
    in order to fit the format required by SciPy's solve_banded and solveh_banded.

    """
    output = np.full((2 if upper_only else 3, data_size), -1.)

    output[0, 0] = 0
    output[1, 0] = output[1, -1] = 1
    output[1, 1:-1] = 2

    if add_zeros:
        zeros = np.zeros((1, data_size))

    if upper_only:
        if add_zeros:
            output = np.concatenate((zeros, output))
        return output

    output[-1, -1] = 0
    if add_zeros:
        output = np.concatenate((zeros, output, zeros))

    return output


def _setup_whittaker(data, lam, diff_order=2, weights=None, copy_weights=False,
                     upper_only=True, reverse_diags=False):
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
    upper_only : boolean, optional
        If True (default), will include only the upper non-zero diagonals of
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
        (`diff_order` + 1, N) if `upper_only` is True, otherwise
        (`diff_order` * 2 + 1, N).

    weight_array : numpy.ndarray, shape (N,), optional
        The weighting array.

    Raises
    ------
    ValueError
        Raised is `diff_order` is less than 1.
    ValueError
        Raised if `weights` and `data` do not have the same shape.

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
    # use hard-coded values for diff_order of 1 and 2 since it is much faster
    # TODO need to do a shape check to ensure the vast versions are valid; len(y)
    # must be >= 2 * diff_order + 1 (almost surely is); if not, need to do the
    # actual calculation with the difference matrix, or just raise an error
    if diff_order == 1:
        diagonal_data = _diff_1_diags(num_y, upper_only)
    elif diff_order == 2:
        diagonal_data = _diff_2_diags(num_y, upper_only)
    else:  # TODO figure out the general formula to avoid using the sparse matrices
        # csc format is fastest for the D.T * D operation
        diff_matrix = difference_matrix(num_y, diff_order, 'csc')
        diff_matrix = diff_matrix.T * diff_matrix
        diagonal_data = diff_matrix.todia().data[diff_order if upper_only else 0:][::-1]

    if reverse_diags:
        diagonal_data = diagonal_data[::-1]

    if weights is None:
        weight_array = np.ones(num_y)
    else:
        weight_array = np.asarray(weights)
        if copy_weights:
            weight_array = weight_array.copy()

        if weight_array.shape != y.shape:
            raise ValueError('weights must have the same shape as the input data')

    return y, lam * diagonal_data, weight_array


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
        output_half_window = half_window
    else:
        output_half_window = optimize_window(y, **window_kwargs)

    return y, output_half_window


def _setup_smooth(data, half_window=0, **pad_kwargs):
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
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from smoothing.

    Returns
    -------
    numpy.ndarray, shape (N + 2 * half_window)
        The padded array of data.

    """
    return pad_edges(data, half_window, **pad_kwargs)


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


def spline_basis(x, num_knots=10, spline_degree=3, penalized=False):
    """
    Creates the basis matrix for B-splines and P-splines.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        The array of x-values
    num_knots : int, optional
        The number of interior knots for the spline. Default is 10.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    penalized : bool, optional
        Whether the basis matrix should be for a penalized spline or a regular
        B-spline. Default is False, which creates the basis for a B-spline.

    Returns
    -------
    basis : numpy.ndarray
        The basis matrix representing the spline, similar to the Vandermonde matrix
        for polynomials.

    Notes
    -----
    If `penalized` is True, makes the knots uniformly spaced to create p-splines. That
    way, can use a finite difference matrix to impose penalties on the spline.

    `degree` is used instead of `order` like for polynomials since the order of a spline
    is defined by convention as `degree` + 1.

    References
    ----------
    Eilers, P., et al. Twenty years of P-splines. SORT: Statistics and Operations Research
    Transactions, 2015, 39(2), 149-186.

    Hastie, T., et al. The Elements of Statistical Learning. Springer, 2017. Chapter 5.

    """
    spline_order = spline_degree + 1
    if penalized:
        x_min = x.min()
        x_max = x.max()
        # number of sections is num_knots - 1 since counting the first and last knots as inner knots
        dx = (x_max - x_min) / (num_knots - 1)
        knots = np.linspace(
            x_min - spline_degree * dx, x_max + spline_degree * dx,
            num_knots + 2 * spline_degree
        )
    else:
        # TODO maybe provide a better way to select knot positions for regular B-splines
        inner_knots = np.percentile(x, np.linspace(0, 100, num_knots))
        knots = np.concatenate((
            np.repeat(inner_knots[0], spline_degree), inner_knots,
            np.repeat(inner_knots[-1], spline_degree)
        ))

    num_bases = len(knots) - spline_order
    basis = np.empty((num_bases, len(x)))
    coefs = np.zeros(num_bases)
    # TODO would be faster to simply calculate the spline coefficients using de Boor's recursive
    # algorithm; does it also give just the coefficients without all the extra zeros? would be
    # nice if so, since it would use less space; also could then probably make a sparse or banded
    # matrix rather than the full, dense array

    # adapted from Scipy forums at: http://scipy-user.10969.n7.nabble.com/B-spline-basis-functions
    for i in range(num_bases):
        coefs[i] = 1  # evaluate the i-th basis within splev
        basis[i] = splev(x, (knots, coefs, spline_degree))
        coefs[i] = 0  # reset back to zero

    # transpose to get a shape similar to the Vandermonde matrix
    # TODO maybe it's preferable to not transpose, since typically basis.T is used in
    # calculations more than basis; maybe make it a boolean input
    return basis.T


def _setup_splines(data, x_data=None, weights=None, spline_degree=3, num_knots=10,
                   penalized=True, diff_order=3, lam=1, sparse_basis=True):
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
    sparse_basis : bool, optional
        If True (default), will convert the spline basis to a sparse matrix with CSR
        format.

    Returns
    -------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, converted to a numpy array.
    x : numpy.ndarray, shape (N,)
        The x-values for fitting the spline.
    basis : numpy.ndarray or scipy.sparse.csr.csr_matrix
        The spline basis matrix. Is sparse with CSR format if `sparse_basis` is True.
    weight_array : numpy.ndarray, shape (N,)
        The weight array for fitting a polynomial to the data.
    penalty_matrix : scipy.sparse.csr.csr_matrix
        The penalty matrix for the spline. Only returned if `penalized` is True.

    Raises
    ------
    ValueError
        Raised is `diff_order` is less than 1 or if `weights` and `data` do not have
        the same shape.

    Warns
    -----
    UserWarning
        Raised if `diff_order` is greater than 4.

    """
    y, x = _yx_arrays(data, x_data)
    if weights is not None:
        weight_array = np.asarray(weights)
        if weight_array.shape != y.shape:
            raise ValueError('weights must have the same shape as the input data')
    else:
        weight_array = np.ones(y.shape[0])
    basis = spline_basis(x, num_knots, spline_degree, penalized)
    if sparse_basis:
        basis = csr_matrix(basis)
    if not penalized:
        return y, x, basis, weight_array

    if diff_order < 1:
        raise ValueError(
            'the differential order must be > 0 for spline methods'
        )
    elif diff_order > 4:
        warnings.warn(
            ('differential orders greater than 4 can have numerical issues;'
             ' consider using a differential order of 2 or 3 instead'),
            ParameterWarning
        )
    diff_matrix = difference_matrix(basis.shape[1], diff_order, 'csc')
    penalty_matrix = lam * diff_matrix.T * diff_matrix

    return y, x, basis, weight_array, penalty_matrix


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
    main_diag_idx = diff_order if using_pentapy else -1
    diagonals[main_diag_idx] = diagonals[main_diag_idx] + weight_array
    if using_pentapy:
        smooth_y = _pentapy_solve(diagonals, weight_array * y, True, True, _pentapy_solver())
    else:
        smooth_y = solveh_banded(
            diagonals, weight_array * y, overwrite_ab=True, overwrite_b=True, check_finite=False
        )

    return smooth_y, weight_array
