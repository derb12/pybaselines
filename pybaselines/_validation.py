# -*- coding: utf-8 -*-
"""Code for validating inputs.

Created on December 9, 2021
@author: Donald Erb

"""

import numpy as np


def _check_scalar(data, desired_length, fill_scalar=False, coerce_0d=True, **asarray_kwargs):
    """
    Checks if the input is scalar and potentially coerces it to the desired length.

    Only intended for one dimensional data.

    Parameters
    ----------
    data : array-like
        Either a scalar value or an array. Array-like inputs with only 1 item will also
        be considered scalar.
    desired_length : int
        If `data` is an array, `desired_length` is the length the array must have. If `data`
        is a scalar and `fill_scalar` is True, then `desired_length` is the length of the output.
    fill_scalar : bool, optional
        If True and `data` is a scalar, then will output an array with a length of
        `desired_length`. Default is False, which leaves scalar values unchanged.
    coerce_0d : bool, optional
        If True (default) and `data` is an array-like, `output` will be a scalar. If
        False, `output` will also be an array with shape (1,).
    **asarray_kwargs : dict
        Additional keyword arguments to pass to :func:`numpy.asarray`.

    Returns
    -------
    output : numpy.ndarray or numpy.number
        The array of values or the single array scalar, depending on the input parameters.
    is_scalar : bool
        True if the input was a scalar value or had a length of 1; otherwise, is False.

    Raises
    ------
    ValueError
        Raised if `data` is not a scalar and its length is not equal to `desired_length`.

    """
    output = np.asarray(data, **asarray_kwargs)
    ndim = output.ndim
    if not ndim:
        is_scalar = True
    else:
        if ndim > 1:  # coerce to 1d shape
            output = output.reshape(-1)
        len_output = len(output)
        if len_output == 1 and coerce_0d:
            is_scalar = True
            output = np.asarray(output[0], **asarray_kwargs)
        else:
            is_scalar = False

    if is_scalar:
        if fill_scalar:
            output = np.full(desired_length, output)
        else:
            # index with an empty tuple to get the single scalar while maintaining the numpy dtype
            output = output[()]
    elif desired_length is not None and len_output != desired_length:
        raise ValueError(f'desired length was {desired_length} but instead got {len_output}')

    return output, is_scalar


def _check_scalar_variable(value, allow_zero=False, variable_name='lam', two_d=False,
                           **asarray_kwargs):
    """
    Ensures the input is a scalar value.

    Parameters
    ----------
    value : numpy.Number or array-like
        The value to check.
    allow_zero : bool, optional
        If False (default), only allows `value` > 0. If True, allows `value` >= 0.
    variable_name : str, optional
        The name displayed if an error occurs. Default is 'lam'.
    two_d : bool, optional
        If True, will output an array with two values. If False (default), will
        return a single scalar value.
    **asarray_kwargs : dict
        Additional keyword arguments to pass to :func:`numpy.asarray`.

    Returns
    -------
    output : numpy.Number or numpy.ndarray[numpy.Number, numpy.Number]
        The verified scalar value(s).

    Raises
    ------
    ValueError
        Raised if `value` is less than or equal to 0 if `allow_zero` is False or
        less than 0 if `allow_zero` is True.

    """
    if two_d:
        desired_length = 2
        fill_scalar = True
    else:
        desired_length = 1
        fill_scalar = False
    output = _check_scalar(value, desired_length, fill_scalar=fill_scalar, **asarray_kwargs)[0]
    if allow_zero:
        operation = np.less
        text = 'greater than or equal to'
    else:
        operation = np.less_equal
        text = 'greater than'
    if np.any(operation(output, 0)):
        raise ValueError(f'{variable_name} must be {text} 0')

    return output


def _check_array(array, dtype=None, order=None, check_finite=False, ensure_1d=True,
                 ensure_2d=False, two_d=False):
    """
    Validates the shape and values of the input array and controls the output parameters.

    Parameters
    ----------
    array : array-like
        The input array to check.
    dtype : type or numpy.dtype, optional
        The dtype to cast the output array. Default is None, which uses the typing of `array`.
    order : {None, 'C', 'F'}, optional
        The order for the output array. Default is None, which will use the default array
        ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
    check_finite : bool, optional
        If True, will raise an error if any values in `array` are not finite. Default is False,
        which skips the check.
    ensure_1d : bool, optional
        If True (default), will raise an error if the shape of `array` is not a one dimensional
        array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
    ensure_2d : bool, optional
        If True, will raise an error if `array` is not a two dimensional array or a three
        dimensional array with shape (M, N, 1), (1, M, N), or (M, 1, N). Default is False.
    two_d : bool, optional
        If True, will raise an error if the shape of `array` is not a two dimensional array with
        shape (M, N) where M or N must be greater than 1.

    Returns
    -------
    output : numpy.ndarray
        The array after performing all validations.

    Raises
    ------
    ValueError
        Raised if `ensure_1d` is True and `array` does not have a shape of (N,) or
        (N, 1) or (1, N).

    Notes
    -----
    If `ensure_1d` is True and `array` has a shape of (N, 1) or (1, N), it is reshaped to
    (N,) for better compatibility for all functions. Likewise, `ensure_2d` will flatten to
    (M, N).

    """
    if check_finite:
        array_func = np.asarray_chkfinite
    else:
        array_func = np.asarray
    output = array_func(array, dtype=dtype, order=order)
    if ensure_1d:
        output = np.array(output, copy=False, ndmin=1)
        dimensions = output.ndim
        if dimensions == 2 and 1 in output.shape:
            output = output.reshape(-1)
        elif dimensions != 1:
            raise ValueError('must be a one dimensional array')
    elif two_d:
        output = np.array(output, copy=False, ndmin=2)
        dimensions = output.ndim
        if dimensions == 2 and 1 in output.shape:
            raise ValueError(
                'input data must be a two dimensional array with more than just one row or column'
            )
        if ensure_2d:
            if dimensions == 3 and 1 in output.shape:
                output_shape = np.array(output.shape)
                flat_dims = ~np.equal(output_shape, 1)
                output = output.reshape(output_shape[flat_dims]).shape
            elif dimensions != 2:
                raise ValueError('must be a two dimensional array')
    elif ensure_2d and not two_d:
        raise ValueError('two_d must be True if using ensure_2d')

    return output


def _check_sized_array(array, length, dtype=None, order=None, check_finite=False,
                       ensure_1d=True, axis=-1, name='weights', ensure_2d=False,
                       two_d=False):
    """
    Validates the input array and ensures its length is correct.

    Parameters
    ----------
    array : array-like
        The input array to check.
    length : int
        The length that the input should have on the specified `axis`.
    dtype : type or numpy.dtype, optional
        The dtype to cast the output array. Default is None, which uses the typing of `array`.
    order : {None, 'C', 'F'}, optional
        The order for the output array. Default is None, which will use the default array
        ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
    check_finite : bool, optional
        If True, will raise an error if any values if `array` are not finite. Default is False,
        which skips the check.
    ensure_1d : bool, optional
        If True (default), will raise an error if the shape of `array` is not a one dimensional
        array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
    axis : int, optional
        The axis of the input on which to check its length. Default is -1.
    name : str, optional
        The name for the variable if an exception is raised. Default is 'weights'.

    Returns
    -------
    output : numpy.ndarray
        The array after performing all validations.

    Raises
    ------
    ValueError
        Raised if `array` does not match `length` on the given `axis`.

    """
    output = _check_array(
        array, dtype=dtype, order=order, check_finite=check_finite, ensure_1d=ensure_1d,
        ensure_2d=ensure_2d, two_d=two_d
    )
    if not np.equal(output.shape[axis], length).all():
        raise ValueError(
            f'length mismatch for {name}; expected {length} but got {output.shape[axis]}'
        )
    return output


def _yx_arrays(data, x_data=None, check_finite=False, dtype=None, order=None, ensure_1d=True,
               axis=-1):
    """
    Converts input data into numpy arrays and provides x data if none is given.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1. to 1. with N points.
    check_finite : bool, optional
        If True, will raise an error if any values if `array` are not finite. Default is False,
        which skips the check.
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
    y : numpy.ndarray, shape (N,)
        A numpy array of the y-values of the measured data.
    x : numpy.ndarray, shape (N,)
        A numpy array of the x-values of the measured data, or a created array.

    Notes
    -----
    Does not change the scale/domain of the input `x_data` if it is given, only
    converts it to an array.

    """
    y = _check_array(
        data, dtype=dtype, order=order, check_finite=check_finite, ensure_1d=ensure_1d
    )
    len_y = y.shape[axis]
    if x_data is None:
        x = np.linspace(-1, 1, len_y)
    else:
        x = _check_sized_array(
            x_data, len_y, dtype=dtype, order=order, check_finite=check_finite,
            ensure_1d=True, axis=0, name='x_data'
        )

    return y, x


def _yxz_arrays(data, x_data=None, z_data=None, check_finite=False, dtype=None, order=None,
                ensure_2d=True, x_axis=-2, z_axis=-1):
    """
    Converts input data into numpy arrays and provides x and z data if none are given.

    Parameters
    ----------
    data : array-like, shape (M, N)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (M,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1. to 1. with N points.
    z_data : array-like, shape (N,), optional
        The z-values of the measured data. Default is None, which will create an
        array from -1. to 1. with N points.
    check_finite : bool, optional
        If True, will raise an error if any values if `array` are not finite. Default is False,
        which skips the check.
    dtype : type or numpy.dtype, optional
        The dtype to cast the output array. Default is None, which uses the typing of `array`.
    order : {None, 'C', 'F'}, optional
        The order for the output array. Default is None, which will use the default array
        ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
    ensure_2d : bool, optional
        If True (default), will raise an error if the shape of `array` is not a two dimensional
        array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).

    Returns
    -------
    y : numpy.ndarray, shape (M, N)
        A numpy array of the y-values of the measured data.
    x : numpy.ndarray, shape (M,)
        A numpy array of the x-values of the measured data, or a created array.
    z : numpy.ndarray, shape (N,)
        A numpy array of the z-values of the measured data, or a created array.

    Notes
    -----
    Does not change the scale/domain of the input `x_data` or `z_data` if they
    are given, only converts them to arrays.

    """
    y = _check_array(
        data, dtype=dtype, order=order, check_finite=check_finite, ensure_1d=False,
        ensure_2d=ensure_2d, two_d=True
    )
    x_len = y.shape[x_axis]
    z_len = y.shape[z_axis]
    if x_data is None:
        x = np.linspace(-1, 1, x_len)
    else:
        x = _check_sized_array(
            x_data, x_len, dtype=dtype, order=order, check_finite=check_finite,
            ensure_1d=True, axis=0, name='x_data'
        )
    if z_data is None:
        z = np.linspace(-1, 1, z_len)
    else:
        z = _check_sized_array(
            z_data, z_len, dtype=dtype, order=order, check_finite=check_finite,
            ensure_1d=True, axis=0, name='z_data'
        )

    return y, x, z


def _check_lam(lam, allow_zero=False, two_d=False, dtype=float):
    """
    Ensures the regularization parameter `lam` is a scalar greater than 0.

    Parameters
    ----------
    lam : float or array-like
        The regularization parameter, lambda, used in Whittaker smoothing and
        penalized splines.
    allow_zero : bool
        If False (default), only allows `lam` values > 0. If True, allows `lam` >= 0.
    two_d : bool, optional
        If True, will output an array with two values. If False (default), will
        return a single scalar value.
    dtype : type or numpy.dtype, optional
        The dtype to cast the lam value. Default is float.

    Returns
    -------
    numpy.Number or numpy.ndarray[numpy.Number, numpy.Number]
        The verified `lam` value(s).

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
    NOTE for future: if multiplying an array `lam` with the penalties in banded format,
    do not reverse the order (ie. keep it like the output of sparse.dia.data), multiply
    by the array, and then shift the rows based on the difference order (same procedure
    as done for aspls). That will give the same output as
    ``(diags(lam) @ D.T @ D).todia().data[::-1]``.

    """
    return _check_scalar_variable(lam, allow_zero, two_d=two_d, variable_name='lam', dtype=dtype)


def _check_half_window(half_window, allow_zero=False, two_d=False):
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
    two_d : bool, optional
        If True, will output an array with two values. If False (default), will
        return a single scalar value.

    Returns
    -------
    output_half_window : int or numpy.ndarray[int, int]
        The verified half-window value(s).

    Raises
    ------
    TypeError
        Raised if the integer converted `half_window` is not equal to the input
        `half_window`.

    """
    output_half_window = _check_scalar_variable(
        half_window, allow_zero, variable_name='half_window', two_d=two_d, dtype=np.intp
    )
    if not two_d and output_half_window != half_window:
        raise TypeError('half_window must be an integer')

    return output_half_window


def _check_optional_array(data_size, array=None, dtype=None, order=None, check_finite=False,
                          copy_input=False, name='weights', ensure_1d=True, axis=-1):
    """
    Validates the length of the input array or creates an array of ones if no input is given.

    Parameters
    ----------
    data_size : int or Container[int, int]
        The shape that the input should have.
    array : array-like, shape (`data_size`), optional
        The array to validate. Default is None, which will create an array of ones with length
        equal to `data_size`.
    copy_input : bool, optional
        If True, returns a copy of the input `array` if it is not None. Default is False.
    dtype : type or numpy.dtype, optional
        The dtype to cast the output array. Default is None, which uses the typing of `array`.
    order : {None, 'C', 'F'}, optional
        The order for the output array. Default is None, which will use the default array
        ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
    check_finite : bool, optional
        If True, will raise an error if any values if `array` are not finite. Default is False,
        which skips the check.
    name : str, optional
        The name for the variable if an exception is raised. Default is 'weights'.
    ensure_1d : bool, optional
        If True (default), will raise an error if the shape of `array` is not a one dimensional
        array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N). If False,
        will ignore the shape of `array`.
    axis : int, optional
        The axis of the input on which to check its length. Default is -1.

    Returns
    -------
    output_array : numpy.ndarray, shape (`data_size`)
        The validated array or the new ones array.

    """
    if array is None:
        output_array = np.ones(data_size)
    else:
        output_array = _check_sized_array(
            array, data_size, dtype=dtype, order=order, check_finite=check_finite,
            ensure_1d=ensure_1d, name=name, axis=axis
        )
        if copy_input:
            output_array = output_array.copy()

    return output_array


def _get_row_col_values(value, **asarray_kwargs):
    """
    Determines the row and column values for an input that can be scalar or up to length 4.

    Parameters
    ----------
    value : numpy.number or Sequence[numpy.number, ...]
        The value(s) corresponding to the first row, last row, first column, and last
        column.

    Returns
    -------
    output : numpy.ndarray, shape (4,)
        The array of length 4 with values first row, last row, first column, last column.

    Raises
    ------
    ValueError
        Raised if the input value was a sequence with 1, 2, or 4 values.

    """
    # can either be len 1, 2, or 4
    output, scalar_input = _check_scalar(value, None, **asarray_kwargs)
    if scalar_input:
        output = np.full(4, output)
    else:
        len_input = len(output)
        if len_input not in (2, 4):
            raise ValueError('input must either be a single value or an array with length 2 or 4')
        elif len_input == 2:
            output = np.array([output[0], output[0], output[1], output[1]])

    return output
