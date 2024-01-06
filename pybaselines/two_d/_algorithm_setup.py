# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on April 8, 2023
@author: Donald Erb

"""

from contextlib import contextmanager
from functools import partial, wraps
import itertools
import warnings

import numpy as np
from scipy.ndimage import grey_opening

from ..utils import (
    ParameterWarning, _determine_sorts, _inverted_sort, _sort_array2d, pad_edges,
    relative_difference
)
from ._spline_utils import PSpline2D
from .._validation import (
    _check_array, _check_half_window, _check_optional_array, _check_scalar, _check_scalar_variable,
    _check_sized_array, _yxz_arrays
)
from ._whittaker_utils import PenalizedSystem2D


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
    pspline : PSpline2D or None
        The PSpline2D object for setting up and solving penalized spline algorithms. Is None
        if no penalized spline setup has been performed (typically done in :meth:`._setup_spline`).
    vandermonde : numpy.ndarray or None
        The Vandermonde matrix for solving polynomial equations. Is None if no polynomial
        setup has been performed (typically done in :meth:`._setup_polynomial`).
    whittaker_system : PenalizedSystem2D or None
        The PenalizedSystem2D object for setting up and solving Whittaker-smoothing-based
        algorithms. Is None if no Whittaker setup has been performed (typically done in
        :meth:`_setup_whittaker`).
    x : numpy.ndarray or None
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same size as the input `data.shape[-1]` and has min
        and max values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).
    z : numpy.ndarray or None
        The z-values for the object. If initialized with None, then `z` is initialized the
        first function call to have the same size as the input `data.shape[-2]` and has min
        and max values of -1 and 1, respectively.
    z_domain : numpy.ndarray
        The minimum and maximum values of `z`. If `z_data` is None during initialization, then
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
        z_data : array-like, shape (M,), optional
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

        """
        self._len = [None, None]
        x_sort_order = None
        z_sort_order = None
        if x_data is None:
            self.x = None
            self.x_domain = np.array([-1., 1.])
        else:
            self.x = _check_array(x_data, check_finite=check_finite)
            self._len[1] = len(self.x)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)
            x_sort_order, x_inverted_order = _determine_sorts(self.x)
            if x_sort_order is not None:
                self.x = self.x[x_sort_order]

        if z_data is None:
            self.z = None
            self.z_domain = np.array([-1., 1.])
        else:
            self.z = _check_array(z_data, check_finite=check_finite)
            self._len[0] = len(self.z)
            self.z_domain = np.polynomial.polyutils.getdomain(self.z)
            z_sort_order, z_inverted_order = _determine_sorts(self.z)
            if z_sort_order is not None:
                self.z = self.z[z_sort_order]

        if x_sort_order is None and z_sort_order is None:
            self._sort_order = None
            self._inverted_order = None
        elif x_sort_order is None:
            self._sort_order = z_sort_order
            self._inverted_order = z_inverted_order
        elif z_sort_order is None:
            self._sort_order = (..., x_sort_order)
            self._inverted_order = (..., x_inverted_order)
        else:
            self._sort_order = (z_sort_order[:, None], x_sort_order[None, :])
            self._inverted_order = (z_inverted_order[:, None], x_inverted_order[None, :])

        self.whittaker_system = None
        self.vandermonde = None
        self.poly_order = -1
        self.pspline = None
        self._check_finite = check_finite
        self._dtype = output_dtype

    def _return_results(self, baseline, params, dtype, sort_keys=(), ensure_2d=False,
                        reshape_baseline=False, reshape_keys=()):
        """
        Re-orders the input baseline and parameters based on the x ordering.

        If `self._sort_order` is None, then no reordering is performed.

        Parameters
        ----------
        baseline : numpy.ndarray, shape (M, N)
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
        baseline : numpy.ndarray, shape (M, N)
            The input `baseline` after re-ordering and setting to the desired dtype.
        params : dict
            The input `params` after re-ordering the values for `sort_keys`.

        """
        if reshape_baseline:
            if ensure_2d:
                baseline = baseline.reshape(self._len)
            else:
                baseline = baseline.reshape(-1, *self._len)
        for key in reshape_keys:
            if key in params:
                # TODO can any params be non-2d that need reshaped?
                params[key] = params[key].reshape(self._len)

        if self._sort_order is not None:
            for key in sort_keys:
                if key in params:  # some parameters are conditionally output
                    # assumes params all all two dimensional arrays
                    params[key] = params[key][self._inverted_order]

            baseline = _sort_array2d(baseline, sort_order=self._inverted_order)
        baseline = baseline.astype(dtype, copy=False)

        return baseline, params

    @classmethod
    def _register(cls, func=None, *, sort_keys=(), dtype=None, order=None, ensure_2d=True,
                  reshape_baseline=False, reshape_keys=()):
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
        dtype : type or numpy.dtype, optional
            The dtype to cast the output array. Default is None, which uses the typing of `array`.
        order : {None, 'C', 'F'}, optional
            The order for the output array. Default is None, which will use the default array
            ordering. Other valid options are 'C' for C ordering or 'F' for Fortran ordering.
        ensure_2d : bool, optional
            If True (default), will raise an error if the shape of `array` is not a two dimensional
            array with shape (M, N) or a three dimensional array with shape (M, N, 1), (M, 1, N),
            or (1, M, N).
        reshape_baseline : bool, optional
            If True, will reshape the output baseline back into the shape of the input data. If
            False (default), will not modify the output baseline shape.
        reshape_keys : tuple, optional
            The keys within the output parameter dictionary that will need reshaped to match the
            shape of the data. For example, used to convert weights for polynomials from 1D back
            into the original shape. Default is ().

        Returns
        -------
        numpy.ndarray
            The calculated baseline.
        dict
            A dictionary of parameters output by the baseline function.

        """
        if func is None:
            return partial(
                cls._register, dtype=dtype, order=order, ensure_2d=ensure_2d,
                reshape_baseline=reshape_baseline, reshape_keys=reshape_keys
            )

        @wraps(func)
        def inner(self, data=None, *args, **kwargs):
            if data is None:
                # not implementing interp_pts for 2D, so data can never
                # be None in 2D
                raise TypeError('"data" cannot be None')

            reset_x = self.x is not None
            reset_z = self.z is not None
            if reset_x or reset_z:
                if reset_x and reset_z:
                    expected_shape = self._len
                    axis = slice(-2, None)
                elif reset_x:
                    expected_shape = self._len[1]
                    axis = -1
                else:
                    expected_shape = self._len[0]
                    axis = -2
                y = _check_sized_array(
                    data, expected_shape, check_finite=self._check_finite, dtype=dtype,
                    order=order, ensure_1d=False, axis=axis, name='data'
                )
            else:
                y, self.x, self.z = _yxz_arrays(
                    data, self.x, self.z, check_finite=self._check_finite, dtype=dtype,
                    order=order, ensure_2d=ensure_2d
                )

            # update self.x and/or self.z just to ensure dtype and order are correct
            if reset_x:
                x_dtype = self.x.dtype
                self.x = _check_array(
                    self.x, dtype=dtype, order=order, check_finite=False, ensure_1d=False
                )
            else:
                self._len[1] = y.shape[-1]
                self.x = np.linspace(-1, 1, self._len[1])
            if reset_z:
                z_dtype = self.z.dtype
                self.z = _check_array(
                    self.z, dtype=dtype, order=order, check_finite=False, ensure_1d=False
                )
            else:
                self._len[0] = y.shape[-2]
                self.z = np.linspace(-1, 1, self._len[0])

            y = _sort_array2d(y, sort_order=self._sort_order)
            if self._dtype is None:
                output_dtype = y.dtype
            else:
                output_dtype = self._dtype

            baseline, params = func(self, y, *args, **kwargs)
            if reset_x:
                self.x = np.array(self.x, dtype=x_dtype, copy=False)
            if reset_z:
                self.z = np.array(self.z, dtype=z_dtype, copy=False)

            return self._return_results(
                baseline, params, output_dtype, sort_keys, ensure_2d,
                reshape_baseline, reshape_keys
            )

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
        raise NotImplementedError

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

    def _setup_whittaker(self, y, lam=1, diff_order=2, weights=None, copy_weights=False,
                         allow_lower=True, reverse_diags=None):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
        lam : float, optional
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on the penalized least square method
            and the differential order. Default is 1.
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
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite,
            ensure_1d=False
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]
        weight_array = weight_array.ravel()
        if self.whittaker_system is not None:
            self.whittaker_system.reset_diagonals(lam, diff_order, allow_lower, reverse_diags)
        else:
            self.whittaker_system = PenalizedSystem2D(
                self._len, lam, diff_order
            )

        return y.ravel(), weight_array

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=False,
                          calc_pinv=False, copy_weights=False, max_cross=None):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        poly_order : int or Container[int, int], optional
            The polynomial orders for x and z. Default is 2.
        calc_vander : bool, optional
            If True, will calculate and the Vandermonde matrix. Default is False.
        calc_pinv : bool, optional
            If True, and if `return_vander` is True, will calculate and return the
            pseudo-inverse of the Vandermonde matrix. Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        max_cross: int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

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
        Implementation note: the polynomial coefficients, `c`, from solving 2D polynomials
        using ``Ac=b`` where `A` is the flattened Vandermonde and `b` is the flattened data
        corresponds to the matrix below:

            np.array([
                [x^0*z^0, x^0*z^1, ..., x^0*z^n],
                [x^1*z^0, x^1*z^1, ..., x^1*z^n],
                [...],
                [x^m*z^0, x^m*z^1, ..., x^m*z^n]
            ]).flatten()

        """
        weight_array = _check_optional_array(
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite,
            ensure_1d=False
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]
        weight_array = weight_array.ravel()
        poly_orders = _check_scalar(poly_order, 2, True)[0]
        if max_cross is not None:
            max_cross = _check_scalar_variable(
                max_cross, allow_zero=True, variable_name='max_cross'
            )
        if calc_vander:
            if (
                self.vandermonde is None or self._max_cross != max_cross
                or np.any(self.poly_order != poly_order)
            ):
                mapped_x = np.polynomial.polyutils.mapdomain(
                    self.x, self.x_domain, np.array([-1., 1.])
                )
                mapped_z = np.polynomial.polyutils.mapdomain(
                    self.z, self.z_domain, np.array([-1., 1.])
                )
                # rearrange the vandermonde such that it matches the typical A c = b where b
                # is the flattened version of y and c are the coefficients
                self.vandermonde = np.polynomial.polynomial.polyvander2d(
                    *np.meshgrid(mapped_x, mapped_z), [poly_orders[0], poly_orders[1]]
                ).reshape((-1, (poly_orders[0] + 1) * (poly_orders[1] + 1)))

                if max_cross is not None:
                    # lists out (z_0, x_0), (z_1, x_0), etc
                    for idx, val in enumerate(
                        itertools.product(range(poly_orders[0] + 1), range(poly_orders[1] + 1))
                    ):
                        # 0 designates pure z or x terms
                        if 0 not in val and any(v > max_cross for v in val):
                            self.vandermonde[:, idx] = 0

        self.poly_order = poly_orders
        self._max_cross = max_cross
        y = y.ravel()
        if not calc_pinv:
            return y, weight_array
        elif not calc_vander:
            raise ValueError('if calc_pinv is True, then calc_vander must also be True')

        if weights is None:
            pseudo_inverse = np.linalg.pinv(self.vandermonde)
        else:
            pseudo_inverse = np.linalg.pinv(np.sqrt(weight_array)[:, None] * self.vandermonde)

        return y, weight_array, pseudo_inverse

    def _setup_spline(self, y, weights=None, spline_degree=3, num_knots=10,
                      penalized=True, diff_order=3, lam=1, make_basis=True, allow_lower=True,
                      reverse_diags=False, copy_weights=False):
        """
        Sets the starting parameters for doing spline fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`._register`.
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
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite,
            ensure_1d=False
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]
        diff_order = _check_scalar(diff_order, 2, True)[0]
        if make_basis:
            if (diff_order > 4).any():
                warnings.warn(
                    ('differential orders greater than 4 can have numerical issues;'
                     ' consider using a differential order of 2 or 3 instead'),
                    ParameterWarning, stacklevel=2
                )

            if self.pspline is None or not self.pspline.same_basis(num_knots, spline_degree):
                self.pspline = PSpline2D(
                    self.x, self.z, num_knots, spline_degree, self._check_finite, lam, diff_order
                )
            else:
                self.pspline.reset_penalty_diagonals(lam, diff_order)

        return y, weight_array

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
