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
from inspect import signature
import warnings

import numpy as np

from ._banded_utils import PenalizedSystem
from ._spline_utils import PSpline, SplineBasis
from ._validation import (
    _check_array, _check_half_window, _check_optional_array, _check_scalar_variable,
    _check_sized_array, _yx_arrays
)
from .utils import (
    ParameterWarning, SortingWarning, _determine_sorts, _inverted_sort, _sort_array,
    optimize_window, pad_edges
)


class _Algorithm:
    """
    A base class for all algorithm types.

    Contains setup methods for all algorithm types to make more complex algorithms
    easier to set up.

    Attributes
    ----------
    x : numpy.ndarray or None
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same length as the input `data` and has min and max
        values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x_data` is None during initialization, then
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
            If True (default), will raise an error if any values in input data are not finite.
            Setting to False will skip the check. Note that errors may occur if
            `check_finite` is False and the input data contains non-finite values.
        assume_sorted : bool, optional
            If False (default), will sort the input `x_data` values. Otherwise, the input
            is assumed to be sorted, although it will still be checked to be in ascending order.
            Note that some methods will raise an error if `x_data` values are not unique.
        output_dtype : type or numpy.dtype, optional
            The dtype to cast the output array. Default is None, which uses the typing
            of the input data.

        """
        no_x = x_data is None
        if no_x:
            self.x = None
            self.x_domain = np.array([-1., 1.])
            self._size = None
        else:
            # TODO allow int or float32 x-values later; have to address in individual methods
            self.x = _check_array(x_data, dtype=float, check_finite=check_finite)
            self._size = len(self.x)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)
            if assume_sorted and np.any(self.x[1:] < self.x[:-1]):
                warnings.warn(
                    ('x-values must be strictly increasing for many methods, so setting '
                     'assume_sorted to True'), SortingWarning, stacklevel=2
                )
                assume_sorted = False

        if no_x or assume_sorted:
            self._sort_order = None
            self._inverted_order = None
        else:
            self._sort_order, self._inverted_order = _determine_sorts(self.x)
            if self._sort_order is not None:
                self.x = self.x[self._sort_order]

        self.banded_solver = 2
        self._polynomial = None
        self._spline_basis = None
        self._check_finite = check_finite
        self._dtype = output_dtype
        self._validated_x = no_x

    @property
    def _size(self):
        """The length of the Algorithm object."""
        return self.__size

    @_size.setter
    def _size(self, value):
        """Sets the length and shape of the _Algorithm object.

        Parameters
        ----------
        value : int or None
            The length of the dataset.

        Notes
        -----
        Follows NumPy naming conventions where _Algorithm._size is the total number of items,
        and _Algorithm._shape is the number of items in each dimension.

        """
        if value is None:
            self.__size = None
            self._shape = (None,)
        else:
            self.__size = value
            self._shape = (value,)

    @property
    def banded_solver(self):
        """
        Designates the solver to prefer using for solving banded linear systems.

        .. versionadded:: 1.2.0

        An integer between 1 and 4 designating the solver to prefer for solving banded
        linear systems. Setting to 1 or 2 will use the ``PTRANS-I``
        and ``PTRANS-II`` solvers, respectively, from :func:`pentapy.solve` if
        ``pentapy`` is installed and the linear system is pentadiagonal. Otherwise,
        it will use :func:`scipy.linalg.solveh_banded` if the system is symmetric,
        else :func:`scipy.linalg.solve_banded`. Setting ``banded_solver`` to 3
        will only use the SciPy solvers following the same logic, and 4 will
        force usage of :func:`scipy.linalg.solve_banded`. Default is 2.

        This typically does not need to be modified since all solvers have relatively
        the same numerical stability and is mostly for internal testing.

        """
        return self._banded_solver

    @banded_solver.setter
    def banded_solver(self, solver):
        """
        Sets the solver for banded systems and the solver for the optional dependency pentapy.

        Parameters
        ----------
        solver : {1, 2, 3, 4}
            An integer designating the solver. Setting to 1 or 2 will use the ``PTRANS-I``
            and ``PTRANS-II`` solvers, respectively, from :func:`pentapy.solve` if
            ``pentapy`` is installed and the linear system is pentadiagonal. Otherwise,
            it will use :func:`scipy.linalg.solveh_banded` if the system is symmetric,
            else :func:`scipy.linalg.solve_banded`. Setting ``banded_solver`` to 3
            will only use the SciPy solvers following the same logic, and 4 will
            force usage of :func:`scipy.linalg.solve_banded`.

        Raises
        ------
        ValueError
            Raised if `solver` is not an integer between 1 and 4.

        """
        if isinstance(solver, bool) or solver not in {1, 2, 3, 4}:
            # catch True since it can be interpreted as in {1, 2, 3, 4}; would likely
            # not cause issues downsteam, but just eliminate that possibility
            raise ValueError('banded_solver must be an integer with a value in (1, 2, 3, 4)')
        self._banded_solver = solver
        if solver < 3:
            self._pentapy_solver = solver
        else:
            self._pentapy_solver = 1  # default value

    @property
    def pentapy_solver(self):
        """
        The solver if using ``pentapy`` to solve banded equations.

        .. deprecated:: 1.2
            The `pentapy_solver` property is deprecated and will be removed in
            version 1.4. Use :attr:`~.banded_solver` instead.

        """
        warnings.warn(
            ('The `pentapy_solver` attribute is deprecated and will be removed in '
             'version 1.4; use the `banded_solver` attribute instead'),
            DeprecationWarning, stacklevel=2
        )
        return self._pentapy_solver

    @pentapy_solver.setter
    def pentapy_solver(self, value):
        warnings.warn(
            ('Setting the `pentapy_solver` attribute is deprecated and will be removed in '
             'version 1.4, set the `banded_solver` attribute instead'),
            DeprecationWarning, stacklevel=2
        )
        self.banded_solver = value

    def _return_results(self, baseline, params, dtype, sort_keys=(), skip_sorting=False):
        """
        Re-orders the input baseline and parameters based on the x ordering.

        If `self._sort_order` is None, then no reordering is performed.

        Parameters
        ----------
        baseline : numpy.ndarray, shape (N,)
            The baseline output by the baseline function.
        params : dict
            The parameter dictionary output by the baseline function.
        dtype : type or numpy.dtype, optional
            The desired output dtype for the baseline.
        sort_keys : Iterable, optional
            An iterable of keys corresponding to the values in `params` that need
            re-ordering. Default is ().
        skip_sorting : bool, optional
            If True, will skip sorting the output baseline. The keys in `sort_keys` will
            still be sorted. Default is False.

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
                    # assumes params all all just one dimensional arrays
                    params[key] = params[key][self._inverted_order]
            if not skip_sorting:
                baseline = _sort_array(baseline, sort_order=self._inverted_order)

        baseline = np.asarray(baseline, dtype=dtype)

        return baseline, params

    @classmethod
    def _register(cls, func=None, *, sort_keys=(), ensure_1d=True, skip_sorting=False,
                  require_unique_x=False):
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
        ensure_1d : bool, optional
            If True (default), will raise an error if the shape of `array` is not a one dimensional
            array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
        skip_sorting : bool, optional
            If True, will skip sorting the inputs and outputs, which is useful for algorithms that
            use other algorithms so that sorting is already internally done. Default is False.
        require_unique_x : bool, optional
            If True, will check ``self.x`` to ensure all values are unique and will raise an error
            if non-unique values are present. Default is False, which skips the check.

        Returns
        -------
        numpy.ndarray
            The calculated baseline.
        dict
            A dictionary of parameters output by the baseline function.

        """
        if func is None:
            return partial(
                cls._register, sort_keys=sort_keys, ensure_1d=ensure_1d, skip_sorting=skip_sorting,
                require_unique_x=require_unique_x
            )

        @wraps(func)
        def inner(self, data=None, *args, **kwargs):
            if self.x is None:
                if data is None:
                    raise TypeError('"data" and "x_data" cannot both be None')
                input_y = True
                y, self.x = _yx_arrays(
                    data, check_finite=self._check_finite, ensure_1d=ensure_1d
                )
                self._size = y.shape[-1]
            else:
                if require_unique_x and not self._validated_x:
                    if np.any(self.x[1:] == self.x[:-1]):
                        raise ValueError('x-values must be unique for the selected method')
                    else:
                        self._validated_x = True
                if data is not None:
                    input_y = True
                    y = _check_sized_array(
                        data, self._size, check_finite=self._check_finite, ensure_1d=ensure_1d,
                        name='data'
                    )
                else:
                    y = data
                    input_y = False

            if input_y and not skip_sorting:
                y = _sort_array(y, sort_order=self._sort_order)

            if input_y and self._dtype is None:
                output_dtype = y.dtype
            else:
                output_dtype = self._dtype
            # TODO allow int or float32 y-values later?; have to address in individual methods;
            # often x and y need to have the same dtype too
            y = np.asarray(y, dtype=float)

            baseline, params = func(self, y, *args, **kwargs)

            return self._return_results(baseline, params, output_dtype, sort_keys, skip_sorting)

        return inner

    def _override_x(self, new_x, new_sort_order=None):
        """
        Creates a new fitting object for the given x-values.

        Useful when fitting extensions of the x attribute.

        Parameters
        ----------
        new_x : numpy.ndarray, shape (M,)
            The x values to temporarily use.
        new_sort_order : numpy.ndarray, shape (M,), optional
            The sort order for the new x values. Default is None, which will not sort.

        Returns
        -------
        pybaselines._algorithm_setup._Algorithm
            The _Algorithm object with the new x attribute.

        """
        new_object = type(self)(
            x_data=new_x, check_finite=self._check_finite, assume_sorted=True,
            output_dtype=self._dtype
        )
        new_object.banded_solver = self.banded_solver
        new_object._sort_order = new_sort_order
        if new_sort_order is not None:
            new_object._inverted_order = _inverted_sort(new_sort_order)

        return new_object

    def _setup_whittaker(self, y, lam=1, diff_order=2, weights=None, copy_weights=False,
                         allow_lower=True, reverse_diags=None):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
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
        whittaker_system : PenalizedSystem
            The PenalizedSystem for solving the given penalized least squared system.

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
            self._size, weights, copy_input=copy_weights, check_finite=self._check_finite
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]

        allow_lower = allow_lower and self.banded_solver < 4
        allow_pentapy = self.banded_solver < 3

        whittaker_system = PenalizedSystem(
            self._size, lam, diff_order, allow_lower, reverse_diags, allow_pentapy=allow_pentapy,
            pentapy_solver=self._pentapy_solver
        )

        return y, weight_array, whittaker_system

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=False,
                          calc_pinv=False, copy_weights=False):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
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
            self._size, weights, copy_input=copy_weights, check_finite=self._check_finite
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]
        poly_order = _check_scalar_variable(
            poly_order, allow_zero=True, variable_name='polynomial order', dtype=int
        )

        if calc_vander:
            if self._polynomial is None:
                self._polynomial = _PolyHelper(self.x, self.x_domain, poly_order)
            else:
                self._polynomial.recalc_vandermonde(self.x, self.x_domain, poly_order)

        if not calc_pinv:
            return y, weight_array
        elif not calc_vander:
            raise ValueError('if calc_pinv is True, then calc_vander must also be True')

        if weights is None:
            pseudo_inverse = self._polynomial.pseudo_inverse
        else:
            pseudo_inverse = np.linalg.pinv(
                np.sqrt(weight_array)[:, None] * self._polynomial.vandermonde
            )

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
            array by :meth:`~._Algorithm._register`.
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
        pspline : PSpline
            The PSpline object for solving the given penalized least squared system. Only
            returned if `make_basis` is True.

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
            self._size, weights, dtype=float, order='C', copy_input=copy_weights,
            check_finite=self._check_finite
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]

        if not make_basis:
            return y, weight_array

        if diff_order > 4:
            warnings.warn(
                ('differential orders greater than 4 can have numerical issues;'
                 ' consider using a differential order of 2 or 3 instead'),
                ParameterWarning, stacklevel=2
            )

        if (
            self._spline_basis is None
            or not self._spline_basis.same_basis(num_knots, spline_degree)
        ):
            self._spline_basis = SplineBasis(self.x, num_knots, spline_degree)

        allow_lower = allow_lower and self.banded_solver < 4
        pspline = PSpline(
            self._spline_basis, lam, diff_order, allow_lower, reverse_diags
        )

        return y, weight_array, pspline

    def _setup_morphology(self, y, half_window=None, window_kwargs=None, **kwargs):
        """
        Sets the starting parameters for morphology-based methods.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using pybaselines.morphological.optimize_window.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.optimize_window` for
            estimating the half window if `half_window` is None. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `window_kwargs`.

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
            output_half_window = _check_half_window(half_window, allow_zero=False)
        else:
            window_kwargs = window_kwargs if window_kwargs is not None else {}
            if kwargs:
                warnings.warn(
                    ('Passing additional keyword arguments for optimizing the half_window is '
                     'deprecated and will be removed in version 1.4.0. Place all keyword '
                     'arguments into the "window_kwargs" dictionary instead'),
                    DeprecationWarning, stacklevel=2
                )

            output_half_window = optimize_window(y, **window_kwargs, **kwargs)

        return y, output_half_window

    def _setup_smooth(self, y, half_window=None, pad_type='half', window_multiplier=1,
                      pad_kwargs=None, **kwargs):
        """
        Sets the starting parameters for doing smoothing-based algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
        half_window : int, optional
            The half-window used for the smoothing functions. Used
            to pad the left and right edges of the data to reduce edge
            effects. Default is is None, which sets the half window as the output of
            :func:`pybaselines.utils.optimize_window` multiplied by `window_multiplier`.
        pad_type : {'half', 'full', None}
            If True (default), will pad the input `y` with `half_window` on each side
            before returning. If False, will return the unmodified `y`.
        window_multiplier : int or float, optional
            The multiplier by which the output of :func:`pybaselines.utils.optimize_window`
            will be multiplied if `half_window` is None.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `pad_kwargs`.

        Returns
        -------
        output : numpy.ndarray
            The padded array of data with shape (``N + 2 * output_half_window``,) if `pad_data`
            is True,
            otherwise the non-padded data with shape (``N``,).
        output_half_window : int
            The final half-window used for potentially padding the data.

        """
        if half_window is None:
            output_half_window = max(1, int(window_multiplier * optimize_window(y)))
        else:
            output_half_window = _check_half_window(half_window, allow_zero=False)

        self._deprecate_pad_kwargs(**kwargs)
        if pad_type is None:
            output = y
        else:
            if pad_type == 'half':
                padding_window = output_half_window
            else:
                padding_window = 2 * output_half_window + 1
            pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
            output = pad_edges(y, padding_window, **pad_kwargs, **kwargs)

        return output, output_half_window

    def _deprecate_pad_kwargs(self, **kwargs):
        """Ensures deprecation of passing kwargs for padding."""
        if kwargs:
            warnings.warn(
                ('Passing additional keyword arguments for padding is '
                    'deprecated and will be removed in version 1.4.0. Place all keyword '
                    'arguments into the "pad_kwargs" dictionary instead'),
                DeprecationWarning, stacklevel=2
            )

    def _setup_classification(self, y, weights=None, **kwargs):
        """
        Sets the starting parameters for doing classification algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        **kwargs
            Any keyword arguments passed to the method. Will warn if any.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for the data, with boolean dtype.

        """
        self._deprecate_pad_kwargs(**kwargs)
        weight_array = _check_optional_array(
            self._size, weights, dtype=bool, check_finite=self._check_finite
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]

        return y, weight_array

    def _get_function(self, method, modules):
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
        class_object : pybaselines._algorithm_setup._Algorithm
            The `_Algorithm` object which will be used for fitting.

        Raises
        ------
        AttributeError
            Raised if no matching function is found within the modules.

        """
        function_string = method.lower()
        for module in modules:
            if hasattr(module, function_string):
                func_module = module.__name__.split('.')[-1]
                # if self is a Baseline class, can just use its method
                if hasattr(self, function_string):
                    func = getattr(self, function_string)
                    class_object = self
                else:
                    # have to reset x ordering so that all outputs and parameters are
                    # correctly sorted
                    if self._sort_order is not None:
                        x = self.x[self._inverted_order]
                        assume_sorted = False
                    else:
                        x = self.x
                        assume_sorted = True
                    class_object = getattr(module, '_' + func_module.capitalize())(
                        x, check_finite=self._check_finite, assume_sorted=assume_sorted,
                        output_dtype=self._dtype
                    )
                    class_object.banded_solver = self.banded_solver
                    func = getattr(class_object, function_string)
                break
        else:  # in case no break
            mod_names = [module.__name__ for module in modules]
            raise AttributeError((
                f'unknown method "{method}" or method is not within the allowed '
                f'modules: {mod_names}'
            ))

        return func, func_module, class_object

    def _setup_optimizer(self, y, method, modules, method_kwargs=None, copy_kwargs=True):
        """
        Sets the starting parameters for doing optimizer algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.
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

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        baseline_func : Callable
            The function for fitting the baseline.
        func_module : str
            The string name of the module that contained `fit_func`.
        method_kws : dict
            A dictionary of keyword arguments to pass to `fit_func`.
        class_object : pybaselines._algorithm_setup._Algorithm
            The `_Algorithm` object which will be used for fitting.

        """
        baseline_func, func_module, class_object = self._get_function(method, modules)
        if method_kwargs is None:
            method_kws = {}
        elif copy_kwargs:
            method_kws = method_kwargs.copy()
        else:
            method_kws = method_kwargs

        if 'x_data' in method_kws:
            raise KeyError('"x_data" should not be within the method keyword arguments')

        return y, baseline_func, func_module, method_kws, class_object

    def _setup_misc(self, y):
        """
        Sets the starting parameters for doing miscellaneous algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._register`.

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


def _class_wrapper(klass):
    """
    Wraps a function to call the corresponding class method instead.

    Parameters
    ----------
    klass : _Algorithm
        The class being wrapped.

    """
    def outer(func):
        func_signature = signature(func)
        method = func.__name__

        @wraps(func)
        def inner(*args, **kwargs):
            total_inputs = func_signature.bind(*args, **kwargs)
            x = total_inputs.arguments.pop('x_data', None)
            return getattr(klass(x_data=x), method)(*total_inputs.args, **total_inputs.kwargs)
        return inner

    return outer


class _PolyHelper:
    """
    An object to help with solving polynomials.

    Allows only recalculating the Vandermonde and pseudo-inverse matrices when necessary.

    Attributes
    ----------
    poly_order : int
        The last polynomial order used to calculate the Vadermonde matrix.
    pseudo_inverse : numpy.ndarray or None
        The pseudo-inverse of the current Vandermonde matrix.
    vandermonde : numpy.ndarray
        The Vandermonde matrix for solving polynomial equations.

    """

    def __init__(self, x, x_domain, poly_order):
        """
        Initializes the object and calculates the Vandermonde matrix.

        Parameters
        ----------
        x : numpy.ndarray
            The x-values for the polynomial.
        x_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `x`.
        poly_order : int
            The polynomial order.

        """
        poly_order = _check_scalar_variable(
            poly_order, allow_zero=True, variable_name='polynomial order', dtype=int
        )
        self.poly_order = -1
        self.vandermonde = None
        self._pseudo_inverse = None
        self.pinv_stale = True

        self.recalc_vandermonde(x, x_domain, poly_order)

    def recalc_vandermonde(self, x, x_domain, poly_order):
        """
        Recalculates the Vandermonde matrix for the polynomial only if necessary.

        Also flags whether the pseudo-inverse needs to be recalculated.

        Parameters
        ----------
        x : numpy.ndarray
            The x-values for the polynomial.
        x_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `x`.
        poly_order : int
            The polynomial order.

        """
        if self.vandermonde is None or poly_order > self.poly_order:
            mapped_x = np.polynomial.polyutils.mapdomain(
                x, x_domain, np.array([-1., 1.])
            )
            self.vandermonde = np.polynomial.polynomial.polyvander(mapped_x, poly_order)
            self.pinv_stale = True
        elif poly_order < self.poly_order:
            self.vandermonde = self.vandermonde[:, :poly_order + 1]
            self.pinv_stale = True

        self.poly_order = poly_order

    @property
    def pseudo_inverse(self):
        """
        The pseudo-inverse of the Vandermonde.

        Only recalculates the pseudo-inverse if the Vandermonde has been updated.

        """
        if self.pinv_stale or self._pseudo_inverse is None:
            self._pseudo_inverse = np.linalg.pinv(self.vandermonde)
            self.pinv_stale = False
        return self._pseudo_inverse
