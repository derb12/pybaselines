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

from contextlib import contextmanager
from functools import partial, wraps
from inspect import signature
import warnings

import numpy as np

from ._banded_utils import PenalizedSystem
from ._spline_utils import PSpline
from ._validation import (
    _check_array, _check_half_window, _check_optional_array, _check_scalar_variable,
    _check_sized_array, _yx_arrays
)
from .utils import (
    ParameterWarning, _determine_sorts, _inverted_sort, _sort_array, optimize_window, pad_edges
)


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
        if no penalized spline setup has been performed (typically done in
        :meth:`~_Algorithm._setup_spline`).
    vandermonde : numpy.ndarray or None
        The Vandermonde matrix for solving polynomial equations. Is None if no polynomial
        setup has been performed (typically done in :meth:`~_Algorithm._setup_polynomial`).
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
            If False (default), will sort the input `x_data` values. Otherwise, the
            input is assumed to be sorted. Note that some functions may raise an error
            if `x_data` is not sorted.
        output_dtype : type or numpy.dtype, optional
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
            self._sort_order, self._inverted_order = _determine_sorts(self.x)
            if self._sort_order is not None:
                self.x = self.x[self._sort_order]

        self.whittaker_system = None
        self.vandermonde = None
        self.poly_order = -1
        self.pspline = None
        self._check_finite = check_finite
        self._dtype = output_dtype
        self.pentapy_solver = 2

    @property
    def pentapy_solver(self):
        """
        The integer or string designating which solver to use if using pentapy.

        See :func:`pentapy.solve` for available options, although `1` or `2` are the
        most relevant options. Default is 2.

        .. versionadded:: 1.1.0

        """
        return self._pentapy_solver

    @pentapy_solver.setter
    def pentapy_solver(self, value):
        """
        Sets the solver for pentapy.

        Parameters
        ----------
        value : int or str
            The designated solver to use when using pentapy. See :func:`pentapy.core.solve`
            for available options.

        """
        if self.whittaker_system is not None:
            self.whittaker_system.pentapy_solver = value
        self._pentapy_solver = value

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

        baseline = baseline.astype(dtype, copy=False)

        return baseline, params

    @classmethod
    def _register(cls, func=None, *, sort_keys=(), dtype=None, order=None, ensure_1d=True,
                  skip_sorting=False):
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
        ensure_1d : bool, optional
            If True (default), will raise an error if the shape of `array` is not a one dimensional
            array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
        skip_sorting : bool, optional
            If True, will skip sorting the inputs and outputs, which is useful for algorithms that
            use other algorithms so that sorting is already internally done. Default is False.

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
                ensure_1d=ensure_1d, skip_sorting=skip_sorting
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
                    ensure_1d=ensure_1d
                )
                self._len = y.shape[-1]
            else:
                reset_x = True
                if data is not None:
                    input_y = True
                    y = _check_sized_array(
                        data, self._len, check_finite=self._check_finite, dtype=dtype, order=order,
                        ensure_1d=ensure_1d, name='data'
                    )
                else:
                    y = data
                    input_y = False
                # update self.x just to ensure dtype and order are correct
                x_dtype = self.x.dtype
                self.x = _check_array(
                    self.x, dtype=dtype, order=order, check_finite=False, ensure_1d=False
                )

            if input_y and not skip_sorting:
                y = _sort_array(y, sort_order=self._sort_order)

            if input_y and self._dtype is None:
                output_dtype = y.dtype
            else:
                output_dtype = self._dtype

            baseline, params = func(self, y, *args, **kwargs)
            if reset_x:
                self.x = np.array(self.x, dtype=x_dtype, copy=False)

            return self._return_results(baseline, params, output_dtype, sort_keys, skip_sorting)

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

    def _setup_whittaker(self, y, lam=1, diff_order=2, weights=None, copy_weights=False,
                         allow_lower=True, reverse_diags=None):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
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
            self._len, weights, copy_input=copy_weights, check_finite=self._check_finite
        )
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]

        if self.whittaker_system is not None:
            self.whittaker_system.reset_diagonals(lam, diff_order, allow_lower, reverse_diags)
        else:
            self.whittaker_system = PenalizedSystem(
                self._len, lam, diff_order, allow_lower, reverse_diags,
                pentapy_solver=self.pentapy_solver
            )

        return y, weight_array

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=False,
                          calc_pinv=False, copy_weights=False):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
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
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]
        poly_order = _check_scalar_variable(
            poly_order, allow_zero=True, variable_name='polynomial order', dtype=int
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

    def _setup_spline(self, y, weights=None, spline_degree=3, num_knots=10,
                      penalized=True, diff_order=3, lam=1, make_basis=True, allow_lower=True,
                      reverse_diags=False, copy_weights=False):
        """
        Sets the starting parameters for doing spline fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
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
        if self._sort_order is not None and weights is not None:
            weight_array = weight_array[self._sort_order]

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

    def _setup_morphology(self, y, half_window=None, **window_kwargs):
        """
        Sets the starting parameters for morphology-based methods.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
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
            output_half_window = optimize_window(y, **window_kwargs)

        return y, output_half_window

    def _setup_smooth(self, y, half_window=0, allow_zero=True, **pad_kwargs):
        """
        Sets the starting parameters for doing smoothing-based algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
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

    def _setup_classification(self, y, weights=None):
        """
        Sets the starting parameters for doing classification algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~_Algorithm._register`.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for the data, with boolean dtype.

        """
        weight_array = _check_optional_array(
            self._len, weights, dtype=bool, check_finite=self._check_finite
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
            array by :meth:`~_Algorithm._register`.
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

        Warns
        -----
        DeprecationWarning
            Passed if `kwargs` is not empty.

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
            array by :meth:`~_Algorithm._register`.

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
