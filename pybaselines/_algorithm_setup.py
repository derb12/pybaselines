# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on March 31, 2021
@author: Donald Erb

"""

from functools import partial, wraps
from inspect import signature
import warnings

import numpy as np

from ._banded_utils import PenalizedSystem
from ._nd.optimizers import _OptimizerHelper
from ._spline_utils import PSpline, SplineBasis
from ._validation import (
    _check_array, _check_half_window, _check_optional_array, _check_scalar_variable,
    _check_sized_array, _yx_arrays
)
from .utils import (
    ParameterWarning, SortingWarning, _determine_sorts, _inverted_sort, _sort_array,
    estimate_window, pad_edges
)
from .results import PSplineResult, WhittakerResult


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
                 output_dtype='deprecated', mask=None, strict_mask=True):
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
            The dtype to cast the output array. Default behavior keeps the dtype as float.

            .. deprecated:: 1.3
                `output_dtype` is deprecated and will be removed in version 1.5. Use
                :func:`numpy.astype` on the output baseline instead.

        mask : array-like, shape (N,), optional
            A Boolean array, in which all indices that are `True` denote indices that should
            be ignored during baseline correction. If None (default), denotes all values should
            be included.
        strict_mask : bool, optional
            If True (default) and `mask` is not None, calling any method that does not support
            masking will raise an exception. Setting `strict_mask` to False will instead perform
            linear interpolation following `mask` before calling those methods.

        """
        no_x = x_data is None
        if no_x:
            self.x = None
            self.x_domain = np.array([-1., 1.])
            self._size = None
        else:
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

        if output_dtype != 'deprecated':
            warnings.warn(
                'specifying "output_dtype" is deprecated and will be removed in version 1.5. '
                'Use np.astype on the output baseline instead.', DeprecationWarning, stacklevel=3
            )

        self.banded_solver = 2
        self._polynomial = None
        self._spline_basis = None
        self._check_finite = check_finite
        self._dtype = output_dtype
        self._validated_x = no_x
        self._strict_mask = strict_mask
        self.mask = mask

    @property
    def mask(self):
        """
        Designates which points should be omitted during baseline correction.

        .. versionadded:: 1.3.0

        A Boolean array, in which all indices that are `True` denote indices that should
        be ignored during baseline correction. If None, denotes all values should be included.

        Note that not all methods support masking, so by default, a non-None `mask` will raise
        an exception upon calling those methods. This behavior can be changed by setting
        `strict_mask` to `False` during initialization.

        """
        return self._mask

    @mask.setter
    def mask(self, values):
        """
        Sets the baseline fitting mask.

        Parameters
        ----------
        values : array-like, shape (N,), optional
            A Boolean array, in which all indices that are `True` denote indices that should
            be omitted during baseline correction. If None, denotes all values should
            be included.

        """
        if values is None:
            self._mask = None
        else:
            if self._size is None:
                input_mask = _check_array(values, dtype=bool)
                self._size = len(input_mask)
                self.x = np.linspace(-1., 1., self._size)
            else:
                input_mask = _check_sized_array(values, self._size, name='mask', dtype=bool)
            self._mask = _sort_array(input_mask, self._sort_order)
            # TODO should ensure some lower bound for input_mask.sum() so that there
            # are actually enough points to do calculations; maybe 5-10% of the points?

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

        An integer designating the solver. Setting to 1 or 2 will use the ``PTRANS-I``
        and ``PTRANS-II`` solvers, respectively, from [1]_ if ``numba`` is installed
        and the linear system is pentadiagonal. Otherwise, it will use
        :func:`scipy.linalg.solveh_banded` if the system is symmetric, else
        :func:`scipy.linalg.solve_banded`. Setting ``banded_solver`` to 3 will only
        use the SciPy solvers following the same logic, and 4 will force usage of
        :func:`scipy.linalg.solve_banded`. Default is 2

        PTRANS-I and PTRANS-II are the fastest, and solve_banded is the slowest. In terms
        of numerical stability, solveh_banded is the least stable, PTRANS-I and PTRANS-II
        are slightly more stable (LU factorization without pivoting), and solve_banded (LU
        factorization with partial pivoting) is the most stable. In practice, however,
        instability rarely occurs during baseline correction and should be avoided through
        other means (eg. switching from Whittaker-smoothing methods to penalized spline
        methods in order to lower the required `lam` parameter).

        References
        ----------
        .. [1] Askar, S., et al. On Solving Pentadiagonal Linear Systems via
            Transformations. Mathematical Problems in Engineering, 2015, 232456.

        """
        return self._banded_solver

    @banded_solver.setter
    def banded_solver(self, solver):
        """
        Sets the solver to use for banded linear systems.

        Parameters
        ----------
        solver : {1, 2, 3, 4}
            An integer designating the solver. Setting to 1 or 2 will use the ``PTRANS-I``
            and ``PTRANS-II`` solvers, respectively, from [1]_ if ``numba`` is installed
            and the linear system is pentadiagonal. Otherwise, it will use
            :func:`scipy.linalg.solveh_banded` if the system is symmetric, else
            :func:`scipy.linalg.solve_banded`. Setting ``banded_solver`` to 3 will only
            use the SciPy solvers following the same logic, and 4 will force usage of
            :func:`scipy.linalg.solve_banded`.

        Raises
        ------
        ValueError
            Raised if `solver` is not an integer between 1 and 4.

        References
        ----------
        .. [1] Askar, S., et al. On Solving Pentadiagonal Linear Systems via
            Transformations. Mathematical Problems in Engineering, 2015, 232456.

        """
        if isinstance(solver, bool) or solver not in {1, 2, 3, 4}:
            # catch True since it can be interpreted as in {1, 2, 3, 4}; would likely
            # not cause issues downstream, but just eliminate that possibility
            raise ValueError('banded_solver must be an integer with a value in (1, 2, 3, 4)')
        self._banded_solver = solver
        if solver < 3:
            self._penta_solver = solver
        else:
            self._penta_solver = 2  # default value

    @property
    def pentapy_solver(self):
        """
        The solver if using the dedicated pentadiagonal solvers to solve banded equations.

        .. deprecated:: 1.2
            The `pentapy_solver` property is deprecated and will be removed in
            version 1.4. Use :attr:`~.banded_solver` instead.

        """
        warnings.warn(
            ('The `pentapy_solver` attribute is deprecated and will be removed in '
             'version 1.4; use the `banded_solver` attribute instead'),
            DeprecationWarning, stacklevel=2
        )
        return self._penta_solver

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

        if dtype != 'deprecated':
            baseline = np.asarray(baseline, dtype=dtype)

        return baseline, params

    @classmethod
    def _handle_io(cls, func=None, *, sort_keys=(), ensure_dims=True, skip_sorting=False,
                   require_unique=False, reshape_keys=None, mask_support=-1):
        """
        Wraps a baseline method to validate inputs and correct outputs.

        The input data is converted to a numpy array, validated to ensure the length is
        consistent, and ordered to match the input x ordering. The outputs are corrected
        to ensure proper inverted sort ordering and dtype.

        Parameters
        ----------
        func : Callable, optional
            The method that is being decorated. Default is None, which returns a partial function.
        sort_keys : tuple, optional
            The keys within the output parameter dictionary that will need sorting to match the
            sort order of ``self.x``. Default is ().
        ensure_dims : bool, optional
            If True (default), will raise an error if the shape of `array` is not a one dimensional
            array with shape (N,) or a two dimensional array with shape (N, 1) or (1, N).
        skip_sorting : bool, optional
            If True, will skip sorting the inputs and outputs, which is useful for algorithms that
            use other algorithms so that sorting is already internally done. Default is False.
        require_unique : bool, optional
            If True, will check `self.x` to ensure all values are unique and will raise an error
            if non-unique values are present. Default is False, which skips the check.
        reshape_keys : None, optional
            Not used within this method, simply added to have the same call signature
            as `_Algorithm2D._handle_io`.
        mask_support : bool, optional
            An integer designating how the wrapped function handles masking. The default value,
            `-1`, means that masking is not supported and will raise an error if `self.mask` is
            not None. A value of `1` means that masking is supported through weighted
            interpolation and will replace both input data and weights with zeros following the
            mask. A value of `0` means to ignore the mask, for use within some optimizers.
            `mask_support` values not equal to 0 or 1 will replace values within the input data
            with linear interpolation so that no issues with NaN values occur.

        Returns
        -------
        numpy.ndarray
            The calculated baseline.
        dict
            A dictionary of parameters output by the baseline function.

        """
        if func is None:
            return partial(
                cls._handle_io, sort_keys=sort_keys, ensure_dims=ensure_dims,
                skip_sorting=skip_sorting, require_unique=require_unique,
                mask_support=mask_support
            )

        @wraps(func)
        def inner(self, data=None, *args, **kwargs):
            if self.x is None:  # also means self.mask is None
                if data is None:
                    raise TypeError('"data" and "x_data" cannot both be None')
                input_y = True
                y, self.x = _yx_arrays(
                    data, check_finite=self._check_finite, ensure_1d=ensure_dims, dtype=float
                )
                self._size = y.shape[-1]
            else:
                if require_unique and not self._validated_x:
                    if np.any(self.x[1:] == self.x[:-1]):
                        raise ValueError('x-values must be unique for the selected method')
                    else:
                        self._validated_x = True
                if data is not None:
                    input_y = True
                    if self.mask is None:
                        y = _check_sized_array(
                            data, self._size, check_finite=self._check_finite,
                            ensure_1d=ensure_dims, name='data', dtype=float
                        )
                    else:
                        y = _check_sized_array(
                            data, self._size, check_finite=False, ensure_1d=ensure_dims,
                            name='data', dtype=float
                        )
                        if self._check_finite:
                            np.asarray_chkfinite(y[np.logical_not(self.mask)])
                else:
                    y = data
                    input_y = False

            if input_y and not skip_sorting:
                y = _sort_array(y, sort_order=self._sort_order)

            if self.mask is not None:
                if mask_support == -1 and self._strict_mask:
                    raise NotImplementedError(f'masking is not supported for {func.__name__}')

                # TODO maybe add a private bool attribute like "_mask_fill" that allows skipping
                # the zero-filling/interpolation for testing purposes to ensure everything actually
                # works -> probably would only be useful for weighted methods, with non-nan y
                if mask_support != 0:
                    if mask_support == 1:
                        # algorithm strictly uses weighted fitting, so can just zero-fill the mask;
                        # faster than interpolation, so use whenever possible
                        y = np.where(self.mask, 0., y)
                    else:
                        inv_mask = np.logical_not(self.mask)
                        y = np.interp(self.x, self.x[inv_mask], y[inv_mask])
            baseline, params = func(self, y, *args, **kwargs)

            return self._return_results(baseline, params, self._dtype, sort_keys, skip_sorting)

        return inner

    def _override_x(self, new_x, new_sort_order=None, new_mask=None):
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
            output_dtype=self._dtype, strict_mask=self._strict_mask
        )
        new_object.banded_solver = self.banded_solver
        new_object._sort_order = new_sort_order
        if new_sort_order is not None:
            new_object._inverted_order = _inverted_sort(new_sort_order)
        # add mask after setting sort order so it's correctly sorted
        new_object.mask = new_mask

        return new_object

    def _setup_whittaker(self, y, lam=1, diff_order=2, weights=None, copy_weights=False,
                         allow_lower=True, reverse_diags=False):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
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
        reverse_diags : bool, optional
            If True, will reverse the order of the diagonals of the squared difference
            matrix. Default is False.

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
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._size, weights, copy_input=copy_weights or has_mask,
            check_finite=self._check_finite, dtype=float
        )
        if weights is not None:
            weight_array = _sort_array(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0

        allow_lower = allow_lower and self.banded_solver < 4
        allow_penta = self.banded_solver < 3

        whittaker_system = PenalizedSystem(
            self._size, lam, diff_order, allow_lower, reverse_diags, allow_penta=allow_penta,
            penta_solver=self._penta_solver
        )

        return y, weight_array, whittaker_system

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=True,
                          calc_pinv=False, copy_weights=False, max_cross=None):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        poly_order : int, optional
            The polynomial order. Default is 2.
        calc_vander : bool, optional
            If True (default), will calculate and the Vandermonde matrix.
        calc_pinv : bool, optional
            If True, and if `return_vander` is True, will calculate and return the
            pseudo-inverse of the Vandermonde matrix. Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        max_cross : None, optional
            Not used within this method, simply added to have the same call signature
            as `_Algorithm2D._setup_polynomial`.

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
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._size, weights, copy_input=copy_weights or has_mask,
            check_finite=self._check_finite, dtype=float
        )
        if weights is not None:
            weight_array = _sort_array(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0

        if calc_vander:
            if self._polynomial is None:
                self._polynomial = _PolyHelper(self.x, self.x_domain, poly_order)
            else:
                self._polynomial.recalc_vandermonde(self.x, self.x_domain, poly_order)

        if not calc_pinv:
            return y, weight_array
        elif not calc_vander:
            raise ValueError('if calc_pinv is True, then calc_vander must also be True')

        if weights is None and not has_mask:
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
            array by :meth:`~._Algorithm._handle_io`.
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
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._size, weights, dtype=float, order='C',
            copy_input=copy_weights or has_mask, check_finite=self._check_finite
        )
        if weights is not None:
            weight_array = _sort_array(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0

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

    def _setup_pls(self, y, weights=None, spline_degree=None, num_knots=10,
                   diff_order=2, lam=1, allow_lower=True, reverse_diags=False,
                   copy_weights=False, num_eigens=None):
        """
        Sets the starting parameters for methods using penalized least squares.

        Depending on the input of `spline_degree`, will dispatch to either
        `_setup_whittaker` or `_setup_spline`.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        spline_degree : int or None, optional
            If None (default), denotes that the system is using Whittaker smoothing.
            Otherwise, the system is a penalized spline with a spline degree of `spline_degree`.
        num_knots : int, optional
            The number of interior knots for the splines. Only used if `spline_degree` is
            not None. Default is 10.
        diff_order : int, optional
            The integer differential order for the penalty; must be greater than 0.
            Default is 2.
        lam : float, optional
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on `diff_order` and the data size.
            Default is 1.
        allow_lower : boolean, optional
            If True (default), will include only the lower non-zero diagonals of
            the squared difference matrix. If False, will include all non-zero diagonals.
        reverse_diags : boolean, optional
            If True, will reverse the order of the diagonals of the penalty matrix.
            Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        num_eigens : None, optional
            Not used within this method, simply added to have the same call signature
            as `_Algorithm2D._setup_pls`.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for fitting the spline to the data.
        penalized_system : PenalizedSystem or PSpline
            The object for solving the penalized least squared system. If `spline_degree`
            is None, returns a PenalizedSystem object;, otherwise, returns a PSpline.
        result_class : WhittakerResult or PSplineResult
            The result class for defining the solution. If `spline_degree`
            is None, returns WhittakerResult; otherwise, returns PSplineResult.

        """
        if spline_degree is None:
            y, weight_array, penalized_system = self._setup_whittaker(
                y, lam=lam, diff_order=diff_order, weights=weights, copy_weights=copy_weights,
                allow_lower=allow_lower, reverse_diags=reverse_diags
            )
            result_class = WhittakerResult
        else:
            y, weight_array, penalized_system = self._setup_spline(
                y, lam=lam, diff_order=diff_order, weights=weights, copy_weights=copy_weights,
                allow_lower=allow_lower, reverse_diags=reverse_diags,
                spline_degree=spline_degree, num_knots=num_knots, penalized=True, make_basis=True
            )
            result_class = PSplineResult

        return y, weight_array, penalized_system, result_class

    def _setup_morphology(self, y, half_window=None, window_kwargs=None, **kwargs):
        """
        Sets the starting parameters for morphology-based methods.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`pybaselines.utils.estimate_window`.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.estimate_window` for
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

            output_half_window = estimate_window(y, **window_kwargs, **kwargs)

        return y, output_half_window

    def _setup_smooth(self, y, half_window=None, pad_type='half', window_multiplier=1,
                      pad_kwargs=None, **kwargs):
        """
        Sets the starting parameters for doing smoothing-based algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
        half_window : int, optional
            The half-window used for the smoothing functions. Used
            to pad the left and right edges of the data to reduce edge
            effects. Default is is None, which sets the half window as the output of
            :func:`pybaselines.utils.estimate_window` multiplied by `window_multiplier`.
        pad_type : {'half', 'full', None}
            If True (default), will pad the input `y` with `half_window` on each side
            before returning. If False, will return the unmodified `y`.
        window_multiplier : int or float, optional
            The multiplier by which the output of :func:`pybaselines.utils.estimate_window`
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
            output_half_window = max(1, int(window_multiplier * estimate_window(y)))
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
            array by :meth:`~._Algorithm._handle_io`.
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
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._size, weights, dtype=bool, check_finite=self._check_finite,
            copy_input=has_mask
        )
        if weights is not None:
            weight_array = _sort_array(weight_array, self._sort_order)
        if has_mask:
            # for classification methods, weight = False means masked points cannot be
            # used to determine the background, which seems like the correct logic
            weight_array[self.mask] = False

        return y, weight_array

    def _spawn_fitter(self, method, ensure_new=False):
        """
        Creates an appropriate fitting object for the indicated method.

        Parameters
        ----------
        method : str
            The string name of the desired method.
        ensure_new : bool, optional
            If True, will ensure that the output `class_object`
            correspond to a new object rather than `self`.

        Returns
        -------
        class_object : pybaselines._algorithm_setup._Algorithm
            The `_Algorithm` object which will be used for fitting.

        Raises
        ------
        AttributeError
            Raised if `method` is not an available Baseline method.

        """
        self_has = hasattr(self, method)

        # if self is a Baseline class, can just use its method
        if self_has and not ensure_new:
            class_object = self
        else:
            if self_has:
                klass = self.__class__
            else:
                # just directly use Baseline rather than the individual private classes
                from .api import Baseline
                if not hasattr(Baseline, method):
                    raise AttributeError(f'{method} is not a valid method')
                klass = Baseline
            # have to reset x ordering so that all outputs and parameters are
            # correctly sorted
            if self._sort_order is not None:
                x = self.x[self._inverted_order]
                assume_sorted = False
            else:
                x = self.x
                assume_sorted = True
            class_object = klass(
                x, check_finite=self._check_finite, assume_sorted=assume_sorted,
                output_dtype=self._dtype, mask=self.mask, strict_mask=self._strict_mask
            )
            class_object.banded_solver = self.banded_solver

        return class_object

    def _setup_optimizer(self, y, method, method_param=None, method_kwargs=None, copy_kwargs=True,
                         ensure_new=False, needed_params=None):
        """
        Sets the starting parameters for doing optimizer algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.
        method : str
            The string name of the desired function, like 'asls'. Case does not matter.
        method_param : dict, optional
            A dictionary indicating potential parameter keys to use, with the default having
            a key of None. For example, a `method_param` of {'method1': 'a', None: ('b', 'c')}
            would specify that parameter 'a' should be used for `method`='method1'; otherwise,
            either 'b' or 'c' could be potential parameters, which would then be filtered by
            looking at the signature of the indicated method. Default is None, which indicates
            that the optimizer method being used does not require any parameter key.
        method_kwargs : dict, optional
            A dictionary of keyword arguments to pass to the fitting function. Default
            is None, which uses an empty dictionary.
        copy_kwargs : bool, optional
            If True (default), will copy the input `method_kwargs` so that the input
            dictionary is not modified within the function.
        ensure_new : bool, optional
            If True, will ensure that the output `class_object` and `baseline_func`
            correspond to a new object rather than `self`. This is to ensure
            thread safety for methods which would modify internal state not typically
            assumed to change when using threading, such as changing polynomial degrees.
            Default is False.
        needed_params : Iterable, optional
            An iterature of other necessary parameter keys that the method must have in its
            signature. For example ['weights', 'tol'] would error if either 'weights' or 'tol'
            are not valid inputs. Default is None.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        optimizer_obj : _OptimizerHelper
            The object containing the fitting object to use and all relevant fields
            for optimizer-type methods.
        method_kws : dict
            A dictionary of keyword arguments to pass to `fit_func`.

        Raises
        ------
        KeyError
            Raised if method_kwargs has the 'x_data' key.

        """
        optimizer_obj = _OptimizerHelper(
            method, self, ensure_new=ensure_new, method_param=method_param,
            needed_params=needed_params
        )
        if method_kwargs is None:
            method_kws = {}
        elif copy_kwargs:
            method_kws = method_kwargs.copy()
        else:
            method_kws = method_kwargs

        if 'x_data' in method_kws:
            raise KeyError('"x_data" should not be within the method keyword arguments')

        return y, optimizer_obj, method_kws

    def _setup_misc(self, y):
        """
        Sets the starting parameters for doing miscellaneous algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm._handle_io`.

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

    Returns
    -------
    Callable
        The wrapped function.

    """
    def outer(func):
        func_signature = None  # delay computing the signature until it is actually needed
        method = func.__name__

        @wraps(func)
        def inner(*args, **kwargs):
            nonlocal func_signature
            if func_signature is None:
                func_signature = signature(func)
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
        The last polynomial order used to calculate the Vandermonde matrix.
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
        poly_order = _check_scalar_variable(
            poly_order, allow_zero=True, variable_name='polynomial order', dtype=int
        )
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
