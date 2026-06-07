# -*- coding: utf-8 -*-
"""Setup code for the various algorithm types in pybaselines.

Created on April 8, 2023
@author: Donald Erb

"""

from functools import partial, wraps
import itertools
import warnings

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from ..results import PSplineResult2D, WhittakerResult2D
from .._nd.optimizers import _OptimizerHelper
from .._validation import (
    _check_array, _check_half_window, _check_optional_array, _check_scalar_variable,
    _check_sized_array, _yxz_arrays
)
from ..utils import (
    ParameterWarning, SortingWarning, _determine_sorts, _sort_array2d, estimate_window, pad_edges2d
)
from ._spline_utils import PSpline2D, SplineBasis2D
from ._whittaker_utils import WhittakerSystem2D


class _Algorithm2D:
    """
    A base class for all 2D algorithm types.

    Contains setup methods for all algorithm types to make more complex algorithms
    easier to set up.

    Attributes
    ----------
    x : numpy.ndarray or None
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same size as the input `data.shape[-2]` and has min
        and max values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).
    z : numpy.ndarray or None
        The z-values for the object. If initialized with None, then `z` is initialized the
        first function call to have the same size as the input `data.shape[-1]` and has min
        and max values of -1 and 1, respectively.
    z_domain : numpy.ndarray
        The minimum and maximum values of `z`. If `z_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).

    """

    def __init__(self, x_data=None, z_data=None, check_finite=True, assume_sorted=False,
                 output_dtype='deprecated', mask=None, strict_mask=True):
        """
        Initializes the algorithm object.

        Parameters
        ----------
        x_data : array-like, shape (M,), optional
            The x-values of the measured data (independent variable for the rows). Default is
            None, which will create an array from -1 to 1 during the first function call with
            length equal to the number of rows in the input data.
        z_data : array-like, shape (N,), optional
            The z-values of the measured data (independent variable for the columns). Default is
            None, which will create an array from -1 to 1 during the first function call with
            length equal to the number of columns in the input data.
        check_finite : bool, optional
            If True (default), will raise an error if any values in input data are not finite.
            Setting to False will skip the check. Note that errors may occur if
            `check_finite` is False and the input data contains non-finite values.
        assume_sorted : bool, optional
            If False (default), will sort the input `x_data` and `z_data` values. Otherwise,
            the input is assumed to be sorted, although they will still be checked to be in
            ascending order. Note that some methods will raise an error if `x_data` or `z_data`
            values are not unique.
        output_dtype : type or numpy.dtype, optional
            The dtype to cast the output array. Default behavior keeps the dtype as float.

            .. deprecated:: 1.3
                `output_dtype` is deprecated and will be removed in version 1.5. Use
                :func:`numpy.astype` on the output baseline instead.

        mask : array-like, shape (M, N), optional
            A Boolean array, in which all indices that are `True` denote indices that should
            be ignored during baseline correction. If None (default), denotes all values should
            be included.
        strict_mask : bool, optional
            If True (default) and `mask` is not None, calling any method that does not support
            masking or calculations using the mask that produce invalid results will raise an
            exception. If `strict_mask` is False, will instead perform linear interpolation
            following `mask` before performing baseline correction to prevent those issues.

        """
        x_sort_order = None
        z_sort_order = None
        self._shape = None
        no_x = x_data is None
        no_z = z_data is None
        if no_x:
            self.x = None
            self.x_domain = np.array([-1., 1.])
        else:
            self.x = _check_array(x_data, dtype=float, check_finite=check_finite)
            if assume_sorted and np.any(self.x[1:] < self.x[:-1]):
                warnings.warn(
                    ('x-values must be strictly increasing for many methods, so setting '
                     'assume_sorted to True'), SortingWarning, stacklevel=2
                )
                assume_sorted = False

            self._shape = (len(self.x), None)
            self.x_domain = np.polynomial.polyutils.getdomain(self.x)
            if not assume_sorted:
                x_sort_order, x_inverted_order = _determine_sorts(self.x)
                if x_sort_order is not None:
                    self.x = self.x[x_sort_order]

        if no_z:
            self.z = None
            self.z_domain = np.array([-1., 1.])
        else:
            self.z = _check_array(z_data, dtype=float, check_finite=check_finite)
            if assume_sorted and np.any(self.z[1:] < self.z[:-1]):
                warnings.warn(
                    ('z-values must be strictly increasing for many methods, so setting '
                     'assume_sorted to True'), SortingWarning, stacklevel=2
                )
                assume_sorted = False
            self._shape = (self._shape[0], len(self.z))
            self.z_domain = np.polynomial.polyutils.getdomain(self.z)
            if not assume_sorted:
                z_sort_order, z_inverted_order = _determine_sorts(self.z)
                if z_sort_order is not None:
                    self.z = self.z[z_sort_order]

        if x_sort_order is None and z_sort_order is None:
            self._sort_order = None
            self._inverted_order = None
        elif z_sort_order is None:
            self._sort_order = x_sort_order
            self._inverted_order = x_inverted_order
        elif x_sort_order is None:
            self._sort_order = (..., z_sort_order)
            self._inverted_order = (..., z_inverted_order)
        else:
            self._sort_order = (x_sort_order[:, None], z_sort_order[None, :])
            self._inverted_order = (x_inverted_order[:, None], z_inverted_order[None, :])

        if output_dtype != 'deprecated':
            warnings.warn(
                'specifying "output_dtype" is deprecated and will be removed in version 1.5. '
                'Use np.astype on the output baseline instead.', DeprecationWarning, stacklevel=3
            )

        self._polynomial = None
        self._spline_basis = None
        self._check_finite = check_finite
        self._dtype = output_dtype
        self.banded_solver = 2
        self._validated_x = no_x
        self._validated_z = no_z
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
        values : array-like, shape (M, N), optional
            A Boolean array, in which all indices that are `True` denote indices that should
            be omitted during baseline correction. If None, denotes all values should
            be included.

        """
        if values is None:
            self._mask = None
        else:
            if self._shape == (None, None):
                input_mask = _check_array(
                    values, dtype=bool, ensure_1d=False, ensure_2d=True, two_d=True
                )
                self._shape = input_mask.shape
                self.x = np.linspace(-1., 1., self._shape[0])
                self.z = np.linspace(-1., 1., self._shape[1])
            elif self._shape[0] is None:
                input_mask = _check_sized_array(
                    values, self._shape[1], name='mask', dtype=bool,
                    ensure_1d=False, ensure_2d=True, two_d=True, axis=1
                )
                self._shape = input_mask.shape
                self.x = np.linspace(-1., 1., self._shape[0])
            elif self._shape[1] is None:
                input_mask = _check_sized_array(
                    values, self._shape[0], name='mask', dtype=bool,
                    ensure_1d=False, ensure_2d=True, two_d=True, axis=0
                )
                self._shape = input_mask.shape
                self.z = np.linspace(-1., 1., self._shape[1])
            else:
                input_mask = _check_sized_array(
                    values, self._shape, name='mask', dtype=bool,
                    ensure_1d=False, ensure_2d=True, two_d=True, axis=slice(None)
                )
            self._mask = _sort_array2d(input_mask, self._sort_order)
            # TODO should ensure some lower bound for input_mask.sum() so that there
            # are actually enough points to do calculations; maybe 5-10% of the points?

    @property
    def _shape(self):
        """The shape of the _Algorithm2D object."""
        return self.__shape

    @_shape.setter
    def _shape(self, value):
        """Sets the shape and size of the _Algorithm2D object.

        Parameters
        ----------
        value : Sequence[int, int] or None
            The shape of the dataset.

        Notes
        -----
        Follows NumPy naming conventions where _Algorithm2D._size is the total number of items,
        and _Algorithm2D._shape is the number of items in each dimension.

        """
        if value is None:
            self._size = None
            self.__shape = (None, None)
        elif None in value:
            if value[0] is None:
                self.__shape = (self.__shape[0], value[1])
                self._size = None
            else:
                self.__shape = (value[0], self.__shape[1])
                self._size = None
        else:
            self.__shape = tuple(value)
            self._size = np.prod(self._shape)

    @property
    def banded_solver(self):
        """
        Designates the solver to prefer using for solving 1D banded linear systems.

        .. versionadded:: 1.2.0

        Only used to pass to a new :class:`~.Baseline` object when using
        :meth:`.Baseline2D.individual_axes`. An integer between 1 and 4 designating
        the solver to prefer for solving banded linear systems in 1D. See
        :attr:`~.Baseline.banded_solver` for more information.  Default is 2.

        """
        return self._banded_solver

    @banded_solver.setter
    def banded_solver(self, solver):
        """
        Sets the solver to use for 1D banded linear systems.

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
        # note that self._pentapy_solver is not set here since it is unused in 2D
        self._banded_solver = solver

    @property
    def pentapy_solver(self):
        """
        The solver if using the dedicated pentadiagonal solvers to solve banded equations.

        .. deprecated:: 1.2
            The `pentapy_solver` property is deprecated and will be removed in
            version 1.4. Use :attr:`~Baseline2D.banded_solver` instead.

        """
        warnings.warn(
            ('The `pentapy_solver` attribute is deprecated and will be removed in '
             'version 1.4; use the `banded_solver` attribute instead'),
            DeprecationWarning, stacklevel=2
        )
        return self.banded_solver

    @pentapy_solver.setter
    def pentapy_solver(self, value):
        warnings.warn(
            ('Setting the `pentapy_solver` attribute is deprecated and will be removed in '
             'version 1.4; set the `banded_solver` attribute instead'),
            DeprecationWarning, stacklevel=2
        )
        self.banded_solver = value

    def _return_results(self, baseline, params, dtype, sort_keys=(), ensure_dims=True,
                        reshape_keys=(), skip_sorting=False):
        """
        Re-orders the input baseline and parameters based on the x ordering.

        If `self._sort_order` is None, then no reordering is performed.

        Parameters
        ----------
        baseline : numpy.ndarray, shape (M, N)
            The baseline output by the baseline function.
        params : dict
            The parameter dictionary output by the baseline function.
        dtype : type or numpy.dtype, optional
            The desired output dtype for the baseline.
        sort_keys : Iterable, optional
            An iterable of keys corresponding to the values in `params` that need
            re-ordering. Default is ().
        ensure_dims : bool, optional
            If True (default), will raise an error if the shape of `array` is not a two dimensional
            array with shape (M, N) or a three dimensional array with shape (M, N, 1), (M, 1, N),
            or (1, M, N).
        reshape_keys : tuple, optional
            The keys within the output parameter dictionary that will need reshaped to match the
            shape of the data. For example, used to convert weights for polynomials from 1D back
            into the original shape. Default is ().
        skip_sorting : bool, optional
            If True, will skip sorting the output baseline. The keys in `sort_keys` will
            still be sorted. Default is False.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The input `baseline` after re-ordering and setting to the desired dtype.
        params : dict
            The input `params` after re-ordering the values for `sort_keys`.

        """
        ndims = baseline.ndim
        if ndims == 1 and ensure_dims:  # raveled to 1D within the method
            baseline = baseline.reshape(self._shape)
        elif ndims == 2 and not ensure_dims:  # 3D input raveled to (P, M * N)
            baseline = baseline.reshape(-1, *self._shape)

        for key in reshape_keys:
            if key in params:
                # TODO can any params be non-2d that need reshaped?
                params[key] = params[key].reshape(self._shape)

        if self._sort_order is not None:
            for key in sort_keys:
                if key in params:  # some parameters are conditionally output
                    # assumes params all all two dimensional arrays
                    params[key] = params[key][self._inverted_order]

            if not skip_sorting:
                baseline = _sort_array2d(baseline, sort_order=self._inverted_order)
        if dtype != 'deprecated':
            baseline = np.asarray(baseline, dtype=dtype)

        return baseline, params

    @classmethod
    def _handle_io(cls, func=None, *, sort_keys=(), ensure_dims=True, reshape_keys=(),
                   skip_sorting=False, require_unique=False, mask_support=-1):
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
            sort order of ``self.x`` and ``self.z``. Default is ().
        ensure_dims : bool, optional
            If True (default), will raise an error if the shape of `array` is not a two dimensional
            array with shape (M, N) or a three dimensional array with shape (M, N, 1), (M, 1, N),
            or (1, M, N).
        reshape_keys : tuple, optional
            The keys within the output parameter dictionary that will need reshaped to match the
            shape of the data. For example, used to convert weights for polynomials from 1D back
            into the original shape. Default is ().
        skip_sorting : bool, optional
            If True, will skip sorting the output baseline. The keys in `sort_keys` will
            still be sorted. Default is False.
        require_unique : bool, optional
            If True, will check ``self.x`` and ``self.z`` to ensure all values are unique and will
            raise an error if non-unique values are present. Default is False, which skips the
            check.
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
                reshape_keys=reshape_keys, skip_sorting=skip_sorting,
                require_unique=require_unique, mask_support=mask_support
            )

        @wraps(func)
        def inner(self, data=None, *args, **kwargs):
            if data is None:
                # not implementing interp_pts for 2D, so data can never
                # be None in 2D
                raise TypeError('"data" cannot be None')

            has_x = self.x is not None
            has_z = self.z is not None
            if has_x or has_z:
                if has_x and has_z:
                    expected_shape = self._shape
                    axis = slice(-2, None)
                elif has_x:
                    expected_shape = self._shape[0]
                    axis = -2
                else:
                    expected_shape = self._shape[1]
                    axis = -1
                if self.mask is None:
                    y = _check_sized_array(
                        data, expected_shape, check_finite=self._check_finite, ensure_1d=False,
                        axis=axis, name='data', ensure_2d=ensure_dims, two_d=True, dtype=float
                    )
                else:
                    y = _check_sized_array(
                        data, expected_shape, check_finite=False, ensure_1d=False,
                        axis=axis, name='data', ensure_2d=ensure_dims, two_d=True, dtype=float
                    )
                    if self._check_finite:
                        np.asarray_chkfinite(y[..., np.logical_not(self.mask)])
            else:  # also means self.mask is None
                y, self.x, self.z = _yxz_arrays(
                    data, self.x, self.z, check_finite=self._check_finite, ensure_2d=ensure_dims,
                    dtype=float
                )

            if not has_x:
                self._shape = (y.shape[-2], self._shape[1])
                self.x = np.linspace(-1, 1, self._shape[0])
            elif require_unique and not self._validated_x:
                if np.any(self.x[1:] == self.x[:-1]):
                    raise ValueError('x-values must be unique for the selected method')
                else:
                    self._validated_x = True
            if not has_z:
                self._shape = (self._shape[0], y.shape[-1])
                self.z = np.linspace(-1, 1, self._shape[1])
            elif require_unique and not self._validated_z:
                if np.any(self.z[1:] == self.z[:-1]):
                    raise ValueError('z-values must be unique for the selected method')
                else:
                    self._validated_z = True

            if not skip_sorting:
                y = _sort_array2d(y, sort_order=self._sort_order)

            if self.mask is not None:
                if mask_support == -1 and self._strict_mask:
                    raise NotImplementedError('masking is not supported for this method')

                if mask_support != 0:
                    if mask_support == 1:
                        # algorithm strictly uses weighted fitting, so can just zero-fill the mask;
                        # faster than interpolation, so use whenever possible
                        y = np.where(self.mask, 0., y)
                    else:
                        inv_mask = np.logical_not(self.mask)
                        X, Z = np.meshgrid(self.x, self.z, indexing='ij')
                        if y.ndim == 2:
                            # cannot use RegularGridInterpolator since non-masked points are not
                            # guaranteed to be on a regular grid
                            y = LinearNDInterpolator(
                                np.stack((X[inv_mask], Z[inv_mask]), axis=-1), y[inv_mask],
                                fill_value=0
                            )(np.stack((X, Z), axis=-1)).reshape(self._shape)

                        else:
                            y = np.stack(
                                [
                                    LinearNDInterpolator(
                                        np.stack((X[inv_mask], Z[inv_mask]), axis=-1),
                                        row[inv_mask], fill_value=0
                                    )(np.stack((X, Z), axis=-1)).reshape(self._shape) for row in y
                                ], axis=0
                            )

            baseline, params = func(self, y, *args, **kwargs)

            return self._return_results(
                baseline, params, dtype=self._dtype, sort_keys=sort_keys, ensure_dims=ensure_dims,
                reshape_keys=reshape_keys, skip_sorting=skip_sorting
            )

        return inner

    def _override_x(self, new_x, new_sort_order=None, new_mask=None):
        """
        Creates a new fitting object for the given x-values and z-values.

        Useful when fitting extensions of the x attribute.

        Parameters
        ----------
        new_x : numpy.ndarray
            The x values to temporarily use.
        new_sort_order : numpy.ndarray, optional
            The sort order for the new x values. Default is None, which will not sort.
        new_mask : numpy.ndarray, optional
            The mask for the new values. Default is None.

        Returns
        -------
        pybaselines._algorithm_setup._Algorithm2D
            The _Algorithm object with the new x attribute.

        Raises
        ------
        NotImplementedError
            Raised since this usage is not currently needed so no 2D implementation was made.

        """
        raise NotImplementedError

    def _setup_whittaker(self, y, lam=1, diff_order=2, weights=None, copy_weights=False,
                         num_eigens=None):
        """
        Sets the starting parameters for doing penalized least squares.

        Parameters
        ----------
        y : numpy.ndarray, shape (M ,N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        lam : float or Sequence[float, float], optional
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on the penalized least square method
            and the differential order. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The integer differential order; must be greater than 0. Default is 2.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape (M, N) and all values set to 1.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for eigendecomposition. Typical values are between 5 and 30, with higher values
            needed for baselines with more curvature. If None, will solve the linear system
            using the full analytical solution, which is typically much slower.
            Default is None.

        Returns
        -------
        y : numpy.ndarray, shape (``M * N``)
            The y-values of the measured data after flattening.
        weight_array : numpy.ndarray, shape (``M * N``)
            The weight array after flattening.
        whittaker_system : WhittakerSystem2D
            The WhittakerSystem2D for solving the given penalized least squared system.

        Raises
        ------
        ValueError
            Raised is `diff_order` is less than 1.

        Warns
        -----
        ParameterWarning
            Raised if `diff_order` is greater than 3.

        """
        diff_order = _check_scalar_variable(
            diff_order, allow_zero=False, variable_name='difference order', two_d=True, dtype=int
        )
        if (diff_order > 3).any():
            warnings.warn(
                ('difference orders greater than 3 can have numerical issues;'
                 ' consider using a difference order of 2 or 1 instead'),
                ParameterWarning, stacklevel=2
            )
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._shape, weights, copy_input=copy_weights or has_mask,
            check_finite=self._check_finite, ensure_1d=False, axis=slice(None), dtype=float
        )
        if weights is not None:
            weight_array = _sort_array2d(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0

        # TODO can probably keep the basis for reuse if using SVD, like _setup_spline does, and
        # retain the unmodified penalties for the rows and columns if possible to skip that
        # calculation as well
        whittaker_system = WhittakerSystem2D(
            self._shape, lam, diff_order, num_eigens
        )
        if not whittaker_system._using_svd:
            y = y.ravel()
            weight_array = weight_array.ravel()

        return y, weight_array, whittaker_system

    def _setup_polynomial(self, y, weights=None, poly_order=2, calc_vander=True,
                          calc_pinv=False, copy_weights=False, max_cross=None):
        """
        Sets the starting parameters for doing polynomial fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns, respectively. Default is 2.
        calc_vander : bool, optional
            If True (default), will calculate and the Vandermonde matrix.
        calc_pinv : bool, optional
            If True, and if `return_vander` is True, will calculate and return the
            pseudo-inverse of the Vandermonde matrix. Default is False.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        max_cross : int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0. Default is None, which
            does not limit the cross terms.

        Returns
        -------
        y : numpy.ndarray, shape (``M * N``)
            The y-values of the measured data after flattening.
        weight_array : numpy.ndarray, shape (``M * N``)
            The weight array for fitting a polynomial to the data after flattening.
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
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._shape, weights, copy_input=copy_weights or has_mask,
            check_finite=self._check_finite, ensure_1d=False, axis=slice(None), dtype=float
        )
        if weights is not None:
            weight_array = _sort_array2d(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0
        weight_array = weight_array.ravel()

        if calc_vander:
            if self._polynomial is None:
                self._polynomial = _PolyHelper2D(
                    self.x, self.z, self.x_domain, self.z_domain, poly_order, max_cross
                )
            else:
                self._polynomial.recalc_vandermonde(
                    self.x, self.z, self.x_domain, self.z_domain, poly_order, max_cross
                )

        y = y.ravel()
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
                      penalized=True, diff_order=3, lam=1, make_basis=True, copy_weights=False):
        """
        Sets the starting parameters for doing spline fitting.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        spline_degree : int or Sequence[int, int], optional
            The degree of the spline. Default is 3, which is a cubic spline.
        num_knots : int or Sequence[int, int], optional
            The number of interior knots for the splines. Default is 10.
        penalized : bool, optional
            Whether the basis matrix should be for a penalized spline or a regular
            B-spline. Default is True, which creates the basis for a penalized spline.
        diff_order : int or Sequence[int, int], optional
            The integer differential order for the spline penalty; must be greater than 0.
            Default is 3. Only used if `penalized` is True.
        lam : float or Sequence[float, float], optional
            The smoothing parameter, lambda. Typical values are between 10 and
            1e8, but it strongly depends on the number of knots and the difference order.
            Default is 1.
        make_basis : bool, optional
            If True (default), will create the matrix containing the spline basis functions.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.

        Returns
        -------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data.
        weight_array : numpy.ndarray, shape (M, N)
            The weight array for fitting the spline to the data.
        pspline : PSpline2D
            The PSpline2D object for solving the given penalized least squared system. Only
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
            self._shape, weights, copy_input=copy_weights or has_mask,
            check_finite=self._check_finite, ensure_1d=False, axis=slice(None), dtype=float
        )
        if weights is not None:
            weight_array = _sort_array2d(weight_array, self._sort_order)
        if has_mask:
            weight_array[self.mask] = 0
        diff_order = _check_scalar_variable(
            diff_order, allow_zero=False, variable_name='difference order', two_d=True, dtype=int
        )
        if not make_basis:
            return y, weight_array

        if (diff_order > 4).any():
            warnings.warn(
                ('differential orders greater than 4 can have numerical issues;'
                    ' consider using a differential order of 2 or 3 instead'),
                ParameterWarning, stacklevel=2
            )

        if (
            self._spline_basis is None
            or not self._spline_basis.same_basis(num_knots, spline_degree)
        ):
            self._spline_basis = SplineBasis2D(self.x, self.z, num_knots, spline_degree)

        # TODO should probably also retain the unmodified penalties for the rows and
        # columns if possible to skip that calculation as well
        pspline = PSpline2D(self._spline_basis, lam, diff_order)

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
            Not used within this method, simply added to have the same call signature
            as `_Algorithm._setup_pls`.
        reverse_diags : boolean, optional
            Not used within this method, simply added to have the same call signature
            as `_Algorithm._setup_pls`.
        copy_weights : boolean, optional
            If True, will copy the array of input weights. Only needed if the
            algorithm changes the weights in-place. Default is False.
        num_eigens : int or Sequence[int, int] or None
            Only used if `spline_degree` is None. The number of eigenvalues for the rows
            and columns, respectively, to use for eigendecomposition. If None, will s
            olve the linear system using the full analytical solution, which is typically
            much slower. Default is None.

        Returns
        -------
        y : numpy.ndarray, shape (N,)
            The y-values of the measured data, converted to a numpy array.
        weight_array : numpy.ndarray, shape (N,)
            The weight array for fitting the spline to the data.
        penalized_system : WhittakerSystem2D or PSpline2D
            The object for solving the penalized least squared system. If `spline_degree`
            is None, returns a WhittakerSystem2D object;, otherwise, returns a PSpline2D.
        result_class : WhittakerResult2D or PSplineResult2D
            The result class for defining the solution. If `spline_degree`
            is None, returns WhittakerResult2D; otherwise, returns PSplineResult2D.

        """
        if spline_degree is None:
            y, weight_array, penalized_system = self._setup_whittaker(
                y, lam=lam, diff_order=diff_order, weights=weights, copy_weights=copy_weights,
                num_eigens=num_eigens
            )
            result_class = WhittakerResult2D
        else:
            y, weight_array, penalized_system = self._setup_spline(
                y, lam=lam, diff_order=diff_order, weights=weights, copy_weights=copy_weights,
                spline_degree=spline_degree, num_knots=num_knots, penalized=True, make_basis=True
            )
            result_class = PSplineResult2D

        return y, weight_array, penalized_system, result_class

    def _setup_morphology(self, y, half_window=None, window_kwargs=None, **kwargs):
        """
        Sets the starting parameters for morphology-based methods.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        half_window : int or Sequence[int, int], optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`pybaselines.utils.estimate_window`.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to
            :func:`pybaselines.utils.estimate_window`.
        **kwargs
            Deprecated in version 1.2.0 and will be removed in version 1.4.0. Pass keyword
            arguments for :func:`.estimate_window` using `window_kwargs`.

        Returns
        -------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data.
        output_half_window : np.ndarray[int, int]
            The accepted half windows.

        Notes
        -----
        Ensures that window size is odd since morphological operations operate in
        the range [-output_half_window, ..., output_half_window].

        Half windows are dealt with rather than full window sizes to clarify their
        usage. SciPy morphology operations deal with full window sizes.

        """
        if half_window is not None:
            output_half_window = _check_half_window(half_window, two_d=True)
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

    def _setup_smooth(self, y, half_window=None, window_multiplier=1, pad_kwargs=None, **kwargs):
        """
        Sets the starting parameters for doing smoothing-based algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        half_window : int or Sequence[int, int], optional
            The half-window used for the smoothing functions. Used
            to pad the left and right edges of the data to reduce edge
            effects. Default is is None, which sets the half window as the output of
            :func:`pybaselines.utils.estimate_window` multiplied by `window_multiplier`.
        window_multiplier : int, optional
            The value to multiply the output of :func:`.estimate_window` if half_window
            is None. Default is 1.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges2d` for padding
            the edges of the data to prevent edge effects from smoothing. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `pad_kwargs`.

        Returns
        -------
        numpy.ndarray, shape (``M + 2 * half_window[0]``, ``N + 2 * half_window[1]``)
            The padded array of data.
        output_hw : np.ndarray[int, int]
            The accepted half windows.

        """
        if half_window is not None:
            output_hw = _check_half_window(half_window, allow_zero=False, two_d=True)
        else:
            output_hw = window_multiplier * estimate_window(y)

        pad_kwargs = pad_kwargs if pad_kwargs is not None else {}
        if kwargs:
            warnings.warn(
                ('Passing additional keyword arguments for padding is '
                    'deprecated and will be removed in version 1.4.0. Place all keyword '
                    'arguments into the "pad_kwargs" dictionary instead'),
                DeprecationWarning, stacklevel=2
            )

        return pad_edges2d(y, output_hw, **pad_kwargs, **kwargs), output_hw

    def _setup_classification(self, y, weights=None):
        """
        Sets the starting parameters for doing classification algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.

        Returns
        -------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data.
        weight_array : numpy.ndarray, shape (M, N)
            The weight array for the data, with boolean dtype.

        """
        has_mask = self.mask is not None
        weight_array = _check_optional_array(
            self._shape, weights, check_finite=self._check_finite, dtype=bool,
            ensure_1d=False, axis=slice(None), copy_input=has_mask
        )
        if weights is not None:
            weight_array = _sort_array2d(weight_array, self._sort_order)
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
        class_object : pybaselines.two_d_algorithm_setup._Algorithm2D
            The `_Algorithm2D` object which will be used for fitting.

        Raises
        ------
        AttributeError
            Raised if `method` is not an available Baseline2D method.

        """
        self_has = hasattr(self, method)

        # if self is a Baseline2D class, can just use its method
        if self_has and not ensure_new:
            class_object = self
        else:
            if self_has:
                klass = self.__class__
            else:
                # just directly use Baseline2D rather than the individual private classes
                from .api import Baseline2D
                if not hasattr(Baseline2D, method):
                    raise AttributeError(f'{method} is not a valid method')
                klass = Baseline2D
            # have to reset x and z ordering so that all outputs and parameters are
            # correctly sorted
            mask = self.mask
            if self._sort_order is None:
                x = self.x
                z = self.z
                assume_sorted = True
            else:
                assume_sorted = False
                if isinstance(self._sort_order, tuple):
                    if self._sort_order[0] is Ellipsis:
                        x = self.x
                        z = self.z[self._inverted_order[1]]
                    else:
                        x = self.x[self._inverted_order[0][:, 0]]
                        z = self.z[self._inverted_order[1][0]]
                else:
                    x = self.x[self._inverted_order]
                    z = self.z
                if self.mask is not None:
                    mask = _sort_array2d(self._mask, self._inverted_order)

            class_object = klass(
                x, z, check_finite=self._check_finite, assume_sorted=assume_sorted,
                output_dtype=self._dtype, mask=mask, strict_mask=self._strict_mask
            )
            class_object.banded_solver = self.banded_solver

        return class_object

    def _setup_optimizer(self, y, method, method_param=None, method_kwargs=None, copy_kwargs=True,
                         ensure_new=False, needed_params=None):
        """
        Sets the starting parameters for doing optimizer algorithms.

        Parameters
        ----------
        y : numpy.ndarray
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.
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
        y : numpy.ndarray
            The y-values of the measured data.
        optimizer_obj : _OptimizerHelper
            The object containing the fitting object to use and all relevant fields
            for optimizer-type methods.
        method_kws : dict
            A dictionary of keyword arguments to pass to `fit_func`.

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

        return y, optimizer_obj, method_kws

    def _setup_misc(self, y):
        """
        Sets the starting parameters for doing miscellaneous algorithms.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data, already converted to a numpy
            array by :meth:`~._Algorithm2D._handle_io`.

        Returns
        -------
        y : numpy.ndarray, shape (M, N)
            The y-values of the measured data.

        Notes
        -----
        Since the miscellaneous functions are not related, the only use of this
        function is for aliasing the input `data` to `y`.

        """
        return y


class _PolyHelper2D:
    """
    An object to help with solving polynomials.

    Allows only recalculating the Vandermonde and pseudo-inverse matrices when necessary.

    Attributes
    ----------
    max_cross : int
        The maximum degree for the cross terms.
    poly_order : Container[int, int]
        The last polynomial order used to calculate the Vandermonde matrix.
    pseudo_inverse : numpy.ndarray or None
        The pseudo-inverse of the current Vandermonde matrix.
    vandermonde : numpy.ndarray
        The Vandermonde matrix for solving polynomial equations.

    """

    def __init__(self, x, z, x_domain, z_domain, poly_order, max_cross):
        """
        Initializes the object and calculates the Vandermonde matrix.

        Parameters
        ----------
        x : numpy.ndarray
            The x-values for the polynomial.
        z : numpy.ndarray
            The z-values for the polynomial.
        x_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `x`.
        z_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `z`.
        poly_order : int or Container[int, int]
            The polynomial orders for the rows and columns, respectively.
        max_cross : int
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0.

        """
        self.poly_order = -1
        self.max_cross = None
        self.vandermonde = None
        self._pseudo_inverse = None
        self.pinv_stale = True

        self.recalc_vandermonde(x, z, x_domain, z_domain, poly_order, max_cross)

    def recalc_vandermonde(self, x, z, x_domain, z_domain, poly_order, max_cross):
        """
        Recalculates the Vandermonde matrix for the polynomial only if necessary.

        Also flags whether the pseudo-inverse needs to be recalculated.

        Parameters
        ----------
        x : numpy.ndarray
            The x-values for the polynomial.
        z : numpy.ndarray
            The z-values for the polynomial.
        x_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `x`.
        z_domain : numpy.ndarray, shape (2,)
            The minimum and maximum values of `z`.
        poly_order : int or Container[int, int]
            The polynomial orders for the rows and columns, respectively.
        max_cross : int
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            `x z**2`, `x**2 z`, and `x**2 z**2` would all be set to 0.

        """
        if max_cross is not None:
            max_cross = _check_scalar_variable(
                max_cross, allow_zero=True, variable_name='max_cross', dtype=int
            )
        poly_orders = _check_scalar_variable(
            poly_order, allow_zero=True, variable_name='polynomial order', two_d=True, dtype=int
        )

        # TODO if self.max_cross is None and x- and z- poly_orders are
        # less than self.poly_order, then can just using slicing to reuse
        # the vandermonde like for 1D; would then just need to recalc pinv
        if (
            self.vandermonde is None or self.max_cross != max_cross
            or np.any(self.poly_order != poly_orders)
        ):
            self.pinv_stale = True
            mapped_x = np.polynomial.polyutils.mapdomain(
                x, x_domain, np.array([-1., 1.])
            )
            mapped_z = np.polynomial.polyutils.mapdomain(
                z, z_domain, np.array([-1., 1.])
            )
            # rearrange the vandermonde such that it matches the typical A c = b where b
            # is the flattened version of y and c are the coefficients
            self.vandermonde = np.polynomial.polynomial.polyvander2d(
                *np.meshgrid(mapped_x, mapped_z, indexing='ij'),
                [poly_orders[0], poly_orders[1]]
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
        self.max_cross = max_cross

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
