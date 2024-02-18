# -*- coding: utf-8 -*-
"""Helper functions for working with banded linear systems.

Created on December 8, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded

from ._compat import _HAS_PENTAPY, _pentapy_solve, dia_object, diags, identity
from ._validation import _check_lam


def _shift_rows(matrix, upper_diagonals=2, lower_diagonals=None):
    """
    Shifts the upper and lower diagonals of a banded matrix in compressed form.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to be shifted. Note that all modifications are done in-place.
    upper_diagonals : int, optional
        The number of upper diagonals to shift. Default is 2.
    lower_diagonals : int, optional
        The number of lower diagonals to shift. Default is None, which uses the
        same value as `upper_diagonals`.

    Returns
    -------
    matrix : numpy.ndarray
        The shifted matrix.

    Notes
    -----
    Necessary to match the diagonal matrix format required by SciPy's solve_banded
    function.

    Performs the following transformation with ``upper_diagonals=2`` and ``lower_diagonals=2``
    (left is input, right is output):

        [[a b c ... d 0 0]        [[0 0 a ... b c d]
         [e f g ... h i 0]         [0 e f ... g h i]
         [j k l ... m n o]   -->   [j k l ... m n o]
         [0 p q ... r s t]         [p q r ... s t 0]
         [0 0 u ... v w x]]        [u v w ... x 0 0]]

    The right matrix would be directly obtained when using SciPy's sparse diagonal
    matrices, but when using multiplication with NumPy arrays, the result is the
    left matrix, which has to be shifted to match the desired format.

    """
    for row, shift in enumerate(range(-upper_diagonals, 0)):
        matrix[row, -shift:] = matrix[row, :shift]
        matrix[row, :-shift] = 0

    if lower_diagonals is None:
        lower_diagonals = upper_diagonals
    for pos_row, shift in enumerate(range(lower_diagonals, 0, -1), 1):
        row = -pos_row
        matrix[row, :-shift] = matrix[row, shift:]
        matrix[row, -shift:] = 0

    return matrix


def _lower_to_full(ab):
    """
    Converts a lower banded array to a symmetric banded array.

    The lower bands are flipped and then shifted to make the upper bands.

    Parameters
    ----------
    ab : numpy.ndarray, shape (M, N)
        The lower banded array.

    Returns
    -------
    ab_full : numpy.ndarray, shape (``2 * M - 1``, N)
        The full, symmetric banded array.

    """
    ab_rows, ab_columns = ab.shape
    ab_full = np.concatenate((np.zeros((ab_rows - 1, ab_columns)), ab))
    ab_full[:ab_rows - 1] = ab[1:][::-1]
    _shift_rows(ab_full, upper_diagonals=ab_rows - 1, lower_diagonals=0)

    return ab_full


def _pad_diagonals(diagonals, padding, lower_only=True):
    """
    Pads the diagonals of a banded matrix with zeros.

    Parameters
    ----------
    diagonals : numpy.ndarray, shape (A, N)
        The diagonals of the banded matrix with `A` bands.
    padding : int
        The number of rows of zeros to pad the input with. If `padding` is less than
        or equal to 0, then no padding is applied.
    lower_only : bool, optional
        If True (default), will only add the padding rows to the bottom on the
        input array. If False, will add the padding to both the top and bottom
        of the input.

    Returns
    -------
    output : numpy.ndarray
        The padded diagonals. Has a shape of (``A + padding``, N) if `lower_only` is
        True, otherwise (``A + 2 * padding``, N).

    """
    if padding > 0:
        pad_layers = np.zeros((padding, diagonals.shape[1]))
        if lower_only:
            output = np.concatenate((diagonals, pad_layers))
        else:
            output = np.concatenate((pad_layers, diagonals, pad_layers))
    else:
        output = diagonals

    return output


def _add_diagonals(array_1, array_2, lower_only=True):
    """
    Adds two arrays containing the diagonals of banded matrices.

    The array with the least rows is padded with zeros to allow the sum of the two arrays.

    Parameters
    ----------
    array_1 : numpy.ndarray, shape (A, N)
        An array to add.
    array_2 : numpy.ndarray, shape (B, N)
        An array to add.
    lower_only : bool, optional
        If True (default), will only add zero padding to the bottom of the smaller
        array. If False, will add half of the zero padding to both the top and bottom
        of the smaller array.

    Returns
    -------
    summed_diagonals : numpy.ndarray, shape (`max(A, B)`, N)
        The addition of `a` and `b` after adding the correct zero padding.

    Raises
    ------
    ValueError
        Raised if `a.shape[1]` and `b.shape[1]` are not equal or if `lower` is False
        and `abs(a.shape[0] - b.shape[0])` is not even.

    """
    a, b = np.atleast_2d(array_1, array_2)
    a_shape = a.shape
    b_shape = b.shape
    if a_shape[1] != b_shape[1]:
        raise ValueError((
            f'the diagonal arrays have a dimension mismatch; {a_shape[1]} and {b_shape[1]}'
            ' should be equal'
        ))
    row_mismatch = a_shape[0] - b_shape[0]
    if row_mismatch == 0:
        summed_diagonals = a + b
    else:
        abs_mismatch = abs(row_mismatch)
        if lower_only:
            padding = np.zeros((abs_mismatch, a_shape[1]))
            if row_mismatch > 0:
                summed_diagonals = a + np.concatenate((b, padding))
            else:
                summed_diagonals = np.concatenate((a, padding)) + b
        else:
            if abs_mismatch % 2:
                raise ValueError(
                    'row mismatch between the arrays must be even if lower_only=False, '
                    f'but instead was {abs_mismatch}'
                )
            padding = np.zeros((abs_mismatch // 2, a_shape[1]))
            if row_mismatch > 0:
                summed_diagonals = a + np.concatenate((padding, b, padding))
            else:
                summed_diagonals = np.concatenate((padding, a, padding)) + b

    return summed_diagonals


def difference_matrix(data_size, diff_order=2, diff_format=None):
    """
    Creates an n-order finite-difference matrix.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : int, optional
        The integer differential order; must be >= 0. Default is 2.
    diff_format : str or None, optional
        The sparse format to use for the difference matrix. Default is None,
        which will use the default specified in :func:`scipy.sparse.diags`.

    Returns
    -------
    diff_matrix : scipy.sparse.spmatrix or scipy.sparse._sparray
        The sparse difference matrix.

    Raises
    ------
    ValueError
        Raised if `diff_order` or `data_size` is negative.

    Notes
    -----
    The resulting matrices are sparse versions of::

        import numpy as np
        np.diff(np.eye(data_size), diff_order, axis=0)

    This implementation allows using the differential matrices are they
    are written in various publications, ie. ``D.T @ D``.

    Most baseline algorithms use 2nd order differential matrices when
    doing penalized least squared fitting or Whittaker-smoothing-based fitting.

    """
    if diff_order < 0:
        raise ValueError('the differential order must be >= 0')
    elif data_size < 0:
        raise ValueError('data size must be >= 0')
    elif diff_order > data_size:
        # do not issue warning or exception to maintain parity with np.diff
        diff_order = data_size

    if diff_order == 0:
        # faster to directly create identity matrix
        diff_matrix = identity(data_size, format=diff_format)
    else:
        diagonals = np.zeros(2 * diff_order + 1)
        diagonals[diff_order] = 1
        for _ in range(diff_order):
            diagonals = diagonals[:-1] - diagonals[1:]

        diff_matrix = diags(
            diagonals, np.arange(diff_order + 1),
            shape=(data_size - diff_order, data_size), format=diff_format
        )

    return diff_matrix


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

    diagonals = _pad_diagonals(diagonals, padding, lower_only=lower_only)

    return diagonals


def diff_penalty_matrix(data_size, diff_order=2, diff_format='csr'):
    """
    Creates the finite difference penalty matrix.

    If `D` is the finite difference matrix, then the finite difference penalty
    matrix is defined as ``D.T @ D``.

    Parameters
    ----------
    data_size : int
        The number of data points.
    diff_order : int, optional
        The integer differential order; must be >= 0. Default is 2.
    diff_format : str or None, optional
        The sparse format to use for the difference matrix. Default is 'csr'.

    Returns
    -------
    penalty_matrix : scipy.sparse.spmatrix or scipy.sparse._sparray
        The sparse difference penalty matrix.

    Raises
    ------
    ValueError
        Raised if `diff_order` is greater or equal to `data_size`.

    Notes
    -----
    Equivalent to calling::

        from pybaselines.utils import difference_matrix
        diff_matrix = difference_matrix(data_size, diff_order)
        penalty_matrix = diff_matrix.T @ diff_matrix

    but should be faster since the bands within the penalty matrix can be gotten
    without the matrix multiplication.

    """
    if data_size <= diff_order:
        raise ValueError('data size must be greater than or equal to the difference order.')
    penalty_bands = diff_penalty_diagonals(data_size, diff_order, lower_only=False)
    penalty_matrix = dia_object(
        (penalty_bands, np.arange(diff_order, -diff_order - 1, -1)), shape=(data_size, data_size),
    ).asformat(diff_format)

    return penalty_matrix


def _pentapy_solver(ab, y, check_output=False, pentapy_solver=2):
    """
    Convenience function for calling pentapy's solver with defaults already set.

    Solves the linear system :math:`A @ x = y` for `x`, given the matrix `A` in
    banded format, `ab`. The default settings of :func`:pentapy.solve` are
    already set for the fastest configuration.

    Parameters
    ----------
    ab : array-like, shape (M, N)
        The matrix `A` in row-wise banded format (see :func:`pentapy.solve`).
    y : array-like, shape (N,)
        The right-hand side of the equation.
    check_output : bool, optional
        If True, will check the output solution for non-finite values, which can often
        occur without warning from the solver when all values are close to zero.
        Default is False.
    pentapy_solver : int or str, optional
        The integer or string designating which solver to use if using pentapy. See
        :func:`pentapy.solve` for available options, although `1` or `2` are the
        most relevant options. Default is 2.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The solution to the linear system.

    """
    output = _pentapy_solve(ab, y, is_flat=True, index_row_wise=True, solver=pentapy_solver)
    if check_output and not np.isfinite(output.dot(output)):
        raise np.linalg.LinAlgError('non-finite value encountered in pentapy solver output')

    return output


class PenalizedSystem:
    """
    An object for setting up and solving banded penalized least squares linear systems.

    Attributes
    ----------
    diff_order : int
        The difference order of the penalty.
    lower : bool
        If True, the penalty uses only the lower bands of the symmetric banded penalty. Will
        use :func:`scipy.linalg.solveh_banded` for solving. If False, contains both the upper
        and lower bands of the penalty and will use either :func:`scipy.linalg.solve_banded`
        (if `using_pentapy` is False) or :func:`._pentapy_solver` when solving.
    main_diagonal : numpy.ndarray
        The main diagonal of the penalty matrix.  Is updated when adding additional matrices
        to the penalty through :meth:`.add_penalty`, and takes into account whether the penalty
        is only the lower bands or the total bands.
    main_diagonal_index : int
        The index of the main diagonal for `penalty`. Is updated when adding additional matrices
        to the penalty through :meth:`.add_penalty`, and takes into account whether the penalty
        is only the lower bands or the total bands.
    num_bands : int
        The number of bands in the penalty. The number of bands is assumbed to be symmetric,
        so the number of upper and lower bands should both be equal to `num_bands`.
    original_diagonals : numpy.ndarray
        The original penalty diagonals before multiplying by `lam` or adding any padding.
        Maintained so that repeated computations with different `lam` values can be quickly
        set up. `original_diagonals` can be either the full or lower bands of the penalty,
        and may be reveresed, it depends on the set up. Reset by calling
        :meth:`~PenalizedSystem.reset_diagonals`.
    penalty : numpy.ndarray
        The current penalty. Originally is `original_diagonals` after multiplying by `lam`
        and applying padding, but can also be changed by calling
        :meth:`~PenalizedSystem.add_penalty`.
        Reset by calling :meth:`~PenalizedSystem.reset_diagonals`.
    pentapy_solver : int or str
        The integer or string designating which solver to use if using pentapy. See
        :func:`pentapy.solve` for available options, although `1` or `2` are the
        most relevant options. Default is 2.
    reversed : bool
        If True, the penalty is reversed of the typical LAPACK banded format. Useful if
        multiplying the penalty with an array since the rows get shifted, or if using pentapy's
        solver.
    using_pentapy : bool
        If True, will use pentapy's solver when solving.

    """

    def __init__(self, data_size, lam=1, diff_order=2, allow_lower=True,
                 reverse_diags=None, allow_pentapy=True, padding=0, pentapy_solver=2):
        """
        Initializes the banded system.

        Parameters
        ----------
        data_size : int
            The number of data points for the system.
        lam : float, optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int, optional
            The difference order of the penalty. Default is 2 (second order difference).
        allow_lower : bool, optional
            If True (default), will allow only using the lower bands of the penalty matrix,
            which allows using :func:`scipy.linalg.solveh_banded` instead of the slightly
            slower :func:`scipy.linalg.solve_banded`.
        reverse_diags : {None, False, True}, optional
            If True, will reverse the order of the diagonals of the squared difference
            matrix. If False, will never reverse the diagonals. If None (default), will
            only reverse the diagonals if using pentapy's solver.
        allow_pentapy : bool, optional
            If True (default), will allow using pentapy's solver if `diff_order` is 2
            and pentapy is installed. pentapy's solver is faster than scipy's banded solvers.
        padding : int, optional
            The number of extra layers of zeros to add to the bottom and potentially
            the top if the full bands are used. Default is 0, which adds no extra
            layers. Negative `padding` is treated as equivalent to 0.
        pentapy_solver : int or str, optional
            The integer or string designating which solver to use if using pentapy. See
            :func:`pentapy.core.solve` for available options, although `1` or `2` are the
            most relevant options. Default is 2.

        """
        self._len = data_size
        self.original_diagonals = None
        self.pentapy_solver = pentapy_solver

        self.reset_diagonals(
            lam, diff_order, allow_lower, reverse_diags, allow_pentapy, padding=padding
        )

    def add_penalty(self, penalty):
        """
        Updates `self.penalty` with an additional penalty and updates the bands.

        Parameters
        ----------
        penalty : array-like
            The additional penalty to add to `self.penalty`.

        Returns
        -------
        numpy.ndarray
            The updated `self.penalty`.

        """
        self.penalty = _add_diagonals(self.penalty, penalty, lower_only=self.lower)
        self._update_bands()
        return self.penalty

    def _update_bands(self):
        """Updates the number of bands and the index of the main diagonal in `self.penalty`."""
        if self.lower:
            self.num_bands = len(self.penalty) - 1
        else:
            self.num_bands = len(self.penalty) // 2
        self.main_diagonal_index = 0 if self.lower else self.num_bands
        self.main_diagonal = self.penalty[self.main_diagonal_index].copy()

    def add_diagonal(self, value):
        """
        Adds a diagonal array or float to the original penalty matrix.

        Parameters
        ----------
        value : float or numpy.ndarray
            The number or array to add to the main diagonal of the penalty.

        Returns
        -------
        numpy.ndarray
            The penalty with the main diagonal updated.

        """
        self.penalty[self.main_diagonal_index] = self.main_diagonal + value
        return self.penalty

    def reset_diagonals(self, lam=1, diff_order=2, allow_lower=True, reverse_diags=None,
                        allow_pentapy=True, padding=0):
        """
        Resets the diagonals of the system and all of the attributes.

        Useful for reusing the penalized system for a different `lam` value.

        Parameters
        ----------
        lam : float, optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int, optional
            The difference order of the penalty. Default is 2 (second order difference).
        allow_lower : bool, optional
            If True (default), will allow only using the lower bands of the penalty matrix,
            which allows using :func:`scipy.linalg.solveh_banded` instead of the slightly
            slower :func:`scipy.linalg.solve_banded`.
        reverse_diags : {None, False, True}, optional
            If True, will reverse the order of the diagonals of the squared difference
            matrix. If False, will never reverse the diagonals. If None (default), will
            only reverse the diagonals if using pentapy's solver.
        allow_pentapy : bool, optional
            If True (default), will allow using pentapy's solver if `diff_order` is 2
            and pentapy is installed. pentapy's solver is faster than scipy's banded solvers.
        padding : int, optional
            The number of extra layers of zeros to add to the bottom and potentially
            the top if the full bands are used. Default is 0, which adds no extra
            layers. Negative `padding` is treated as equivalent to 0.

        """
        using_pentapy = allow_pentapy and _HAS_PENTAPY and diff_order == 2
        if allow_lower and not using_pentapy:
            lower_only = True
        else:
            lower_only = False
        if reverse_diags or (using_pentapy and reverse_diags is None):
            needs_reversed = True
        else:
            needs_reversed = False

        if self.original_diagonals is None or self.diff_order != diff_order:
            diagonal_data = diff_penalty_diagonals(self._len, diff_order, lower_only)
            if needs_reversed:
                diagonal_data = diagonal_data[::-1]
            self.original_diagonals = diagonal_data
        else:
            if self.lower and not lower_only:
                self.original_diagonals = _lower_to_full(self.original_diagonals)
            if (self.reversed and not needs_reversed) or (not self.reversed and needs_reversed):
                self.original_diagonals = self.original_diagonals[::-1]
            if not self.lower and lower_only:
                self.original_diagonals = self.original_diagonals[self.diff_order:]

        self.diff_order = diff_order
        self.lower = lower_only
        self.using_pentapy = using_pentapy
        self.reversed = needs_reversed

        self.lam = _check_lam(lam, allow_zero=False)
        self.penalty = self.lam * _pad_diagonals(self.original_diagonals, padding, self.lower)
        self._update_bands()

    def solve(self, lhs, rhs, overwrite_ab=False, overwrite_b=False,
              check_finite=False, l_and_u=None, check_output=False):
        """
        Solves the equation ``A @ x = rhs``, given `A` in banded format as `lhs`.

        Parameters
        ----------
        lhs : array-like, shape (M, N)
            The left-hand side of the equation, in banded format. `lhs` is assumed to be
            some slight modification of `self.penalty` in the same format (reversed, lower,
            number of bands, etc. are all the same).
        rhs : array-like, shape (N,)
            The right-hand side of the equation.
        overwrite_ab : bool, optional
            Whether to overwrite `lhs` when using :func:`scipy.linalg.solveh_banded` or
            :func:`scipy.linalg.solve_banded`. Default is False.
        overwrite_b : bool, optional
            Whether to overwrite `rhs` when using :func:`scipy.linalg.solveh_banded` or
            :func:`scipy.linalg.solve_banded`. Default is False.
        check_finite : bool, optional
            Whether to check if the inputs are finite when using
            :func:`scipy.linalg.solveh_banded` or :func:`scipy.linalg.solve_banded`.
            Default is False.
        l_and_u : Container(int, int), optional
            The number of lower and upper bands in `lhs` when using
            :func:`scipy.linalg.solve_banded`. Default is None, which uses
            (``len(lhs) // 2``, ``len(lhs) // 2``).
        check_output : bool, optional
            If True, will check the output for non-finite values when using
            :func:`._pentapy_solver`. Default is False.

        Returns
        -------
        output : numpy.ndarray, shape (N,)
            The solution to the linear system, `x`.

        """
        if self.using_pentapy:
            output = _pentapy_solver(
                lhs, rhs, check_output=check_output, pentapy_solver=self.pentapy_solver
            )
        elif self.lower:
            output = solveh_banded(
                lhs, rhs, overwrite_ab=overwrite_ab,
                overwrite_b=overwrite_b, lower=True, check_finite=check_finite
            )
        else:
            if l_and_u is None:
                num_bands = len(lhs) // 2
                l_and_u = (num_bands, num_bands)
            output = solve_banded(
                l_and_u, lhs, rhs, overwrite_ab=overwrite_ab,
                overwrite_b=overwrite_b, check_finite=check_finite
            )

        return output

    def reverse_penalty(self):
        """
        Reverses the penalty and original diagonals for the system.

        Raises
        ------
        ValueError
            Raised if `self.lower` is True, since reversing the half diagonals does
            not make physical sense.

        """
        if self.lower:
            raise ValueError('cannot reverse diagonals when self.lower is True')
        self.penalty = self.penalty[::-1]
        self.original_diagonals = self.original_diagonals[::-1]
        self.reversed = not self.reversed
