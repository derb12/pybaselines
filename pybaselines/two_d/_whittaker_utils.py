# -*- coding: utf-8 -*-
"""Helper functions for working with penalized linear systems.

Created on April 30, 2023
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solve_banded, solveh_banded
from scipy.sparse import identity, kron, spdiags
from scipy.sparse.linalg import spsolve

from .._banded_utils import diff_penalty_diagonals
from .._validation import _check_lam, _check_scalar


def diff_penalty_matrix(data_size, diff_order=2):
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

    Returns
    -------
    penalty_matrix : scipy.sparse.base.spmatrix
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
    penalty_matrix = spdiags(
        penalty_bands, np.arange(diff_order, -diff_order - 1, -1), data_size, data_size
    )
    return penalty_matrix


class PenalizedSystem2D:
    """
    An object for setting up and solving penalized least squares linear systems.

    Attributes
    ----------
    diff_order : int
        The difference order of the penalty.
    lower : bool
        If True, the penalty uses only the lower bands of the symmetric banded penalty. Will
        use :func:`scipy.linalg.solveh_banded` for solving. If False, contains both the upper
        and lower bands of the penalty and will use either :func:`scipy.linalg.solve_banded`
        (if `using_pentapy` is False) or :func:`._pentapy_solver` when solving.
    main_diagonal_index : int
        The index of the main diagonal for `penalty`. Is updated when adding additional matrices
        to the penalty, and takes into account whether the penalty is only the lower bands or
        the total bands.
    num_bands : int
        The number of bands in the penalty. The number of bands is assumbed to be symmetric,
        so the number of upper and lower bands should both be equal to `num_bands`.
    original_diagonals : numpy.ndarray
        The original penalty diagonals before multiplying by `lam` or adding any padding.
        Maintained so that repeated computations with different `lam` values can be quickly
        set up. `original_diagonals` can be either the full or lower bands of the penalty,
        and may be reveresed, it depends on the set up. Reset by calling
        :meth:`.reset_diagonals`.
    penalty : scipy.sparse.base.spmatrix
        The current penalty. Originally is `original_diagonals` after multiplying by `lam`
        and applying padding, but can also be changed by calling :meth:`.add_penalty`.
        Reset by calling :meth:`.reset_diagonals`.

    Notes
    -----
    Setting up the linear system using banded matrices is faster, but the number of bands is
    actually quite large (`data_size[1]`) due to the Kronecker products, although only
    ``2 * diff_order[0] + 2 * diff_order[1] + 2`` bands are actually nonzero. Despite this, it is
    still significantly faster than using the sparse solver and does not use more memory as
    long as it is only lower banded.

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    """

    def __init__(self, data_size, lam=1, diff_order=2, use_banded=True, use_lower=True):
        """
        Initializes the banded system.

        Parameters
        ----------
        data_size : Sequence[int, int]
            The number of data points for the system.
        lam : float, optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int, optional
            The difference order of the penalty. Default is 2 (second order difference).
        use_banded : bool, optional
            If True (default), will do the setup for solving the system using banded
            matrices rather than sparse matrices.
        use_lower : bool, optional
            If True (default), will allow only using the lower bands of the penalty matrix,
            which allows using :func:`scipy.linalg.solveh_banded` instead of the slightly
            slower :func:`scipy.linalg.solve_banded`. Only relevant if `use_banded` is True.

        """
        self._num_bases = data_size
        self.diff_order = [-1, -1]
        self.lam = [-1, -1]

        self.reset_diagonals(lam, diff_order, use_banded, use_lower)

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
        raise NotImplementedError

    def _update_bands(self):
        """
        Updates the number of bands and the index of the main diagonal in `self.penalty`.

        Only relevant if setup as a banded matrix.

        """
        if self.banded:
            if self.lower:
                self.num_bands = self.penalty.shape[0] - 1
            else:
                self.num_bands = self.penalty.shape[0] // 2
            self.main_diagonal_index = 0 if self.lower else self.num_bands
            self.main_diagonal = self.penalty[self.main_diagonal_index].copy()

    def reset_diagonals(self, lam=1, diff_order=2, use_banded=True, use_lower=True):
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
        use_banded : bool, optional
            If True (default), will do the setup for solving the system using banded
            matrices rather than sparse matrices.

        """
        self.diff_order = _check_scalar(diff_order, 2, True)[0]
        self.lam = [_check_lam(val) for val in _check_scalar(lam, 2, True)[0]]
        self.lower = use_lower
        self.banded = use_banded
        if (self.diff_order < 1).any():
            raise ValueError('the difference order must be > 0')

        penalty_rows = diff_penalty_matrix(self._num_bases[0], self.diff_order[0])
        penalty_columns = diff_penalty_matrix(self._num_bases[1], self.diff_order[1])

        # multiplying lam by the Kronecker product is the same as multiplying just D.T @ D with lam
        P1 = kron(self.lam[0] * penalty_rows, identity(self._num_bases[1]))
        P2 = kron(identity(self._num_bases[0]), self.lam[1] * penalty_columns)
        penalty = P1 + P2
        if self.banded:
            penalty = penalty.todia()
            sparse_bands = (penalty).data
            offsets = penalty.offsets
            index_offset = np.max(offsets)
            penalty_bands = np.zeros((index_offset * 2 + 1, sparse_bands.shape[1]))
            for index, banded_index in enumerate(offsets):
                penalty_bands[abs(banded_index - index_offset)] = sparse_bands[index]
            self.penalty = penalty_bands
            if self.lower:
                self.penalty = self.penalty[self.penalty.shape[0] // 2:]
            self._update_bands()
        else:
            self.penalty = penalty
            self.main_diagonal = self.penalty.diagonal()
            self.main_diagonal_index = 0

    def solve(self, lhs, rhs, overwrite_ab=False, overwrite_b=False,
              check_finite=False, l_and_u=None):
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

        Returns
        -------
        output : numpy.ndarray, shape (N,)
            The solution to the linear system, `x`.

        """
        if self.banded:
            if self.lower:
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
        else:
            output = spsolve(lhs, rhs, permc_spec='NATURAL')

        return output

    def add_diagonal(self, value):
        """
        Adds a diagonal array to the original penalty matrix.

        Parameters
        ----------
        value : numpy.ndarray
            The diagonal array to add to the penalty matrix.

        Returns
        -------
        scipy.sparse.base.spmatrix
            The penalty matrix with the main diagonal updated.

        """
        if self.banded:
            self.penalty[self.main_diagonal_index] = self.main_diagonal + value
        else:
            self.penalty.setdiag(self.main_diagonal + value)
        return self.penalty

    def reset_diagonal(self):
        """Sets the main diagonal of the penalty matrix back to its original value."""
        if self.banded:
            self.penalty[self.main_diagonal_index] = self.main_diagonal
        else:
            self.penalty.setdiag(self.main_diagonal)

    def reverse_penalty(self):
        """
        Reverses the penalty and original diagonals for the system.

        Raises
        ------
        ValueError
            Raised if `self.lower` is True, since reversing the half diagonals does
            not make physical sense.

        """
        raise NotImplementedError

        if self.lower:
            raise ValueError('cannot reverse diagonals when self.lower is True')
        self.penalty = self.penalty[::-1]
        self.original_diagonals = self.original_diagonals[::-1]
        self.reversed = not self.reversed
