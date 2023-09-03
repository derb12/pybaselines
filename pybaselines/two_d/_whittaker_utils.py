# -*- coding: utf-8 -*-
"""Helper functions for working with penalized linear systems.

Created on April 30, 2023
@author: Donald Erb

"""

from scipy.sparse import identity, kron
from scipy.sparse.linalg import spsolve

from .._banded_utils import difference_matrix
from .._validation import _check_lam, _check_scalar


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
    penalty : numpy.ndarray
        The current penalty. Originally is `original_diagonals` after multiplying by `lam`
        and applying padding, but can also be changed by calling :meth:`.add_penalty`.
        Reset by calling :meth:`.reset_diagonals`.
    reversed : bool
        If True, the penalty is reversed of the typical LAPACK banded format. Useful if
        multiplying the penalty with an array since the rows get shifted, or if using pentapy's
        solver.
    using_pentapy : bool
        If True, will use pentapy's solver when solving.

    """

    def __init__(self, data_size, lam=1, diff_order=2, allow_lower=True,
                 reverse_diags=None, allow_pentapy=True, padding=0):
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

        """
        self.shape = data_size
        self.original_diagonals = None

        self.diff_order = _check_scalar(diff_order, 2, True)[0]
        self.lam = [_check_lam(val) for val in _check_scalar(lam, 2, True)[0]]
        D1 = difference_matrix(self.shape[0], self.diff_order[0])
        D2 = difference_matrix(self.shape[1], self.diff_order[1])

        P1 = self.lam[0] * kron(D1.T @ D1, identity(self.shape[1]))
        P2 = self.lam[1] * kron(identity(self.shape[0]), D2.T @ D2)

        self.penalty = P1 + P2

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
        output = spsolve(lhs, rhs, permc_spec='NATURAL')

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
