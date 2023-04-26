# -*- coding: utf-8 -*-
"""Helper functions for using splines.

Created on April 25, 2023
@author: Donald Erb

"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .._banded_utils import difference_matrix
from .._spline_utils import _spline_basis, _spline_knots
from ._validation import _check_array, _check_lam, _check_scalar


class PSpline2D:
    """
    A Penalized Spline, which penalizes the difference of the spline coefficients.

    Penalized splines (P-Splines) are solved with the following equation
    ``(B.T @ W @ B + P) c = B.T @ W @ y`` where `c` is the spline coefficients, `B` is the
    spline basis, the weights are the diagonal of `W`, the penalty is `P`, and `y` is the
    fit data. The penalty `P` is usually in the form ``lam * D.T @ D``, where `lam` is a
    penalty factor and `D` is the matrix version of the finite difference operator.

    Attributes
    ----------
    basis : scipy.sparse.csr.csr_matrix, shape (N, M)
        The spline basis. Has a shape of (`N,` `M`), where `N` is the number of points
        in `x`, and `M` is the number of basis functions (equal to ``K - spline_degree - 1``
        or equivalently ``num_knots + spline_degree - 1``).
    coef : None or numpy.ndarray, shape (M,)
        The spline coefficients. Is None if :meth:`.solve_pspline` has not been called
        at least once.
    knots : numpy.ndarray, shape (K,)
        The knots for the spline. Has a shape of `K`, which is equal to
        ``num_knots + 2 * spline_degree``.
    num_knots : int
        The number of internal knots (including the endpoints). The total number of knots
        for the spline, `K`, is equal to ``num_knots + 2 * spline_degree``.
    spline_degree : int
        The degree of the spline (eg. a cubic spline would have a `spline_degree` of 3).
    x : numpy.ndarray, shape (N,)
        The x-values for the spline.

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    """

    def __init__(self, x, z, num_knots=100, spline_degree=3, check_finite=False, lam=1,
                 diff_order=2):
        """
        Initializes the penalized spline by calculating the basis and penalty.

        Parameters
        ----------
        x : array-like, shape (N,)
            The x-values for the spline.
        z : array-like, shape (L,)
            The z-values for the spline.
        num_knots : int or Sequence(int, int), optional
            The number of internal knots for the spline, including the endpoints.
            Default is 100.
        spline_degree : int or Sequence(int, int), optional
            The degree of the spline. Default is 3, which is a cubic spline.
        check_finite : bool, optional
            If True, will raise an error if any values in `x` are not finite. Default
            is False, which skips the check.
        lam : float or Sequence(float, float), optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence(int, int), optional
            The difference order of the penalty. Default is 2 (second order difference).

        Raises
        ------
        ValueError
            Raised if `spline_degree` is less than 0 or if `diff_order` is less than 1
            or greater than or equal to the number of spline basis functions
            (``num_knots + spline_degree - 1``).

        """
        self.x = _check_array(x, dtype=float, check_finite=check_finite, ensure_1d=True)
        self.z = _check_array(z, dtype=float, check_finite=check_finite, ensure_1d=True)
        self.shape = (len(x), len(z))

        self.num_knots = _check_scalar(num_knots, 2, True)[0]
        self.diff_order = _check_scalar(diff_order, 2, True)[0]
        self.spline_degree = _check_scalar(spline_degree, 2, True)[0]
        self.lam = [_check_lam(val) for val in _check_scalar(lam, 2, True)[0]]

        self.knots_1 = _spline_knots(self.x, self.num_knots[0], self.spline_degree[0], True)
        self.basis_1 = _spline_basis(self.x, self.knots_1, self.spline_degree[0])

        self.knots_2 = _spline_knots(self.z, self.num_knots[1], self.spline_degree[1], True)
        self.basis_2 = _spline_basis(self.z, self.knots_2, self.spline_degree[1])
        self._num_bases = np.array([self.basis_1.shape[1], self.basis_2.shape[1]])

        self.basis = sparse.kron(self.basis_2, self.basis_1)
        self.coef = None

        D1 = difference_matrix(self._num_bases[0], self.diff_order[0])
        D2 = difference_matrix(self._num_bases[1], self.diff_order[1])

        P1 = self.lam[0] * sparse.kron(D1.T @ D1, sparse.identity(self._num_bases[1]))
        P2 = self.lam[1] * sparse.kron(sparse.identity(self._num_bases[0]), D2.T @ D2)
        self.penalty = P1 + P2

        if (self.diff_order >= self._num_bases).any():
            raise ValueError((
                'the difference order must be less than the number of basis '
                'functions, which is the number of knots + spline degree - 1'
            ))
        elif (self.spline_degree < 0).any():
            raise ValueError('spline degree must be greater than or equal to 0')

    def same_basis(self, num_knots=100, spline_degree=3):
        """
        Sees if the current basis is equivalent to the input number of knots of spline degree.

        Parameters
        ----------
        num_knots : int, optional
            The number of knots for the new spline. Default is 100.
        spline_degree : int, optional
            The degree of the new spline. Default is 3.

        Returns
        -------
        bool
            True if the input number of knots and spline degree are equivalent to the current
            spline basis of the object.

        """
        return False  # TODO will need to check both basis matrices

    def reset_penalty_diagonals(self, lam=1, diff_order=2, allow_lower=True, reverse_diags=False):
        """
        Resets the penalty diagonals of the system and all of the attributes.

        Useful for reusing the penalty diagonals without having to recalculate the spline basis.

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
        reverse_diags : bool, optional
            If True, will reverse the order of the diagonals of the squared difference
            matrix. If False (default), will never reverse the diagonals.

        Notes
        -----
        `allow_pentapy` is always set to False since the time needed to go from a lower to full
        banded matrix and shifting the rows removes any speedup from using pentapy's solver. It
        also reduces the complexity of setting up the equations.

        Adds padding to the penalty diagonals to accomodate the different shapes of the spline
        basis and the penalty to speed up calculations when the two are added.

        """

    def solve_pspline(self, y, weights, penalty=None, rhs_extra=None):
        """
        Solves the coefficients for a weighted penalized spline.

        Solves the linear equation ``(B.T @ W @ B + P) c = B.T @ W @ y`` for the spline
        coefficients, `c`, given the spline basis, `B`, the weights (diagonal of `W`), the
        penalty `P`, and `y`, and returns the resulting spline, ``B @ c``. Attempts to
        calculate ``B.T @ W @ B`` and ``B.T @ W @ y`` as a banded system to speed up
        the calculation.

        Parameters
        ----------
        y : numpy.ndarray, shape (N,)
            The y-values for fitting the spline.
        weights : numpy.ndarray, shape (N,)
            The weights for each y-value.
        penalty : numpy.ndarray, shape (D, N)
            The finite difference penalty matrix, in LAPACK's lower banded format (see
            :func:`scipy.linalg.solveh_banded`) if `lower_only` is True or the full banded
            format (see :func:`scipy.linalg.solve_banded`) if `lower_only` is False.
        rhs_extra : float or numpy.ndarray, shape (N,), optional
            If supplied, `rhs_extra` will be added to the right hand side (``B.T @ W @ y``)
            of the equation before solving. Default is None, which adds nothing.

        Returns
        -------
        numpy.ndarray, shape (N,)
            The spline, corresponding to ``B @ c``, where `c` are the solved spline
            coefficients and `B` is the spline basis.

        """
        # TODO investigate whether the other algorithm in Eilers's paper is more efficient
        # memory- or time-wise
        CWT = self.basis.multiply(
            np.repeat(
                weights, self._num_bases[0] * self._num_bases[1]
            ).reshape(self.shape[0] * self.shape[1], -1)
        ).T
        CWC = CWT @ self.basis
        CWy = CWT @ y

        self.coef = spsolve(CWC + self.penalty, CWy)

        return self.basis @ self.coef
