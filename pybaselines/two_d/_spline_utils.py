# -*- coding: utf-8 -*-
"""Helper functions for using splines.

Created on April 25, 2023
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import kron
from scipy.sparse.linalg import spsolve

from .._compat import csr_object
from .._spline_utils import _spline_basis, _spline_knots
from .._validation import _check_array, _check_scalar_variable
from ._whittaker_utils import PenalizedSystem2D, _face_splitting


class PSpline2D(PenalizedSystem2D):
    """
    A Penalized Spline, which penalizes the difference of the spline coefficients.

    Penalized splines (P-Splines) are solved with the following equation
    ``(B.T @ W @ B + P) c = B.T @ W @ y`` where `c` is the spline coefficients, `B` is the
    spline basis, the weights are the diagonal of `W`, the penalty is `P`, and `y` is the
    fit data. The penalty `P` is usually in the form ``lam * D.T @ D``, where `lam` is a
    penalty factor and `D` is the matrix version of the finite difference operator.

    Attributes
    ----------
    basis_r : scipy.sparse.csr.csr_matrix, shape (N, P)
        The spline basis for the rows. Has a shape of (`N,` `P`), where `N` is the number of
        points in `x`, and `P` is the number of basis functions (equal to ``K - spline_degree - 1``
        or equivalently ``num_knots[0] + spline_degree[0] - 1``).
    basis_c : scipy.sparse.csr.csr_matrix, shape (M, Q)
        The spline basis for the columns. Has a shape of (`M,` `Q`), where `M` is the number of
        points in `z`, and `Q` is the number of basis functions (equal to ``K - spline_degree - 1``
        or equivalently ``num_knots[1] + spline_degree[1] - 1``).
    coef : None or numpy.ndarray, shape (M,)
        The spline coefficients. Is None if :meth:`~PSpline2D.solve_pspline` has not been called
        at least once.
    knots_r : numpy.ndarray, shape (K,)
        The knots for the spline along the rows. Has a shape of `K`, which is equal to
        ``num_knots[0] + 2 * spline_degree[0]``.
    knots_c : numpy.ndarray, shape (L,)
        The knots for the spline along the columns. Has a shape of `L`, which is equal to
        ``num_knots[1] + 2 * spline_degree[2]``.
    num_knots : numpy.ndarray([int, int])
        The number of internal knots (including the endpoints) for x and z. The total number of
        knots for the spline, `K`, is equal to ``num_knots + 2 * spline_degree``.
    spline_degree : numpy.ndarray([int, int])
        The degree of the spline (eg. a cubic spline would have a `spline_degree` of 3) for
        x and z.
    x : numpy.ndarray, shape (N,)
        The x-values for the spline.
    z : numpy.ndarray, shape (M,)
        The z-values for the spline.

    Notes
    -----
    If the penalty is symmetric, the sparse system could be solved much faster using
    CHOLMOD from SuiteSparse (https://github.com/DrTimothyAldenDavis/SuiteSparse) through
    the python bindings provided by scikit-sparse (https://github.com/scikit-sparse/scikit-sparse),
    but it is not worth implementing here since this code will rarely be used.

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
        z : array-like, shape (M,)
            The z-values for the spline.
        num_knots : int or Sequence[int, int], optional
            The number of internal knots for the spline, including the endpoints.
            Default is 100.
        spline_degree : int or Sequence[int, int], optional
            The degree of the spline. Default is 3, which is a cubic spline.
        check_finite : bool, optional
            If True, will raise an error if any values in `x` are not finite. Default
            is False, which skips the check.
        lam : float or Sequence[float, float], optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty. Default is 2 (second order difference).

        Raises
        ------
        ValueError
            Raised if `spline_degree` is less than 0 or if `diff_order` is less than 1
            or greater than or equal to the number of spline basis functions
            (``num_knots + spline_degree - 1``).

        """
        self.coef = None
        self._basis = None

        self.x = _check_array(x, dtype=float, check_finite=check_finite, ensure_1d=True)
        self.z = _check_array(z, dtype=float, check_finite=check_finite, ensure_1d=True)

        self.num_knots = _check_scalar_variable(
            num_knots, allow_zero=False, variable_name='number of knots', two_d=True, dtype=int
        )
        self.spline_degree = _check_scalar_variable(
            spline_degree, allow_zero=True, variable_name='spline degree', two_d=True, dtype=int
        )

        self.knots_r = _spline_knots(self.x, self.num_knots[0], self.spline_degree[0], True)
        self.basis_r = _spline_basis(self.x, self.knots_r, self.spline_degree[0])

        self.knots_c = _spline_knots(self.z, self.num_knots[1], self.spline_degree[1], True)
        self.basis_c = _spline_basis(self.z, self.knots_c, self.spline_degree[1])

        super().__init__((self.basis_r.shape[1], self.basis_c.shape[1]), lam, diff_order)

        if (self.diff_order >= self._num_bases).any():
            raise ValueError((
                'the difference order must be less than the number of basis '
                'functions, which is the number of knots + spline degree - 1'
            ))

        # TODO how much time is save by precomputaing G_r and G_c rather than computing
        # each iteration in solve? -> worth the memory usage?
        self._G_r = _face_splitting(self.basis_r)
        self._G_c = _face_splitting(self.basis_c)

    def same_basis(self, num_knots=100, spline_degree=3):
        """
        Sees if the current basis is equivalent to the input number of knots of spline degree.

        Parameters
        ----------
        num_knots : int or Sequence[int, int], optional
            The number of knots for the new spline. Default is 100.
        spline_degree : int or Sequence[int, int], optional
            The degree of the new spline. Default is 3.

        Returns
        -------
        bool
            True if the input number of knots and spline degree are equivalent to the current
            spline basis of the object.

        """
        # TODO should give a way to update only one of the basis functions, which
        # would also need to update the penalty
        num_knots = _check_scalar_variable(
            num_knots, allow_zero=False, variable_name='number of knots', two_d=True, dtype=int
        )
        spline_degree = _check_scalar_variable(
            spline_degree, allow_zero=True, variable_name='spline degree', two_d=True, dtype=int
        )

        return (
            np.array_equal(num_knots, self.num_knots)
            and np.array_equal(spline_degree, self.spline_degree)
        )

    def reset_penalty(self, lam=1, diff_order=2):
        """
        Resets the penalty of the system and all of the attributes.

        Useful for reusing the penalty diagonals without having to recalculate the spline basis.

        Parameters
        ----------
        lam : float or Sequence[float, float], optional
            The penalty factor applied to the difference matrix. Larger values produce
            smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty. Default is 2 (second order difference).

        """
        self.reset_diagonals(lam, diff_order)

    def solve(self, y, weights, penalty=None, rhs_extra=None):
        """
        Solves the coefficients for a weighted penalized spline.

        Solves the linear equation ``(B.T @ W @ B + P) c = B.T @ W @ y`` for the spline
        coefficients, `c`, given the spline basis, `B`, the weights (diagonal of `W`), the
        penalty `P`, and `y`, and returns the resulting spline, ``B @ c``. Attempts to
        calculate ``B.T @ W @ B`` and ``B.T @ W @ y`` as a banded system to speed up
        the calculation.

        Parameters
        ----------
        y : numpy.ndarray, shape (M, N)
            The y-values for fitting the spline.
        weights : numpy.ndarray, shape (M, N)
            The weights for each y-value.
        penalty : numpy.ndarray, shape (``M * N``, ``M * N``)
            The finite difference penalty matrix, in LAPACK's lower banded format (see
            :func:`scipy.linalg.solveh_banded`) if `lower_only` is True or the full banded
            format (see :func:`scipy.linalg.solve_banded`) if `lower_only` is False.
        rhs_extra : float or numpy.ndarray, shape (``M * N``,), optional
            If supplied, `rhs_extra` will be added to the right hand side (``B.T @ W @ y``)
            of the equation before solving. Default is None, which adds nothing.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The spline, corresponding to ``B @ c``, where `c` are the solved spline
            coefficients and `B` is the spline basis.

        Notes
        -----
        Uses the more efficient algorithm from Eilers's paper, although the memory usage
        is higher than the straigtforward method when the number of knots is high; however,
        it is significantly faster and memory efficient when the number of knots is lower,
        which will be the more typical use case.

        """
        # do not save intermediate results since they are memory intensive for high number of knots
        F = csr_object(
            np.transpose(
                (self._G_r.T @ weights @ self._G_c).reshape(
                    (self._num_bases[0], self._num_bases[0], self._num_bases[1], self._num_bases[1])
                ),
                [0, 2, 1, 3]
            ).reshape(
                (self._num_bases[0] * self._num_bases[1], self._num_bases[0] * self._num_bases[1])
            )
        )
        if penalty is None:
            penalty = self.penalty

        rhs = (self.basis_r.T @ (weights * y) @ self.basis_c).ravel()
        if rhs_extra is not None:
            rhs = rhs + rhs_extra

        self.coef = spsolve(F + penalty, rhs)
        output = self.basis_r @ self.coef.reshape(self._num_bases) @ self.basis_c.T

        return output

    @property
    def basis(self):
        """
        The full spline basis matrix.

        This is a lazy implementation since the full basis is typically not needed for
        computations.

        """
        if self._basis is None:
            self._basis = kron(self.basis_r, self.basis_c)
        return self._basis

    @property
    def tck(self):
        """
        The knots, spline coefficients, and spline degree to reconstruct the spline.

        Convenience function for easily reconstructing the last solved spline with outside
        modules, such as with SciPy's `NdBSpline`, to allow for other usages such as evaulating
        with different x- and z-values.

        Raises
        ------
        ValueError
            Raised if `solve_pspline` has not been called yet, meaning that the spline has not
            yet been constructed.

        Notes
        -----
        To use with :class:`scipy.interpolate.NdBSpline`, the setup would look like:

            from scipy.interpolate import NdBspline
            pspline = Pspline2D(x, z, ...)
            pspline_fit = pspline.solve(...)
            XZ = np.array(np.meshgrid(x, z)).T  # same as zipping the meshgrid and rearranging
            fit = NdBSpline(pspline.tck)(XZ)  # fit == pspline_fit

        """
        if self.coef is None:
            raise ValueError('No spline coefficients, need to call "solve_pspline" first.')
        return (
            (self.knots_r, self.knots_c), self.coef.reshape(self._num_bases), self.spline_degree
        )
