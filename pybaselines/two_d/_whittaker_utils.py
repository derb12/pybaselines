# -*- coding: utf-8 -*-
"""Helper functions for working with penalized linear systems.

Created on April 30, 2023
@author: Donald Erb

"""

import warnings

import numpy as np
from scipy.linalg import eig_banded, eigh_tridiagonal, solve
from scipy.sparse import kron
from scipy.sparse.linalg import spsolve

from .._banded_utils import diff_penalty_diagonals, diff_penalty_matrix
from .._compat import identity
from .._validation import _check_lam, _check_scalar, _check_scalar_variable
from ..utils import ParameterWarning


def _face_splitting(basis):
    """
    Performs the face-splitting product on the input two dimensional basis matrix.

    Parameters
    ----------
    basis : numpy.ndarray or scipy.sparse.spmatrix or scipy.sparse._sparray
        The two dimensional dense or sparse matrix, with shape (`M`, `N`).

    Returns
    -------
    scipy.sparse.spmatrix or scipy.sparse._sparray
        The face-splitting product of the input basis matrix with itself, with
        shape (`M`, `N**2`).

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    https://en.wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Face-splitting_product

    """
    ones = np.ones((1, basis.shape[1]))
    return kron(basis, ones).multiply(kron(ones, basis))


class PenalizedSystem2D:
    """
    An object for setting up and solving penalized least squares linear systems.

    Attributes
    ----------
    diff_order : numpy.ndarray[int, int]
        The difference order of the penalty.
    main_diagonal : numpy.ndarray
        The values along the main diagonal of the penalty matrix.
    penalty : scipy.sparse.base.spmatrix
        The current penalty. Originally is `original_diagonals` after multiplying by `lam`
        and applying padding, but can also be changed by calling
        :meth:`~PenalizedSystem2D.add_penalty`. Reset by calling
        :meth:`~PenalizedSystem2D.reset_diagonals`.

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

    def __init__(self, data_size, lam=1, diff_order=2):
        """
        Initializes the banded system.

        Parameters
        ----------
        data_size : Sequence[int, int]
            The number of data points for the system.
        lam : float or Sequence[int, int], optional
            The penalty factor applied to the difference matrix for the rows and columns,
            respectively. If a single value is given, both will use the same value. Larger
            values produce smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty for the rows and columns, respectively. If
            a single value is given, both will use the same value.
            Default is 2 (second order difference).

        """
        self._num_bases = data_size
        self.reset_diagonals(lam, diff_order)

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
        self.penalty = self.penalty + penalty
        self._update_bands()

        return self.penalty

    def _update_bands(self):
        """
        Updates the number of bands and the index of the main diagonal in `self.penalty`.

        Only relevant if setup as a banded matrix.

        """
        self.main_diagonal = self.penalty.diagonal()

    def reset_diagonals(self, lam=1, diff_order=2):
        """
        Resets the diagonals of the system and all of the attributes.

        Useful for reusing the penalized system for a different `lam` value.

        Parameters
        ----------
        lam : float or Sequence[int, int], optional
            The penalty factor applied to the difference matrix for the rows and columns,
            respectively. If a single value is given, both will use the same value. Larger
            values produce smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty for the rows and columns, respectively. If
            a single value is given, both will use the same value.
            Default is 2 (second order difference).

        """
        self.diff_order = _check_scalar_variable(
            diff_order, allow_zero=False, variable_name='difference order', two_d=True, dtype=int
        )
        self.lam = _check_lam(lam, two_d=True)

        penalty_rows = diff_penalty_matrix(self._num_bases[0], self.diff_order[0])
        penalty_columns = diff_penalty_matrix(self._num_bases[1], self.diff_order[1])

        # multiplying lam by the Kronecker product is the same as multiplying just D.T @ D with lam
        P_rows = kron(self.lam[0] * penalty_rows, identity(self._num_bases[1]))
        P_columns = kron(identity(self._num_bases[0]), self.lam[1] * penalty_columns)
        self.penalty = P_rows + P_columns

        self._update_bands()

    def solve(self, y, weights, penalty=None, rhs_extra=None):
        """
        Solves the equation ``A @ x = b``.

        Parameters
        ----------
        y : numpy.ndarray
            The y-values for fitting the spline.
        weights : numpy.ndarray
            The weights for each y-value. Will also be added to the diagonal of the
            penalty.
        penalty : numpy.ndarray
            The penalty to use for solving. Default is None which uses the object's
            penalty.
        rhs_extra : float or numpy.ndarray, optional
            If supplied, `rhs_extra` will be added to the right hand side
            of the equation before solving. Default is None, which adds nothing.

        Returns
        -------
        numpy.ndarray, shape (N,)
            The solution to the linear system, `x`.

        """
        if penalty is None:
            lhs = self.add_diagonal(weights)
        else:
            penalty.setdiag(penalty.diagonal() + weights)
            lhs = penalty
        rhs = weights * y
        if rhs_extra is not None:
            rhs = rhs + rhs_extra

        return self.direct_solve(lhs, rhs)

    def direct_solve(self, lhs, rhs):
        return spsolve(lhs, rhs)

    def add_diagonal(self, value):
        """
        Adds a diagonal array to the original penalty matrix.

        Parameters
        ----------
        value : float or numpy.ndarray
            The diagonal array to add to the penalty matrix.

        Returns
        -------
        scipy.sparse.base.spmatrix
            The penalty matrix with the main diagonal updated.

        """
        self.penalty.setdiag(self.main_diagonal + value)
        return self.penalty

    def reset_diagonal(self):
        """Sets the main diagonal of the penalty matrix back to its original value."""
        self.penalty.setdiag(self.main_diagonal)


class WhittakerSystem2D(PenalizedSystem2D):
    """
    Sets up and solves Whittaker smoothing using the analytical solution or eigendecomposition.

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

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
    (Preprint), 2023.

    """

    def __init__(self, data_size, lam=1, diff_order=2, num_eigens=None):
        """
        Initializes the penalized spline by calculating the basis and penalty.

        Parameters
        ----------
        data_size : Sequence[int, int]
            The number of data points for the system.
        lam : float or Sequence[int, int], optional
            The penalty factor applied to the difference matrix for the rows and columns,
            respectively. If a single value is given, both will use the same value. Larger
            values produce smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty for the rows and columns, respectively. If
            a single value is given, both will use the same value.
            Default is 2 (second order difference).
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for the eigendecomposition. If None, will solve the linear system using the full
            analytical solution, which is typically much slower.

        """
        # TODO should figure out a way to better merge PenalizedSystem2D, PSpline2D, and this class
        self.coef = None
        self._basis = None
        self._num_points = data_size
        num_eigens = _check_scalar(num_eigens, 2, fill_scalar=True)[0]
        if (num_eigens == np.array([None, None])).all():
            self._num_bases = data_size
            self._using_svd = False
        elif None in num_eigens:
            raise ValueError('eigenvalues must be None or non-None integers')
        else:
            self._num_bases = _check_scalar_variable(
                num_eigens, allow_zero=False, variable_name='eigenvalues', two_d=True, dtype=int
            )
            self._using_svd = True
        self.reset_diagonals(lam, diff_order)

    def reset_diagonals(self, lam=1, diff_order=2):
        """
        Resets the diagonals of the system and all of the attributes.

        Useful for reusing the penalized system for a different `lam` value.

        Parameters
        ----------
        lam : float or Sequence[int, int], optional
            The penalty factor applied to the difference matrix for the rows and columns,
            respectively. If a single value is given, both will use the same value. Larger
            values produce smoother results. Must be greater than 0. Default is 1.
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty for the rows and columns, respectively. If
            a single value is given, both will use the same value.
            Default is 2 (second order difference).

        """
        if not self._using_svd:
            super().reset_diagonals(lam, diff_order)
            return

        self.diff_order = _check_scalar_variable(
            diff_order, allow_zero=False, variable_name='difference order', two_d=True, dtype=int
        )
        self.lam = _check_lam(lam, two_d=True)

        # initially need num_bases to point to the data shape; maybe set a second
        # attribute insteaad
        values_rows, vectors_rows = self._calc_eigenvalues(
            self._num_points[0], self.diff_order[0], self._num_bases[0]
        )
        # TODO if all else matches, just calc the max eigens and use indexing for the lower one
        if (
            self.diff_order[0] == self.diff_order[1]
            and self._num_points[0] == self._num_points[1]
            and self._num_bases[0] == self._num_bases[1]
        ):
            values_columns, vectors_columns = values_rows, vectors_rows
        else:
            values_columns, vectors_columns = self._calc_eigenvalues(
                self._num_points[1], self.diff_order[1], self._num_bases[1]
            )
        # the eigenvalues are a diagonal matrix, so can simplify since
        # kron(diagonal, identity(N)) == np.repeat(diagonal, N) and
        # kron(identity(M), diaonal2) == np.tile(diagonal2, M)
        self.penalty_rows = np.repeat(self.lam[0] * values_rows, self._num_bases[1])
        self.penalty_columns = np.tile(self.lam[1] * values_columns, self._num_bases[0])
        # penalty is a (_num_bases[0] * _num_bases[1],) array
        self.penalty = self.penalty_rows + self.penalty_columns

        self.basis_r = vectors_rows
        self.basis_c = vectors_columns

        # TODO how much time is save by precomputaing G_r and G_c rather than computing
        # each iteration in solve? -> worth the memory usage?
        self._G_r = _face_splitting(self.basis_r)
        self._G_c = _face_splitting(self.basis_c)

    def _calc_eigenvalues(self, data_points, diff_order, num_eigens):
        """
        Calculate the eigenvalues and eigenvectors for the corresponding penalty matrix.

        Parameters
        ----------
        data_points : int
            The number of rows and columns of the square penalty matrix.
        diff_order : int
            The difference order of the penalty.
        num_eigens : int
            The number of smallest eigenvalues that will be used to represent the penalty matrix.

        Returns
        -------
        eigenvalues : np.ndarray, shape (`num_eigens`,)
            The eigenvalues of the penalty matrix for the corresponding difference order.
        eigenvectors : np.ndarray, shape (`data_points`, `num_eigens`)
            The eigenvectors for the penalty matrix.

        Raises
        ------
        ValueError
            Raised if the number of eigenvalues is greater than the number of data
            points.

        Warns
        -----
        ParameterWarning
            If `num_eigens` is less than or equal to `diff_order`, a warning is issue since
            the diagonals of the resulting matrix will no longer be guaranteed to be
            positive-definite. Is also emitted if `num_eigens` is greater than 50 since
            for 2D baseline correction, less than 20 eigenvalues is typically required.

        Notes
        -----
        The lowest `diff_order` eigenvalues are supposed to be zero while they end up
        being ~ +- 1e-15, so their values are set to 0.

        The penalty matrix has a matrix rank (number of nonzero eigenvalues) of
        ``data_points - diff_order``. The lowest `diff_order` eigenvalues are all
        zero, so the system is not guaranteed to be positive definite when solving the
        penalized least squares fit unless all weights are >~ 1e-5 (just a guess, but
        the meaning is that weights must be some magnitude greater than zero), which is
        not guaranteed for all Whittaker-smoothing-based algorithms. Thus, a clear
        warning needs to be issued since otherwise this detail can be hidden.

        References
        ----------
        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        penalty_bands = diff_penalty_diagonals(data_points, diff_order, lower_only=True)
        if num_eigens > data_points:
            raise ValueError((
                'The maximum number of eigenvalues cannot be greater '
                'than the number of data points.'
            ))
        elif num_eigens <= diff_order:
            warnings.warn(
                ('Setting the number of eigenvalues to be greater than the difference order '
                 'in order to not cause numerical instability'), ParameterWarning, stacklevel=2
            )
        elif num_eigens > 50:
            warnings.warn(
                ('For 2D baseline correction, typically only 5-20 eigenvalues are required to '
                 'fully approximate the baseline, and higher values will cause signifcant '
                 'slowdown'), ParameterWarning, stacklevel=2
            )

        if diff_order == 1:
            eigenvalues, eigenvectors = eigh_tridiagonal(
                penalty_bands[0], penalty_bands[1, :-1], select='i',
                select_range=(0, num_eigens - 1)
            )
        else:
            eigenvalues, eigenvectors = eig_banded(
                penalty_bands, lower=True, select='i',
                select_range=(0, num_eigens - 1), overwrite_a_band=True
            )

        # TODO do the corresponding eigenvectors in eigenvectors[:, :diff_order] need updated
        # too to match the resetting of the eigenvalues?
        eigenvalues[:diff_order] = 0

        return eigenvalues, eigenvectors

    def update_penalty(self, lam):
        if not self._using_svd:
            raise ValueError('Must call reset_diagonals if not using eigendecomposition')
        lam = _check_lam(lam, two_d=True)
        self.penalty_rows = (lam[0] / self.lam[0]) * self.penalty_rows
        self.penalty_columns = (lam[1] / self.lam[1]) * self.penalty_columns

        self.lam = lam
        self.penalty = self.penalty_rows + self.penalty_columns

    def same_basis(self, diff_order=2, num_eigens=None):
        """
        Sees if the current basis is equivalent to the input number of eigenvalues and diff order.

        Always returns False if the previous setup did not use eigendecomposition or if
        the input maximum number of eigenvalues is None.

        Parameters
        ----------
        diff_order : int or Sequence[int, int], optional
            The difference order of the penalty for the rows and columns, respectively. If
            a single value is given, both will use the same value.
            Default is 2 (second order difference).
        num_eigens : int or Sequence[int, int] or None
            The number of eigenvalues for the rows and columns, respectively, to use
            for the eigendecomposition. If None, will solve the linear system using the full
            analytical solution, which is typically much slower.

        Returns
        -------
        bool
            True if the input number of eigenvalues and difference order are equivalent to the
            current setup for the object.

        """
        # TODO should give a way to update only one of the basis functions, which
        # would also need to update the penalty
        num_eigens = _check_scalar(num_eigens, 2, fill_scalar=True)[0]
        if (num_eigens == np.array([None, None])).all() or not self._using_svd:
            return False

        diff_order = _check_scalar_variable(
            diff_order, allow_zero=False, variable_name='difference order', two_d=True, dtype=int
        )

        num_eigens = _check_scalar_variable(
            num_eigens, allow_zero=False, variable_name='eigenvalues', two_d=True, dtype=int
        )
        return (
            np.array_equal(diff_order, self.diff_order)
            and np.array_equal(num_eigens, self._num_bases)
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
        # TODO is this even needed?
        self.reset_diagonals(lam, diff_order)

    def _make_btwb(self, weights):
        """Computes ``Basis.T @ Weights @ Basis`` using a more efficient method.

        References
        ----------
        Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
        Statistics and Data Analysis, 2006, 50(1), 61-76.

        """
        # do not save intermediate results since they are memory intensive for high number of bases
        # note to self: F is not sparse when the basis functions are eigenvectors since the
        # eigenvector matrices are fully dense; it is however symmetric and positive definite
        F = np.transpose(
                (self._G_r.T @ weights @ self._G_c).reshape(
                    (self._num_bases[0], self._num_bases[0], self._num_bases[1], self._num_bases[1])
                ),
                [0, 2, 1, 3]
            ).reshape(
                (self._num_bases[0] * self._num_bases[1], self._num_bases[0] * self._num_bases[1])
        )

        return F

    def solve(self, y, weights, penalty=None, rhs_extra=None, assume_a='pos'):
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
        if not self._using_svd:
            return super().solve(y, weights, penalty, rhs_extra)

        rhs = (self.basis_r.T @ (weights * y) @ self.basis_c).ravel()
        if rhs_extra is not None:
            rhs = rhs + rhs_extra

        if penalty is None:
            penalty = self.penalty

        lhs = self._make_btwb(weights)
        # TODO could use cho_factor and save the factorization to call within _calc_dof to make
        # the call save time since it would only be used after the weights are finalized -> would
        # only be valid if assume_a is 'pos', which all current methods are but in the future that
        # may not be guaranteed; better to be explicit and keep it as two separate steps
        np.fill_diagonal(lhs, lhs.diagonal() + penalty)
        self.coef = solve(
            lhs, rhs, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False,
            assume_a=assume_a
        )

        output = self.basis_r @ self.coef.reshape(self._num_bases) @ self.basis_c.T

        return output

    @property
    def basis(self):
        """
        The full spline basis matrix.

        This is a lazy implementation since the full basis is typically not needed for
        computations.

        """
        if not self._using_svd:
            # Could maybe just make a basis using identities? But this should not be called
            # from outside so no reason to implement
            raise ValueError('No basis matrix when not using eigendecomposition')

        if self._basis is None:
            self._basis = kron(self.basis_r, self.basis_c)
        return self._basis

    def _calc_dof(self, weights, assume_a='pos'):
        """
        Calculates the effective degrees of freedom for each eigenvalue.

        Parameters
        ----------
        weights : numpy.ndarray
            The weights array.
        assume_a : str, optional
            A string describing the nature of the penalized system. See :func:`scipy.linalg.solve`
            for valid inputs. Default is 'pos'.

        Returns
        -------
        numpy.ndarray, shape (num_eigens[0], num_eigens[1])
            The effective degrees of freedom associated with each eigenvalue.

        Raises
        ------
        ValueError
            Raised if the WhittakerSystem2D object is not using eigendecomposition (was
            initialized with ``num_eigens=None``).

        References
        ----------
        Biessy, G. Revisiting Whittaker-Henderson Smoothing. https://hal.science/hal-04124043
        (Preprint), 2023.

        """
        if not self._using_svd:
            # Could maybe just output a matrix of ones?
            raise ValueError(
                'Cannot calculate degrees of freedom when not using eigendecomposition'
            )
        lhs = self._make_btwb(weights)
        rhs = lhs.copy()
        np.fill_diagonal(lhs, lhs.diagonal() + self.penalty)
        dof = solve(
            lhs, rhs, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False,
            assume_a=assume_a
        )

        return dof.diagonal().reshape(self._num_bases)
