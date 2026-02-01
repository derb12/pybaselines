# -*- coding: utf-8 -*-
"""Objects for calculating additional terms from results of analytical baseline correction methods.

Created on November 15, 2025
@author: Donald Erb

"""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import issparse
from scipy.sparse.linalg import factorized

from ._banded_utils import _banded_to_sparse, _add_diagonals
from ._compat import diags, _sparse_col_index
from .utils import _get_rng


class WhittakerResult:
    """
    Represents the result of Whittaker smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    Whittaker smoothing. This class should not be initialized by external users.

    """

    def __init__(self, penalized_object, weights=None, lhs=None, rhs_extra=None):
        """
        Initializes the result object.

        In the most basic formulation, Whittaker smoothing solves ``(W + P) @ v = W @ y``.
        Then the hat matrix would be ``(W + P)^-1 @ W``. For more complex usages, the
        equation can be expressed as ``lhs @ v = (W + rhs_extra) @ y`` with a corresponding
        hat matrix of ``lhs^-1 @ (W + rhs_extra)``.

        Parameters
        ----------
        penalized_object : pybaselines._banded_utils.PenalizedSystem
            The penalized system object used for solving.
        weights : numpy.ndarray, shape (N,) optional
            The weights used to solve the system. Default is None, which will set
            all weights to 1.
        lhs : numpy.ndarray, optional
            The left hand side of the normal equation. Default is None, which will assume that
            `lhs` is the addition of ``diags(weights)`` and ``pentalized_object.penalty``.
        rhs_extra : numpy.ndarray or scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            Additional terms besides the weights within the right hand side of the hat matrix.
            Default is None.

        """
        self._penalized_object = penalized_object
        self._hat_lhs = lhs
        self._hat_rhs = None
        self._rhs_extra = rhs_extra
        self._trace = None
        if weights is None:
            weights = np.ones(self._shape)
        self._weights = weights

    @property
    def _shape(self):
        """The shape of the penalized system.

        Returns
        -------
        tuple[int, int]
            The penalized system's shape.

        """
        # TODO need to add an attribute to join 1D and 2D PenalizedSystem and PSpline objects
        # so that this can just access that attribute rather than having to modify for each
        # subclass
        return self._basis_shape

    @property
    def _size(self):
        """The total size of the penalized system.

        Returns
        -------
        int
            The penalized system's size.

        """
        return np.prod(self._shape)

    @property
    def _basis_shape(self):
        """The shape of the system's basis matrix.

        Returns
        -------
        tuple[int, int]
            The penalized system's basis shape.

        """
        return self._penalized_object._num_bases

    @property
    def _basis_size(self):
        """The total size of the system's basis matrix.

        Returns
        -------
        int
            The system's basis matrix size.

        """
        return np.prod(self._basis_shape)

    @property
    def _lhs(self):
        """
        The left hand side of the hat matrix in banded format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        numpy.ndarray
            The array representing the left hand side of the hat matrix.

        """
        if self._hat_lhs is None:
            self._hat_lhs = self._penalized_object.add_diagonal(self._weights)
        return self._hat_lhs

    @property
    def _rhs(self):
        """
        The right hand side of the hat matrix in sparse format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse object representing the right hand side of the hat matrix.

        """
        if self._hat_rhs is None:
            if self._rhs_extra is None:
                self._hat_rhs = diags(self._weights)
            else:
                if not issparse(self._rhs_extra):
                    self._rhs_extra = _banded_to_sparse(
                        self._rhs_extra, lower=self._penalized_object.lower
                    )
                self._rhs_extra.setdiag(self._rhs_extra.diagonal() + self._weights)
                self._hat_rhs = self._rhs_extra
        return self._hat_rhs

    def effective_dimension(self, n_samples=0, rng=1234):
        """
        Calculates the effective dimension from the trace of the hat matrix.

        For typical Whittaker smoothing, the linear equation would be
        ``(W + P) v = W @ y``. Then the hat matrix would be ``(W + P)^-1 @ W``.
        The effective dimension for the system can be estimated as the trace
        of the hat matrix.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (N, `n_samples`) Rademacher random variables
            (ie. either -1 or 1).
        rng : int or numpy.random.Generator or numpy.random.RandomState
            The integer for the seed of the random number generator or an existing generating
            object to use for the stochastic trace estimation.

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective dimension for
            the system.

        Raises
        ------
        TypeError
            Raised if `n_samples` is not an integer greater than or equal to 0.

        Notes
        -----
        For systems larger than ~1000 data points, it is heavily suggested to use stochastic
        trace estimation since the time required for the analytical solution calculation scales
        poorly with size.

        References
        ----------
        Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

        Hutchinson, M. A stochastic estimator of the trace of the influence matrix for laplacian
        smoothing splines. Communications in Statistics - Simulation and Computation, (1990),
        19(2), 433-450.

        Meyer, R., et al. Hutch++: Optimal Stochastic Trace Estimation. 2021 Symposium on
        Simplicity in Algorithms (SOSA), (2021), 142-155.

        """
        # NOTE if diff_order is 2 and matrix is symmetric, could use the fast trace calculation from
        # Frasso G, Eilers PH. L- and V-curves for optimal smoothing. Statistical Modelling.
        # (2014), 15(1), 91-111. https://doi.org/10.1177/1471082X14549288, which is in turn based on
        # Hutchinson, M, et al. Smoothing noisy data with spline functions. Numerische Mathematik.
        # (1985), 47, 99-106. https://doi.org/10.1007/BF01389878
        # For non-symmetric matrices, can use the slightly more involved algorithm from:
        # Erisman, A., et al. On Computing Certain Elements of the Inverse of a Sparse Matrix.
        # Communication of the ACM. (1975) 18(3), 177-179. https://doi.org/10.1145/360680.360704
        # -> worth the effort? -> maybe...? For diff_order=2 and symmetric lhs, the timing is
        # much faster than even the stochastic calculation and does not increase much with data
        # size, and it provides the exact trace rather than an estimate -> however, this is only
        # useful for GCV/BIC calculations atm, which are going to be very very rarely used -> could
        # allow calculating the full inverse hat diagonal to allow calculating the baseline fit
        # errors, but that's still incredibly niche...
        # Also note that doing so would require performing inv(lhs) @ rhs, which is typically less
        # numerically stable than solve(lhs, rhs) and would be complicated for non diagonal rhs;
        # as such, I'd rather not implement it and just leave the above for reference.

        # TODO could maybe make default n_samples to None and decide to use analytical or
        # stochastic trace based on data size; data size > 1000 use stochastic with default
        # n_samples = 100?
        if n_samples == 0:
            if self._trace is not None:
                return self._trace
            use_analytic = True
        else:
            if n_samples < 0 or not isinstance(n_samples, int):
                raise TypeError('n_samples must be a non-negative integer')
            use_analytic = False

        if use_analytic:
            # compute each diagonal of the hat matrix separately so that the full
            # hat matrix does not need to be stored in memory
            # note to self: sparse factorization is the worst case scenario (non-symmetric lhs and
            # diff_order != 2), but it is still much faster than individual solves through
            # solve_banded
            factorization = self._penalized_object.factorize(self._lhs)
            trace = 0
            if self._rhs_extra is None:
                # note: about an order of magnitude faster to omit the sparse rhs for the simple
                # case of lhs @ v = w * y
                eye = np.zeros(self._size)
                for i in range(self._size):
                    eye[i] = self._weights[i]
                    trace += self._penalized_object.factorized_solve(factorization, eye)[i]
                    eye[i] = 0
            else:
                rhs = self._rhs.tocsc()
                for i in range(self._basis_size):
                    trace += self._penalized_object.factorized_solve(
                        factorization, _sparse_col_index(rhs, i)
                    )[i]

            # prevent needing to calculate analytical solution again
            self._trace = trace
        else:
            rng_samples = _get_rng(rng).choice([-1., 1.], size=(self._basis_size, n_samples))
            if self._rhs_extra is None:
                rhs_u = self._weights[:, None] * rng_samples
            else:
                rhs_u = self._rhs.tocsr() @ rng_samples
            # H @ u == (W + P)^-1 @ (W @ u)
            hat_u = self._penalized_object.solve(self._lhs, rhs_u, overwrite_b=True)
            # stochastic trace is the average of the trace of u.T @ H @ u;
            # trace(A.T @ B) == (A * B).sum() (see
            # https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product ),
            # with the latter using less memory and being much faster to compute; for future
            # reference: einsum('ij,ij->', A, B) == (A * B).sum(), but is typically faster
            trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace


class PSplineResult(WhittakerResult):
    """
    Represents the result of penalized spline (P-Spline) smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    P-Spline smoothing. This class should not be initialized by external users.

    """

    def __init__(self, penalized_object, weights=None, rhs_extra=None):
        """
        Initializes the result object.

        In the most basic formulation, the linear equation for P-spline smoothing
        is ``(B.T @ W @ B + P) c = B.T @ W @ y`` and ``v = B @ c``.
        ``(W + P) @ v = W @ y``. Then the hat matrix would be
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W)`` or, equivalently
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality of intermediate calculations.

        For more complex usages, the equation can be expressed as:
        ``(B.T @ W @ B + P) @ c = (B.T @ W + rhs_partial) @ y``, such that the hat is given as:
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial)``, or equivalently
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial) @ B``. Simplifying leads to
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B + rhs_extra)``.

        Parameters
        ----------
        penalized_object : pybaselines._spline_utils.PSpline
            The penalized system object used for solving.
        weights : numpy.ndarray, shape (N,) optional
            The weights used to solve the system. Default is None, which will set
            all weights to 1.
        rhs_extra : numpy.ndarray or scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            Additional terms besides ``B.T @ W @ B`` within the right hand side of the hat
            matrix. Default is None.

        """
        super().__init__(penalized_object, weights=weights, rhs_extra=rhs_extra)
        self._btwb_ = None

    @property
    def _shape(self):
        """The shape of the penalized system.

        Returns
        -------
        tuple[int, int]
            The penalized system's shape.

        """
        return (len(self._penalized_object.basis.x),)

    @property
    def _lhs(self):
        """
        The left hand side of the hat matrix in banded format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        numpy.ndarray
            The array representing the left hand side of the hat matrix.

        """
        if self._hat_lhs is None:
            self._hat_lhs = _add_diagonals(
                self._btwb, self._penalized_object.penalty, self._penalized_object.lower
            )
        return self._hat_lhs

    @property
    def _rhs(self):
        """
        The right hand side of the hat matrix in sparse format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse object representing the right hand side of the hat matrix.

        """
        if self._hat_rhs is None:
            btwb = _banded_to_sparse(self._btwb, lower=self._penalized_object.lower)
            if self._rhs_extra is None:
                self._hat_rhs = btwb
            else:
                if not issparse(self._rhs_extra):
                    self._rhs_extra = _banded_to_sparse(
                        self._rhs_extra, lower=self._penalized_object.lower
                    )
                self._hat_rhs = self.rhs_extra + btwb
        return self._hat_rhs

    @property
    def _btwb(self):
        """
        The matrix multiplication of ``B.T @ W @ B`` in banded format.

        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        numpy.ndarray
            The array representing the matrix multiplication of ``B.T @ W @ B``.

        """
        if self._btwb_ is None:
            self._btwb_ = self._penalized_object._make_btwb(self._weights)
        return self._btwb_

    def effective_dimension(self, n_samples=0, rng=1234):
        """
        Calculates the effective dimension from the trace of the hat matrix.

        For typical P-spline smoothing, the linear equation would be
        ``(B.T @ W @ B + lam * P) c = B.T @ W @ y`` and ``v = B @ c``. Then the hat matrix
        would be ``B @ (B.T @ W @ B + lam * P)^-1 @ (B.T @ W)`` or, equivalently
        ``(B.T @ W @ B + lam * P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality. The effective dimension for the system
        can be estimated as the trace of the hat matrix.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (N, `n_samples`) Rademacher random variables
            (ie. either -1 or 1).

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective dimension for
            the system.

        Raises
        ------
        TypeError
            Raised if `n_samples` is not an integer greater than or equal to 0.

        References
        ----------
        Eilers, P., et al. Flexible Smoothing with B-splines and Penalties. Statistical Science,
        1996, 11(2), 89-121.

        Hutchinson, M. A stochastic estimator of the trace of the influence matrix for laplacian
        smoothing splines. Communications in Statistics - Simulation and Computation, (1990),
        19(2), 433-450.

        Meyer, R., et al. Hutch++: Optimal Stochastic Trace Estimation. 2021 Symposium on
        Simplicity in Algorithms (SOSA), (2021), 142-155.

        """
        # TODO could maybe make default n_samples to None and decide to use analytical or
        # stochastic trace based on data size; data size > 1000 use stochastic with default
        # n_samples = 100?
        if n_samples == 0:
            if self._trace is not None:
                return self._trace
            use_analytic = True
            rhs_format = 'csc'
        else:
            if n_samples < 0 or not isinstance(n_samples, int):
                raise TypeError('n_samples must be a non-negative integer')
            use_analytic = False
            rhs_format = 'csr'

        rhs = self._rhs.asformat(rhs_format)
        if use_analytic:
            # compute each diagonal of the hat matrix separately so that the full
            # hat matrix does not need to be stored in memory
            trace = 0
            factorization = self._penalized_object.factorize(self._lhs)
            for i in range(self._basis_size):
                trace += self._penalized_object.factorized_solve(
                    factorization, _sparse_col_index(rhs, i)
                )[i]
            # prevent needing to calculate analytical solution again
            self._trace = trace
        else:
            rng_samples = _get_rng(rng).choice([-1., 1.], size=(self._basis_size, n_samples))
            # H @ u == (B.T @ W @ B + P)^-1 @ (B.T @ W @ B) @ u
            hat_u = self._penalized_object.solve(self._lhs, rhs @ rng_samples, overwrite_b=True)
            # stochastic trace is the average of the trace of u.T @ H @ u;
            # trace(u.T @ H @ u) == sum(u * (H @ u))
            trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace


class PSplineResult2D(PSplineResult):
    """
    Represents the result of 2D penalized spline (P-Spline) smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    P-Spline smoothing. This class should not be initialized by external users.

    """

    def __init__(self, penalized_object, weights=None, rhs_extra=None):
        """
        Initializes the result object.

        In the most basic formulation, the linear equation for P-spline smoothing
        is ``(B.T @ W @ B + P) c = B.T @ W @ y`` and ``v = B @ c``.
        ``(W + P) @ v = W @ y``. Then the hat matrix would be
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W)`` or, equivalently
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality of intermediate calculations.

        For more complex usages, the equation can be expressed as:
        ``(B.T @ W @ B + P) @ c = (B.T @ W + rhs_partial) @ y``, such that the hat is given as:
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial)``, or equivalently
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial) @ B``. Simplifying leads to
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B + rhs_extra)``.

        Parameters
        ----------
        penalized_object : pybaselines.two_d._spline_utils.PSpline2D
            The penalized system object used for solving.
        weights : numpy.ndarray, shape (M, N) or shape (``M * N``,) optional
            The weights used to solve the system. Default is None, which will set
            all weights to 1.
        rhs_extra : numpy.ndarray or scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            Additional terms besides ``B.T @ W @ B`` within the right hand side of the hat
            matrix. Default is None.

        """
        super().__init__(penalized_object, weights, rhs_extra)
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape(self._shape)

    @property
    def _shape(self):
        """The shape of the penalized system.

        Returns
        -------
        tuple[int, int]
            The penalized system's shape.

        """
        return (len(self._penalized_object.basis.x), len(self._penalized_object.basis.z))

    @property
    def _lhs(self):
        """
        The left hand side of the hat matrix in banded format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.csc_array or scipy.sparse.csc_matrix
            The left hand side of the hat matrix.

        """
        if self._hat_lhs is None:
            self._hat_lhs = (self._btwb + self._penalized_object.penalty).tocsc()
        return self._hat_lhs

    @property
    def _rhs(self):
        """
        The right hand side of the hat matrix in sparse format.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse object representing the right hand side of the hat matrix.

        """
        if self._hat_rhs is None:
            if self._rhs_extra is None:
                self._hat_rhs = self._btwb
            else:
                self._hat_rhs = self._rhs_extra + self._btwb
        return self._hat_rhs

    @property
    def _btwb(self):
        """
        The matrix multiplication of ``B.T @ W @ B`` in full, sparse format.

        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse object representing the matrix multiplication of ``B.T @ W @ B``.

        """
        # TODO can remove once PSpline and PSpline2D unify their btwb method calls; or
        # just keep the docstring since the types are different
        if self._btwb_ is None:
            self._btwb_ = self._penalized_object.basis._make_btwb(self._weights)
        return self._btwb_

    def effective_dimension(self, n_samples=0, rng=1234):
        """
        Calculates the effective dimension from the trace of the hat matrix.

        For typical P-spline smoothing, the linear equation would be
        ``(B.T @ W @ B + lam * P) c = B.T @ W @ y`` and ``v = B @ c``. Then the hat matrix
        would be ``B @ (B.T @ W @ B + lam * P)^-1 @ (B.T @ W)`` or, equivalently
        ``(B.T @ W @ B + lam * P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality. The effective dimension for the system
        can be estimated as the trace of the hat matrix.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (``M * N``, `n_samples`) Rademacher random variables
            (eg. either -1 or 1).

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective dimension for
            the system.

        Raises
        ------
        TypeError
            Raised if `n_samples` is not an integer greater than or equal to 0.

        References
        ----------
        Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
        Statistics and Data Analysis, 2006, 50(1), 61-76.

        Hutchinson, M. A stochastic estimator of the trace of the influence matrix for laplacian
        smoothing splines. Communications in Statistics - Simulation and Computation, (1990),
        19(2), 433-450.

        Meyer, R., et al. Hutch++: Optimal Stochastic Trace Estimation. 2021 Symposium on
        Simplicity in Algorithms (SOSA), (2021), 142-155.

        """
        # TODO unify the PSpline and PSpline2D method namings and availability for factorization
        # and solving so that this can be directly inherited from the PSplineResult object
        if n_samples == 0:
            if self._trace is not None:
                return self._trace
            use_analytic = True
            rhs_format = 'csc'
        else:
            if n_samples < 0 or not isinstance(n_samples, int):
                raise TypeError('n_samples must be a non-negative integer')
            use_analytic = False
            rhs_format = 'csr'

        rhs = self._rhs.asformat(rhs_format)
        if use_analytic:
            # compute each diagonal of the hat matrix separately so that the full
            # hat matrix does not need to be stored in memory
            trace = 0
            factorization = factorized(self._lhs)
            for i in range(self._basis_size):
                trace += factorization(_sparse_col_index(rhs, i))[i]
            # prevent needing to calculate analytical solution again
            self._trace = trace
        else:
            rng_samples = _get_rng(rng).choice([-1., 1.], size=(self._basis_size, n_samples))
            # H @ u == (B.T @ W @ B + P)^-1 @ (B.T @ W @ B) @ u
            hat_u = self._penalized_object.direct_solve(self._lhs, rhs @ rng_samples)
            # stochastic trace is the average of the trace of u.T @ H @ u;
            # trace(u.T @ H @ u) == sum(u * (H @ u))
            trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace


class WhittakerResult2D(WhittakerResult):
    """
    Represents the result of 2D Whittaker smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    Whittaker smoothing. This class should not be initialized by external users.

    """

    def __init__(self, penalized_object, weights=None, lhs=None, rhs_extra=None):
        """
        Initializes the result object.

        In the most basic formulation, Whittaker smoothing solves ``(W + P) @ v = W @ y``.
        Then the hat matrix would be ``(W + P)^-1 @ W``. For more complex usages, the
        equation can be expressed as ``lhs @ v = (W + rhs_extra) @ y`` with a corresponding
        hat matrix of ``lhs^-1 @ (W + rhs_extra)``.

        Parameters
        ----------
        penalized_object : pybaselines.two_d._whittaker_utils.WhittakerSystem2D
            The penalized system object used for solving.
        weights : numpy.ndarray, shape (M, N) or shape (``M * N``,) optional
            The weights used to solve the system. Default is None, which will set
            all weights to 1.
        lhs : scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            The left hand side of the hat matrix. Default is None, which will assume that
            `lhs` is the addition of ``diags(weights)`` and ``pentalized_object.penalty``.
        rhs_extra : scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            Additional terms besides the weights within the right hand side of the hat matrix.
            Default is None.

        """
        super().__init__(penalized_object, weights=weights, lhs=lhs, rhs_extra=rhs_extra)
        self._btwb_ = None
        if self._penalized_object._using_svd and self._weights.ndim == 1:
            self._weights = self._weights.reshape(self._shape)
        elif not self._penalized_object._using_svd and self._weights.ndim == 2:
            self._weights = self._weights.ravel()

    @property
    def _shape(self):
        """The shape of the penalized system.

        Returns
        -------
        tuple[int, int]
            The penalized system's shape.

        """
        # TODO replace/remove once PenalizedSystem2D and WhittakerSystem2D are unified
        if hasattr(self._penalized_object, '_num_points'):
            shape = self._penalized_object._num_points
        else:
            shape = self._penalized_object._num_bases
        return shape

    @property
    def _btwb(self):
        """
        The matrix multiplication of ``B.T @ W @ B`` in full, dense format.

        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        numpy.ndarray
            The array representing the matrix multiplication of ``B.T @ W @ B``.

        """
        if self._btwb_ is None:
            self._btwb_ = self._penalized_object._make_btwb(self._weights)
        return self._btwb_

    @property
    def _lhs(self):
        """
        The left hand side of the hat matrix.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        numpy.ndarray or scipy.sparse.csc_array or scipy.sparse.csc_matrix
            The left hand side of the hat matrix. If using SVD, then the output is a numpy
            array; otherwise, it is a sparse object wit CSC format.

        """
        if self._hat_lhs is None:
            if self._penalized_object._using_svd:
                lhs = self._btwb.copy()
                np.fill_diagonal(lhs, lhs.diagonal() + self._penalized_object.penalty)
                self._hat_lhs = lhs
            else:
                return super()._lhs.tocsc()

        return self._hat_lhs

    @property
    def _rhs(self):
        """
        The right hand side of the hat matrix.

        Given the linear system ``lhs @ v = rhs @ y``, the hat matrix is given as ``lhs^-1 @ rhs.
        Lazy implementation so that the calculation is only performed when needed.

        Returns
        -------
        scipy.sparse.sparray or scipy.sparse.spmatrix
            The sparse object representing the right hand side of the hat matrix.

        """
        if self._hat_rhs is None:
            if self._penalized_object._using_svd:
                self._hat_rhs = self._btwb
            else:
                return super()._rhs

        return self._hat_rhs

    def relative_dof(self):
        """
        Calculates the relative effective degrees of freedom for each eigenvector.

        Returns
        -------
        dof : numpy.ndarray, shape (P, Q)
            The relative effective degrees of freedom associated with each eigenvector
            used for the fit. Each individual effective degree of freedom value is between
            0 and 1, with lower values signifying that the eigenvector was less important
            for the fit.

        Raises
        ------
        ValueError
            Raised if the system was solved analytically rather than using eigendecomposition,
            ie. ``num_eigens`` was set to None.

        """
        if not self._penalized_object._using_svd:
            raise ValueError(
                'Cannot calculate degrees of freedom when not using eigendecomposition'
            )
        dof = solve(self._lhs, self._btwb, check_finite=False, assume_a='pos')
        return dof.diagonal().reshape(self._basis_shape)

    def effective_dimension(self, n_samples=0, rng=1234):
        """
        Calculates the effective dimension from the trace of the hat matrix.

        For typical Whittaker smoothing, the linear equation would be
        ``(W + lam * P) v = W @ y``. Then the hat matrix would be ``(W + lam * P)^-1 @ W``.
        If using SVD, the linear equation is ``(B.T @ W @ B + lam * P) c = B.T @ W @ y``  and
        ``v = B @ c``. Then the hat matrix would be ``B @ (B.T @ W @ B + lam * P)^-1 @ (B.T @ W)``
        or, equivalently ``(B.T @ W @ B + lam * P)^-1 @ (B.T @ W @ B)``. The latter expression
        is preferred since it reduces the dimensionality. The effective dimension for the system
        can be estimated as the trace of the hat matrix.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (``M * N``, `n_samples`) Rademacher random variables
            (eg. either -1 or 1).

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective dimension for
            the system.

        Raises
        ------
        TypeError
            Raised if `n_samples` is not 0 and a non-positive integer.

        Notes
        -----
        If using SVD, the trace will be lower than the actual analytical trace. The relative
        difference is reduced as the number of eigenvalues selected approaches the data
        size.

        References
        ----------
        Biessy, G. Whittaker-Henderson smoothing revisited: A modern statistical framework for
        practical use. ASTIN Bulletin, 2025, 1-31.

        Hutchinson, M. A stochastic estimator of the trace of the influence matrix for laplacian
        smoothing splines. Communications in Statistics - Simulation and Computation, (1990),
        19(2), 433-450.

        Meyer, R., et al. Hutch++: Optimal Stochastic Trace Estimation. 2021 Symposium on
        Simplicity in Algorithms (SOSA), (2021), 142-155.

        """
        if n_samples == 0:
            if self._trace is not None:
                return self._trace
            use_analytic = True
        else:
            if n_samples < 0 or not isinstance(n_samples, int):
                raise TypeError('n_samples must be a non-negative integer')
            use_analytic = False
            rng_samples = _get_rng(rng).choice([-1., 1.], size=(self._basis_size, n_samples))

        if self._penalized_object._using_svd:
            # NOTE the only Whittaker-based algorithms that allow performing SVD for solving
            # all use the simple (W + P) v = w * y formulation, so no need to implement for
            # rhs_extra
            if self._rhs_extra is not None:
                raise NotImplementedError(
                    'rhs_extra is not supported when using eigendecomposition'
                )
            if use_analytic:
                trace = self.relative_dof().sum()
                self._trace = trace
            else:
                # H @ u == (B.T @ W @ B + P)^-1 @ (B.T @ W @ B) @ u
                hat_u = solve(
                    self._lhs, self._rhs @ rng_samples, overwrite_b=True,
                    check_finite=False, assume_a='pos'
                )
                # stochastic trace is the average of the trace of u.T @ H @ u;
                # trace(u.T @ H @ u) == sum(u * (H @ u))
                trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples
        else:
            # TODO unify PenalizedSystem and PenalizedSystem2D methods so that this can be
            # directly inherited from WhittakerResult
            if use_analytic:
                # compute each diagonal of the hat matrix separately so that the full
                # hat matrix does not need to be stored in memory
                trace = 0
                factorization = factorized(self._lhs)
                if self._rhs_extra is None:
                    # note: about an order of magnitude faster to omit the sparse rhs for the simple
                    # case of lhs @ v = w * y
                    eye = np.zeros(self._size)
                    for i in range(self._size):
                        eye[i] = self._weights[i]
                        trace += factorization(eye)[i]
                        eye[i] = 0
                else:
                    rhs = self._rhs.tocsc()
                    for i in range(self._basis_size):
                        trace += factorization(_sparse_col_index(rhs, i))[i]
                self._trace = trace

            else:
                if self._rhs_extra is None:
                    rhs_u = self._weights[:, None] * rng_samples
                else:
                    rhs_u = self._rhs.tocsr() @ rng_samples
                # H @ u == (W + P)^-1 @ (W @ u)
                hat_u = self._penalized_object.direct_solve(self._lhs, rhs_u)
                # stochastic trace is the average of the trace of u.T @ H @ u;
                # trace(u.T @ H @ u) == sum(u * (H @ u))
                trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace
