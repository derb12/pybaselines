# -*- coding: utf-8 -*-
"""Objects for calculating additional terms from results of analytical baseline correction methods.

Created on November 15, 2025
@author: Donald Erb

"""

import numpy as np
from scipy.sparse import issparse

from ._banded_linalg import _cholesky_inv_bands
from ._banded_utils import _banded_to_sparse, _add_diagonals
from ._compat import diags, _sparse_col_index
from .utils import _get_rng


def _rademacher(shape, rng):
    """
    Generates random samples from a Rademacher distribution, ie. equal chances of -1 or 1.

    Parameters
    ----------
    shape : int or tuple[int, ...]
        The shape of the random samples to create.
    rng : int or numpy.random.Generator or numpy.random.RandomState
        The integer for the seed of the random number generator or an existing generating
        object to use for drawing samples.

    Returns
    -------
    numpy.ndarray, shape `shape`
        The generated random samples.

    References
    ----------
    https://en.wikipedia.org/wiki/Rademacher_distribution

    Hutchinson, M. A stochastic estimator of the trace of the influence matrix for laplacian
    smoothing splines. Communications in Statistics - Simulation and Computation, (1990),
    19(2), 433-450.

    """
    return _get_rng(rng).choice([-1., 1.], size=shape)


class WhittakerResult:
    """
    Represents the result of Whittaker smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    Whittaker smoothing.

    This class should **not** be initialized by external users since its
    initialization signature may change without notice as internally required.

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
            weights = np.ones(self._penalized_object.shape)
        self._weights = weights

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

    def edf(self, n_samples=0, rng=1234):
        """
        Calculates the effective degrees of freedom for the linear system.

        For typical Whittaker smoothing, the linear equation is ``(W + P) v = W @ y``, where P
        represents the total penalty. The corresponding hat matrix, H, defined as
        ``v = H @ y`` is ``(W + P)^-1 @ W``. The effective degrees of freedom
        for the system is estimated as the trace of the hat matrix.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (N, `n_samples`) Rademacher random variables
            (ie. either -1 or 1).
        rng : int or numpy.random.Generator or numpy.random.RandomState, optional
            The integer for the seed of the random number generator or an existing generating
            object to use for the stochastic trace estimation. Default is 1234.

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective degrees of freedom for
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

        Hutchinson, M., et al. Smoothing noisy data with spline functions. Numerische Mathematik,
        1985, 47(1), 99-106.

        Meyer, R., et al. Hutch++: Optimal Stochastic Trace Estimation. 2021 Symposium on
        Simplicity in Algorithms (SOSA), (2021), 142-155.

        """
        # TODO For non-symmetric matrices, can use the slightly more involved algorithm from:
        # Erisman, A., et al. On Computing Certain Elements of the Inverse of a Sparse Matrix.
        # Communication of the ACM. (1975) 18(3), 177-179. https://doi.org/10.1145/360680.360704

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
                if len(self._penalized_object.shape) == 1 and self._penalized_object.lower:
                    trace = (
                        _cholesky_inv_bands(factorization, overwrite_f=True)[0] @ self._weights
                    )
                else:
                    # note: about an order of magnitude faster to omit the sparse rhs for the simple
                    # case of lhs @ v = w * y
                    eye = np.zeros(self._penalized_object.tot_bases)
                    for i in range(self._penalized_object.tot_bases):
                        eye[i] = self._weights[i]
                        trace += self._penalized_object.factorized_solve(factorization, eye)[i]
                        eye[i] = 0
            else:
                rhs = self._rhs.tocsc()
                for i in range(self._penalized_object.tot_bases):
                    trace += self._penalized_object.factorized_solve(
                        factorization, _sparse_col_index(rhs, i)
                    )[i]

            # prevent needing to calculate analytical solution again
            self._trace = trace
        else:
            rng_samples = _rademacher((self._penalized_object.tot_bases, n_samples), rng)
            if self._rhs_extra is None:
                rhs_u = self._weights[:, None] * rng_samples
            else:
                rhs_u = self._rhs.tocsr() @ rng_samples
            # H @ u == (W + P)^-1 @ (W @ u)
            hat_u = self._penalized_object.direct_solve(self._lhs, rhs_u, overwrite_b=True)
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
    P-Spline smoothing.

    This class should **not** be initialized by external users since its
    initialization signature may change without notice as internally required.

    """

    def __init__(self, penalized_object, weights=None, rhs_extra=None, penalty=None):
        """
        Initializes the result object.

        In the most basic formulation, the linear equation for P-spline smoothing
        is ``(B.T @ W @ B + P) c = B.T @ W @ y`` and ``v = B @ c``.
        ``(W + P) @ v = W @ y``. Then the hat matrix would be
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W)``. The trace of the hat matrix is
        equivalent to the trace of its rearrangement:
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality of intermediate calculations.

        For more complex usages, the equation can be expressed as:
        ``(B.T @ W @ B + P) @ c = (B.T @ W + rhs_partial) @ y``, such that the hat is given as:
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial)``. The trace of the hat matrix is
        equivalent to the trace of its rearrangement:
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
        penalty : numpy.ndarray, optional
            The penalty `P` for the system, in the same banded format as used by
            `penalized_object`. If None (default), will use ``penalized_object.penalty``.
            If given, will overwrite ``penalized_object.penalty`` with the given penalty.

        """
        super().__init__(penalized_object, weights=weights, rhs_extra=rhs_extra)
        self._btwb_ = None
        if penalty is not None:
            self._penalized_object.penalty = penalty

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
                self._hat_rhs = self._rhs_extra + btwb
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

    @property
    def tck(self):
        """
        The knots, spline coefficients, and spline degree to reconstruct the fit baseline.

        Can be used with SciPy's :class:`scipy.interpolate.BSpline`, to allow for reconstructing
        the fit baseline to allow for other usages such as evaluating with different x-values.

        Returns
        -------
        numpy.ndarray, shape (K,)
            The knots for the spline. Has a shape of `K`, which is equal to
            ``num_knots + 2 * spline_degree``.
        numpy.ndarray, shape (M,)
            The spline coeffieicnts. Has a shape of `M`, which is the number of basis functions
            (equal to ``K - spline_degree - 1`` or equivalently ``num_knots + spline_degree - 1``).
        int
            The degree of the spline.

        """
        return self._penalized_object.tck

    def edf(self, n_samples=0, rng=1234):
        """
        Calculates the effective degrees of freedom for the linear system.

        For typical P-spline smoothing, the linear equation is
        ``(B.T @ W @ B + P) c = B.T @ W @ y`` and ``v = B @ c``, where P represents the total
        penalty. The corresponding hat matrix, H, defined as ``v = H @ y`` is
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W)``. The effective degrees of freedom is
        estimated as the trace of the hat matrix, and is equivalent to the trace of the
        rearrangement ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B)``. The latter expression is
        preferred since it reduces the dimensionality of intermediate calculations.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (N, `n_samples`) Rademacher random variables
            (ie. either -1 or 1).
        rng : int or numpy.random.Generator or numpy.random.RandomState, optional
            The integer for the seed of the random number generator or an existing generating
            object to use for the stochastic trace estimation. Default is 1234.

        Returns
        -------
        trace : float
            The trace of the hat matrix, denoting the effective degrees of freedom for
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
            factorization = self._penalized_object.factorize(self._lhs)
            if (
                len(self._penalized_object.shape) == 1
                and self._penalized_object.lower
                and self._rhs_extra is None
            ):
                lhs_inv_bands = _cholesky_inv_bands(factorization, overwrite_f=True)
                # lhs_inv_bands @ rhs represents all relevant non-zeros since bands of lhs,
                # and thus lhs_inv, is guaranteed to be >= bands of B.T @ W @ B
                trace = (_banded_to_sparse(lhs_inv_bands, lower=True) @ self._rhs).trace()
            else:
                # compute each diagonal of the hat matrix separately so that the full
                # hat matrix does not need to be stored in memory
                trace = 0
                for i in range(self._penalized_object.tot_bases):
                    trace += self._penalized_object.factorized_solve(
                        factorization, _sparse_col_index(rhs, i)
                    )[i]
            # prevent needing to calculate analytical solution again
            self._trace = trace
        else:
            rng_samples = _rademacher((self._penalized_object.tot_bases, n_samples), rng)
            # H @ u == (B.T @ W @ B + P)^-1 @ (B.T @ W @ B) @ u
            hat_u = self._penalized_object.direct_solve(
                self._lhs, rhs @ rng_samples, overwrite_b=True
            )
            # stochastic trace is the average of the trace of u.T @ H @ u;
            # trace(u.T @ H @ u) == sum(u * (H @ u))
            trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace


class PSplineResult2D(PSplineResult):
    """
    Represents the result of 2D penalized spline (P-Spline) smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    P-Spline smoothing.

    This class should **not** be initialized by external users since its
    initialization signature may change without notice as internally required.

    """

    def __init__(self, penalized_object, weights=None, rhs_extra=None, penalty=None):
        """
        Initializes the result object.

        In the most basic formulation, the linear equation for P-spline smoothing
        is ``(B.T @ W @ B + P) c = B.T @ W @ y`` and ``v = B @ c``.
        ``(W + P) @ v = W @ y``. Then the hat matrix would be
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W)``. The trace of the hat matrix is
        equivalent to the trace of its rearrangement:
        ``(B.T @ W @ B + P)^-1 @ (B.T @ W @ B)``. The latter expression is preferred
        since it reduces the dimensionality of intermediate calculations.

        For more complex usages, the equation can be expressed as:
        ``(B.T @ W @ B + P) @ c = (B.T @ W + rhs_partial) @ y``, such that the hat is given as:
        ``B @ (B.T @ W @ B + P)^-1 @ (B.T @ W + rhs_partial)``. The trace of the hat matrix is
        equivalent to the trace of its rearrangement:
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
        penalty : scipy.sparse.sparray or scipy.sparse.spmatrix, optional
            The penalty `P` for the system in full, sparse format. If None (default), will use
            ``penalized_object.penalty``. If given, will overwrite ``penalized_object.penalty``
            with the given penalty.

        """
        super().__init__(penalized_object, weights=weights, rhs_extra=rhs_extra, penalty=penalty)
        if self._weights.ndim == 1:
            self._weights = self._weights.reshape(self._penalized_object.shape)

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
        return super()._btwb  # only overridden to note the return type difference

    @property
    def tck(self):
        """
        The knots, spline coefficients, and spline degree to reconstruct the fit baseline.

        Can be used with SciPy's :class:`scipy.interpolate.NdBSpline`, to allow for reconstructing
        the fit baseline to allow for other usages such as evaluating with different x- and
        z-values.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The knots for the spline along the rows and columns.
        numpy.ndarray, shape (M, N)
            The spline coeffieicnts. Has a shape of (`M`, `N`), corresponding to the number
            of basis functions along the rows and columns.
        numpy.ndarray([int, int])
            The degree of the spline for the rows and columns.

        """
        return super().tck  # only overridden to note the return type difference


class WhittakerResult2D(WhittakerResult):
    """
    Represents the result of 2D Whittaker smoothing.

    Provides methods for extending the solution obtained from baseline algorithms that use
    Whittaker smoothing.

    This class should **not** be initialized by external users since its
    initialization signature may change without notice as internally required.

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

        Raises
        ------
        ValueError
            Raised if both `penalty` and `lhs` are not None.

        """
        super().__init__(penalized_object, weights=weights, lhs=lhs, rhs_extra=rhs_extra)
        self._btwb_ = None

        if self._penalized_object._using_svd and self._weights.ndim == 1:
            self._weights = self._weights.reshape(self._penalized_object.shape)
        elif not self._penalized_object._using_svd and self._weights.ndim == 2:
            self._weights = self._weights.ravel()

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
            array; otherwise, it is a sparse object with CSC format.

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

    def individual_edf(self):
        """
        Calculates the individual effective degrees of freedom for each eigenvector.

        Returns
        -------
        dof : numpy.ndarray, shape (P, Q)
            The effective degrees of freedom associated with each eigenvector
            used for the fit. Each individual effective degree of freedom value is between
            0 and 1, with lower values signifying that the eigenvector contributed less
            to the fit.

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
        dof = self._penalized_object.direct_solve(
            self._lhs, self._btwb, check_finite=False, assume_a='pos'
        )
        return dof.diagonal().reshape(self._penalized_object._num_bases)

    def edf(self, n_samples=0, rng=1234):
        """
        Calculates the effective degrees of freedom for the linear system.

        For typical Whittaker smoothing, the linear equation is ``(W + P) v = W @ y`` where
        P represents the total penalty.
        The corresponding hat matrix, H, defined as ``v = H @ y`` is ``(W + P)^-1 @ W``.
        The effective degrees of freedom for the system is estimated as the trace
        of the hat matrix.

        If using eigendecomposition, the linear equation is ``(B.T @ W @ B + P2) c = B.T @ W @ y``
        and ``v = B @ c``, where P2 represents the total reduced rank penalty. Then the hat matrix
        is ``B @ (B.T @ W @ B + P2)^-1 @ (B.T @ W)``, and its trace is equivalent to the trace
        of its rearrangement ``(B.T @ W @ B + P2)^-1 @ (B.T @ W @ B)``. The latter expression
        is preferred since it reduces the dimensionality.

        Parameters
        ----------
        n_samples : int, optional
            If 0 (default), will calculate the analytical trace. Otherwise, will use stochastic
            trace estimation with a matrix of (``M * N``, `n_samples`) Rademacher random variables
            (eg. either -1 or 1).
        rng : int or numpy.random.Generator or numpy.random.RandomState, optional
            The integer for the seed of the random number generator or an existing generating
            object to use for the stochastic trace estimation. Default is 1234.

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
        If using eigendecomposition, the trace will be lower than the actual analytical trace.
        The relative difference is reduced as the number of eigenvalues selected approaches
        the data size.

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

        if not self._penalized_object._using_svd:
            trace = super().edf(n_samples=n_samples, rng=rng)
        else:
            # NOTE the only Whittaker-based algorithms that allow performing SVD for solving
            # all use the simple (W + P) v = w * y formulation, so no need to implement for
            # rhs_extra
            if self._rhs_extra is not None:
                raise NotImplementedError(
                    'rhs_extra is not supported when using eigendecomposition'
                )
            if use_analytic:
                trace = self.individual_edf().sum()
                self._trace = trace
            else:
                rng_samples = _rademacher((self._penalized_object.tot_bases, n_samples), rng)
                # H @ u == (B.T @ W @ B + P)^-1 @ (B.T @ W @ B) @ u
                hat_u = self._penalized_object.direct_solve(
                    self._lhs, self._rhs @ rng_samples, overwrite_b=True,
                    check_finite=False, assume_a='pos'
                )
                # stochastic trace is the average of the trace of u.T @ H @ u;
                # trace(u.T @ H @ u) == sum(u * (H @ u))
                trace = np.einsum('ij,ij->', rng_samples, hat_u) / n_samples

        return trace
