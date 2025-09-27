# -*- coding: utf-8 -*-
"""Tests for pybaselines._banded_solvers.

The functions `pentapy_ptrans1` and `pentapy_ptrans2` were adapted from `pentapy`
(https://github.com/GeoStat-Framework/pentapy) (last accessed March 25, 2025), which was
licensed under the MIT license below.

The MIT License (MIT)

Copyright (c) 2023 Sebastian Müller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Donald Erb
Created on April 10, 2025

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import solve_banded

from pybaselines import _banded_solvers, _banded_utils
from pybaselines._compat import jit

from .base_tests import get_data


# adapted from pentapy (https://github.com/GeoStat-Framework/pentapy); see license above
@jit(nopython=True, cache=True)
def pentapy_ptrans1(mat_flat, rhs):
    """
    The ptrans1 solver from pentapy, for testing.

    Solves the equation ``A @ x = rhs``, given `A` in row-wise banded format as `mat_flat`.

    Parameters
    ----------
    lhs : numpy.ndarray, shape (5, M)
        The pentadiagonal matrix `A` in row-wise banded format (see :func:`pentapy.solve`).
    rhs : numpy.ndarray, shape (M,) or (M, N)
        The right-hand side of the equation.

    Returns
    -------
    result : numpy.ndarray, shape (M,) or (M, N)
        The solution to the linear equation.
    factorization : numpy.ndarray, shape (5, M)
        The factorization of the input `lhs`.

    Notes
    -----
    All modifications to the function `` from pentapy are noted, except for linting corrections.

    Allows for checking both the calculated result and the factorization of the
    input banded matrix.

    """
    mat_j = mat_flat.shape[1]

    result = np.zeros(rhs.shape)  # modified compared to pentapy to allow multiple rhs
    ze = np.zeros(rhs.shape)  # modified compared to pentapy to allow multiple rhs

    # modified from pentapy to allow for outputting the factorization
    factorization = np.zeros((5, mat_j))
    al = factorization[1]  # np.zeros(mat_j)  alpha variables
    be = factorization[0]  # np.zeros(mat_j)  beta variables
    ga = factorization[3]  # np.zeros(mat_j)  gamma variables
    mu = factorization[2]  # np.zeros(mat_j)  mu variables
    factorization[4] = mat_flat[4]  # e variables, unmodified within the factorization

    mu[0] = mat_flat[2, 0]
    al[0] = mat_flat[1, 0] / mu[0]
    be[0] = mat_flat[0, 0] / mu[0]
    ze[0] = rhs[0] / mu[0]

    ga[1] = mat_flat[3, 1]
    mu[1] = mat_flat[2, 1] - al[0] * ga[1]
    al[1] = (mat_flat[1, 1] - be[0] * ga[1]) / mu[1]
    be[1] = mat_flat[0, 1] / mu[1]
    ze[1] = (rhs[1] - ze[0] * ga[1]) / mu[1]

    for i in range(2, mat_j - 2):
        ga[i] = mat_flat[3, i] - al[i - 2] * mat_flat[4, i]
        mu[i] = mat_flat[2, i] - be[i - 2] * mat_flat[4, i] - al[i - 1] * ga[i]
        al[i] = (mat_flat[1, i] - be[i - 1] * ga[i]) / mu[i]
        be[i] = mat_flat[0, i] / mu[i]
        ze[i] = (rhs[i] - ze[i - 2] * mat_flat[4, i] - ze[i - 1] * ga[i]) / mu[i]

    ga[mat_j - 2] = mat_flat[3, mat_j - 2] - al[mat_j - 4] * mat_flat[4, mat_j - 2]
    mu[mat_j - 2] = (
        mat_flat[2, mat_j - 2]
        - be[mat_j - 4] * mat_flat[4, mat_j - 2]
        - al[mat_j - 3] * ga[mat_j - 2]
    )
    al[mat_j - 2] = (mat_flat[1, mat_j - 2] - be[mat_j - 3] * ga[mat_j - 2]) / mu[mat_j - 2]

    ga[mat_j - 1] = mat_flat[3, mat_j - 1] - al[mat_j - 3] * mat_flat[4, mat_j - 1]
    mu[mat_j - 1] = (
        mat_flat[2, mat_j - 1]
        - be[mat_j - 3] * mat_flat[4, mat_j - 1]
        - al[mat_j - 2] * ga[mat_j - 1]
    )

    ze[mat_j - 2] = (
        rhs[mat_j - 2] - ze[mat_j - 4] * mat_flat[4, mat_j - 2] - ze[mat_j - 3] * ga[mat_j - 2]
    ) / mu[mat_j - 2]
    ze[mat_j - 1] = (
        rhs[mat_j - 1] - ze[mat_j - 3] * mat_flat[4, mat_j - 1] - ze[mat_j - 2] * ga[mat_j - 1]
    ) / mu[mat_j - 1]

    # Backward substitution
    result[mat_j - 1] = ze[mat_j - 1]
    result[mat_j - 2] = ze[mat_j - 2] - al[mat_j - 2] * result[mat_j - 1]

    for i in range(mat_j - 3, -1, -1):
        result[i] = ze[i] - al[i] * result[i + 1] - be[i] * result[i + 2]

    return result, factorization


# adapted from pentapy (https://github.com/GeoStat-Framework/pentapy); see license above
@jit(nopython=True, cache=True)
def pentapy_ptrans2(mat_flat, rhs):
    """
    The ptrans2 solver from pentapy, for testing.

    Solves the equation ``A @ x = rhs``, given `A` in row-wise banded format as `mat_flat`.

    Parameters
    ----------
    lhs : numpy.ndarray, shape (5, M)
        The pentadiagonal matrix `A` in row-wise banded format (see :func:`pentapy.solve`).
    rhs : numpy.ndarray, shape (M,) or (M, N)
        The right-hand side of the equation.

    Returns
    -------
    result : numpy.ndarray, shape (M,) or (M, N)
        The solution to the linear equation.
    factorization : numpy.ndarray, shape (5, M)
        The factorization of the input `lhs`.

    Notes
    -----
    All modifications to the function `` from pentapy are noted, except for linting corrections.

    Allows for checking both the calculated result and the factorization of the
    input banded matrix.

    """
    mat_j = mat_flat.shape[1]

    result = np.zeros(rhs.shape)  # modified compared to pentapy to allow multiple rhs
    we = np.zeros(rhs.shape)  # modified compared to pentapy to allow multiple rhs

    # modified from pentapy to allow for outputting the factorization
    factorization = np.zeros((5, mat_j))
    ro = factorization[1]  # np.zeros(mat_j)  rho variables
    si = factorization[3]  # np.zeros(mat_j)  sigma variables
    ps = factorization[2]  # np.zeros(mat_j)  psi variables
    ph = factorization[4]  # phi variables, unmodified within the factorization
    factorization[0] = mat_flat[0]  # b variables, unmodified within the factorization

    ps[mat_j - 1] = mat_flat[2, mat_j - 1]
    si[mat_j - 1] = mat_flat[3, mat_j - 1] / ps[mat_j - 1]
    ph[mat_j - 1] = mat_flat[4, mat_j - 1] / ps[mat_j - 1]
    we[mat_j - 1] = rhs[mat_j - 1] / ps[mat_j - 1]

    ro[mat_j - 2] = mat_flat[1, mat_j - 2]
    ps[mat_j - 2] = mat_flat[2, mat_j - 2] - si[mat_j - 1] * ro[mat_j - 2]
    si[mat_j - 2] = (mat_flat[3, mat_j - 2] - ph[mat_j - 1] * ro[mat_j - 2]) / ps[mat_j - 2]
    ph[mat_j - 2] = mat_flat[4, mat_j - 2] / ps[mat_j - 2]
    we[mat_j - 2] = (rhs[mat_j - 2] - we[mat_j - 1] * ro[mat_j - 2]) / ps[mat_j - 2]

    for i in range(mat_j - 3, 1, -1):
        ro[i] = mat_flat[1, i] - si[i + 2] * mat_flat[0, i]
        ps[i] = mat_flat[2, i] - ph[i + 2] * mat_flat[0, i] - si[i + 1] * ro[i]
        si[i] = (mat_flat[3, i] - ph[i + 1] * ro[i]) / ps[i]
        ph[i] = mat_flat[4, i] / ps[i]
        we[i] = (rhs[i] - we[i + 2] * mat_flat[0, i] - we[i + 1] * ro[i]) / ps[i]

    ro[1] = mat_flat[1, 1] - si[3] * mat_flat[0, 1]
    ps[1] = mat_flat[2, 1] - ph[3] * mat_flat[0, 1] - si[2] * ro[1]
    si[1] = (mat_flat[3, 1] - ph[2] * ro[1]) / ps[1]

    ro[0] = mat_flat[1, 0] - si[2] * mat_flat[0, 0]
    ps[0] = mat_flat[2, 0] - ph[2] * mat_flat[0, 0] - si[1] * ro[0]

    we[1] = (rhs[1] - we[3] * mat_flat[0, 1] - we[2] * ro[1]) / ps[1]
    we[0] = (rhs[0] - we[2] * mat_flat[0, 0] - we[1] * ro[0]) / ps[0]

    # Foreward substitution
    result[0] = we[0]
    result[1] = we[1] - si[1] * result[0]

    for i in range(2, mat_j):
        result[i] = we[i] - si[i] * result[i - 1] - ph[i] * result[i - 2]

    return result, factorization


@pytest.mark.parametrize('data_size', (100, 1001, 5002))
@pytest.mark.parametrize('solver', (1, 2))
@pytest.mark.parametrize('weights_enum', (0, 1))
@pytest.mark.parametrize('lam', np.logspace(-4, 8, 5, base=10, dtype=float))
@pytest.mark.parametrize('two_d', (False, True))
def test_whittaker_solve_symmetric(data_size, solver, weights_enum, lam, two_d):
    """Ensures banded pentadiagonal solvers work for symmetric Whittaker smoothing systems."""
    x, y = get_data(num_points=data_size)
    if two_d:
        y = y * np.linspace(0.5, 10.5, 10)[:, None]
        assert y.ndim == 2  # sanity check that it's 2d
    if weights_enum == 0:  # equal weights
        weights = np.ones(data_size)
    else:
        weights = np.random.default_rng(123).uniform(1e-6, 1, size=data_size)

    penalized_system = _banded_utils.PenalizedSystem(
        data_size, lam=lam, diff_order=2, allow_lower=False
    )
    lhs = penalized_system.add_diagonal(weights)
    rhs = y * weights
    if two_d:
        rhs = rhs.T

    # treat LU factorization as the "known" solution as a sanity check that the
    # equation was solved correctly
    scipy_solution = solve_banded((2, 2), lhs, rhs)

    # pentapy solver expects lhs to be row-wise banded, which can be done by reversing
    # since lhs is symmetric for this test
    pentapy_solver = {1: pentapy_ptrans1, 2: pentapy_ptrans2}[solver]
    pentapy_solution, pentapy_factorization = pentapy_solver(lhs[::-1], rhs)

    output = _banded_solvers.solve_banded_penta(lhs, rhs, solver=solver)
    # also test the underlying solver
    underlying_solver = {1: _banded_solvers._ptrans_1, 2: _banded_solvers._ptrans_2}[solver]
    output2, info = underlying_solver(lhs, rhs)

    assert_allclose(output, scipy_solution, atol=1e-10, rtol=5e-8)
    assert_allclose(output2, scipy_solution, atol=1e-10, rtol=5e-8)
    # put tighter tolerance on the pentapy comparison since it should be an exact
    # replication of the algorithm
    assert_allclose(output, pentapy_solution, atol=1e-16, rtol=1e-16)
    assert_allclose(output2, pentapy_solution, atol=1e-16, rtol=1e-16)


@pytest.mark.parametrize('data_size', (100, 1001, 5002))
@pytest.mark.parametrize('solver', (1, 2))
@pytest.mark.parametrize('weights_enum', (0, 1))
@pytest.mark.parametrize('lam', np.logspace(-4, 8, 5, base=10, dtype=float))
@pytest.mark.parametrize('two_d', (False, True))
def test_whittaker_solve_nonsymmetric(data_size, solver, weights_enum, lam, two_d):
    """Ensures banded pentadiagonal solvers work for non-symmetric Whittaker smoothing systems."""
    x, y = get_data(num_points=data_size)
    if two_d:
        y = y * np.linspace(0.5, 10.5, 10)[:, None]
        assert y.ndim == 2  # sanity check that it's 2d
    if weights_enum == 0:  # equal weights
        weights = np.ones(data_size)
    else:
        weights = np.random.default_rng(123).uniform(1e-6, 1, size=data_size)

    penalized_system = _banded_utils.PenalizedSystem(
        data_size, lam=lam, diff_order=2, allow_lower=False
    )
    multiplier = 1 - 0.5 * weights  # similar array multiplier as used for drpls method
    lhs = penalized_system.penalty * multiplier
    lhs[2] += weights
    lapack_lhs = _banded_utils._shift_rows(lhs.copy(), 2, 2)
    rhs = y * weights
    if two_d:
        rhs = rhs.T

    # treat LU factorization as the "known" solution as a sanity check that the
    # equation was solved correctly
    scipy_solution = solve_banded((2, 2), lapack_lhs, rhs)

    # pentapy solver expects lhs to be row-wise banded, which should be the result
    # after multiplying by an array
    pentapy_solver = {1: pentapy_ptrans1, 2: pentapy_ptrans2}[solver]
    pentapy_solution, pentapy_factorization = pentapy_solver(lhs, rhs)

    output = _banded_solvers.solve_banded_penta(lapack_lhs, rhs, solver=solver)
    # also test the underlying solver
    underlying_solver = {1: _banded_solvers._ptrans_1, 2: _banded_solvers._ptrans_2}[solver]
    output2, info = underlying_solver(lapack_lhs, rhs)

    assert_allclose(output, scipy_solution, atol=1e-10, rtol=5e-8)
    assert_allclose(output2, scipy_solution, atol=1e-10, rtol=5e-8)
    # put tighter tolerance on the pentapy comparison since it should be an exact
    # replication of the algorithm
    assert_allclose(output, pentapy_solution, atol=1e-16, rtol=1e-16)
    assert_allclose(output2, pentapy_solution, atol=1e-16, rtol=1e-16)


@pytest.mark.parametrize('data_size', (100, 1001, 5002))
@pytest.mark.parametrize('solver', (1, 2))
@pytest.mark.parametrize('overwrite_ab', (False, True))
@pytest.mark.parametrize('overwrite_b', (False, True))
def test_overwrite(data_size, solver, overwrite_ab, overwrite_b):
    """Ensures overwrite_[a][ab] work as intended and do not affect the output solution."""
    x, rhs = get_data(num_points=data_size)

    penalized_system = _banded_utils.PenalizedSystem(
        data_size, lam=5., diff_order=2, allow_lower=False
    )
    lhs = penalized_system.add_diagonal(1.)

    # treat LU factorization as the "known" solution as a sanity check that the
    # equation was solved correctly
    scipy_solution = solve_banded((2, 2), lhs, rhs)
    if overwrite_ab:
        input_lhs = lhs.copy()
    else:
        input_lhs = lhs
    if overwrite_b:
        input_rhs = rhs.copy()
    else:
        input_rhs = rhs

    output = _banded_solvers.solve_banded_penta(
        input_lhs, input_rhs, solver=solver, overwrite_ab=overwrite_ab, overwrite_b=overwrite_b
    )
    # ensure it actually overwrites the input when expected to, and doesn't when not
    if overwrite_ab:
        assert not np.allclose(input_lhs, lhs, rtol=1e-10, atol=1e-10)
    else:
        assert_allclose(input_lhs, lhs, rtol=1e-16, atol=1e-16)
    if overwrite_b:
        assert not np.allclose(input_rhs, rhs, rtol=1e-10, atol=1e-10)
    else:
        assert_allclose(input_rhs, rhs, rtol=1e-16, atol=1e-16)

    # also test the underlying solver
    if overwrite_ab:
        input_lhs = lhs.copy()
    else:
        input_lhs = lhs
    if overwrite_b:
        input_rhs = rhs.copy()
    else:
        input_rhs = rhs
    underlying_solver = {1: _banded_solvers._ptrans_1, 2: _banded_solvers._ptrans_2}[solver]
    output2, info = underlying_solver(
        input_lhs, input_rhs, overwrite_ab=overwrite_ab, overwrite_b=overwrite_b
    )
    assert info == 0
    if overwrite_ab:
        assert not np.allclose(input_lhs, lhs, rtol=1e-10, atol=1e-10)
    else:
        assert_allclose(input_lhs, lhs, rtol=1e-16, atol=1e-16)
    if overwrite_b:
        assert not np.allclose(input_rhs, rhs, rtol=1e-10, atol=1e-10)
    else:
        assert_allclose(input_rhs, rhs, rtol=1e-16, atol=1e-16)

    assert_allclose(output, scipy_solution, atol=1e-10, rtol=5e-8)
    assert_allclose(output2, scipy_solution, atol=1e-10, rtol=5e-8)

    assert_allclose(output, output2, atol=1e-16, rtol=1e-16)


@pytest.mark.parametrize('solver', (1, 2))
def test_paper_solutions(solver):
    """Ensures the test cases given in the reference paper pass.

    Example test cases from Section 4 of [1]_.

    References
    ----------
    .. [1] Askar, S., et al. On Solving Pentadiagonal Linear Systems via Transformations.
           Mathematical Problems in Engineering, 2015, 232456.

    """
    underlying_solver = {1: _banded_solvers._ptrans_1, 2: _banded_solvers._ptrans_2}[solver]

    # case 1
    ab = np.array([
        [0, 0, 1, 5, -2, 1, 5, 2, 4, -3],
        [0, 2, 2, 1, 5, -7, 3, -1, 4, 5],
        [1, 2, 3, -4, 5, 6, 7, -1, 1, 8],
        [3, 2, 1, 2, 1, 2, 1, -2, 4, 0],
        [1, 3, 1, 5, 2, 2, 2, -1, 0, 0]
    ], dtype=float)
    y = np.array([8, 33, 8, 24, 29, 98, 99, 17, 57, 108], dtype=float)
    solution = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    output = _banded_solvers.solve_banded_penta(ab, y, solver=solver)
    output2, info = underlying_solver(ab, y)
    assert info == 0

    # machine precision for float(10) ~ 1.8e-15
    assert_allclose(output, solution, atol=1e-15, rtol=2e-15)
    assert_allclose(output2, solution, atol=1e-15, rtol=2e-15)

    # case 2
    size = 500
    ab = np.zeros((5, 500))
    ab[0, 2:] = 1.
    ab[1, 1:-1] = -4
    ab[1, -1] = -2
    ab[2, 1:-2] = 6
    ab[2, 0] = 9
    ab[2, -2] = 5
    ab[2, -1] = 1
    ab[3, :-2] = -4
    ab[3, -2] = -2
    ab[4, :-1] = 1

    y = np.zeros(size)
    y[0] = 6
    y[1] = -1
    solution = np.ones(size)

    output = _banded_solvers.solve_banded_penta(ab, y, solver=solver)
    output2, info = underlying_solver(ab, y)
    assert info == 0

    # solver 2 should match completely and solver 1 with a relative error of ~1.5856e-7
    rtol = 1e-16 if solver == 2 else 1.59e-7
    assert_allclose(output, solution, atol=1e-15, rtol=rtol)
    assert_allclose(output2, solution, atol=1e-15, rtol=rtol)


@pytest.mark.parametrize('solver', (1, 2))
def test_paper_failures(solver):
    """Ensures the test case given in the reference paper fails when it is supposed to.

    Example test case from Section 4 of [1]_. While the reference states it should fail for
    both algorithms, they present the solution for algorithm 2 (ptrans2) in the following
    paragraph, so just ignore when they say algorithm 2 should fail.

    References
    ----------
    .. [1] Askar, S., et al. On Solving Pentadiagonal Linear Systems via Transformations.
           Mathematical Problems in Engineering, 2015, 232456.

    """
    underlying_solver = {1: _banded_solvers._ptrans_1, 2: _banded_solvers._ptrans_2}[solver]

    ab = np.array([
        [0, 0, 1, 1],
        [0, 2, 7, 5],
        [3, -2, -1, 3],
        [-3, 2, 2, 0],
        [3, 1, 0, 0]
    ], dtype=float)
    y = np.array([6, 3, 9, 6], dtype=float)

    if solver == 1:
        with pytest.raises(np.linalg.LinAlgError):
            _banded_solvers.solve_banded_penta(ab, y, solver=solver)
        output2, info = underlying_solver(ab, y)
        assert info > 0
    else:
        solution = np.array([1, 1, 1, 1], dtype=float)
        output = _banded_solvers.solve_banded_penta(ab, y, solver=solver)
        output2, info = underlying_solver(ab, y)
        assert info == 0
        assert_allclose(output, solution, atol=1e-15, rtol=5e-16)
        assert_allclose(output2, solution, atol=1e-15, rtol=5e-16)


@pytest.mark.parametrize('solver', (1, 2))
def test_low_columns(solver):
    """Ensures solve_banded_penta correctly directs to SciPy when ab is less than 4 columns."""
    penalized_system = _banded_utils.PenalizedSystem(3, allow_lower=False)
    lhs = penalized_system.add_diagonal(1.)
    rhs = np.array([1., 2, 3])

    scipy_solution = solve_banded((2, 2), lhs, rhs)
    output = _banded_solvers.solve_banded_penta(lhs, rhs, solver=solver)

    assert_allclose(output, scipy_solution, atol=1e-10, rtol=1e-15)


@pytest.mark.parametrize('diff_order', (1, 3, 4))
@pytest.mark.parametrize('size', (500, 1001))
@pytest.mark.parametrize('solver', (1, 2))
def test_non_pentadiagonal_fails(diff_order, size, solver):
    """Ensures an error is raised if the input lhs matrix is not pentadiagonal."""
    penalized_system = _banded_utils.PenalizedSystem(
        size, diff_order=diff_order, allow_lower=False
    )
    lhs = penalized_system.add_diagonal(1.)
    rhs = np.random.default_rng(123).normal(0, 0.5, size)
    with pytest.raises(ValueError):
        _banded_solvers.solve_banded_penta(lhs, rhs, solver=solver)


def test_unknown_solver_fails():
    """Ensures only values of 1 or 2 are accepted are valid solver inputs."""
    size = 100
    penalized_system = _banded_utils.PenalizedSystem(size, allow_lower=False)
    lhs = penalized_system.add_diagonal(1.)
    rhs = np.random.default_rng(123).normal(0, 0.5, size)

    solver_inputs = ['1', '2', 0, 3, 4]
    for solver in solver_inputs:
        with pytest.raises(ValueError):
            _banded_solvers.solve_banded_penta(lhs, rhs, solver=solver)
