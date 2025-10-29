# -*- coding: utf-8 -*-
"""Tests for pybaselines._spline_utils.

@author: Donald Erb
Created on November 8, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.interpolate import BSpline
from scipy.linalg import cholesky_banded
from scipy.sparse import issparse
from scipy.sparse.linalg import factorized, spsolve

from pybaselines import _banded_utils, _spline_utils
from pybaselines._compat import diags, _HAS_NUMBA


def _nieve_basis_matrix(x, knots, spline_degree):
    """Simple function for creating the basis matrix for a spline."""
    num_bases = len(knots) - spline_degree - 1
    basis = np.empty((num_bases, len(x)))
    coeffs = np.zeros(num_bases)
    # evaluate each single basis
    for i in range(num_bases):
        coeffs[i] = 1  # evaluate the i-th basis
        basis[i] = BSpline(knots, coeffs, spline_degree)(x)
        coeffs[i] = 0  # reset back to zero

    return basis.T


def test_find_interval():
    """Ensures the correct knot interval is identified when starting from left or right."""
    spline_degree = 3
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # knots are range(0, 9) and extended on left and right with 3 extra values
    knots = np.arange(-3, 13)
    # indices within knots such that knots[index] <= x_i < knots[index + 1]; last
    # knot is 11 rather than 12 since last index == number of basis functions
    expected_indices = [3, 4, 5, 6, 7, 8, 9, 10, 11, 11]
    num_bases = len(knots) - spline_degree - 1

    left_indices = []
    right_indices = []
    for x_val in x:
        # search starts at the left of the knot array
        left_indices.append(
            _spline_utils._find_interval(knots, spline_degree, x_val, 0, num_bases)
        )
        # search starts at the right of the knot array
        right_indices.append(
            _spline_utils._find_interval(knots, spline_degree, x_val, num_bases - 1, num_bases)
        )

    assert_array_equal(left_indices, expected_indices)
    assert_array_equal(right_indices, expected_indices)


@pytest.mark.parametrize('num_knots', (0, 1))
def test_spline_knots_too_few_knots(num_knots):
    """Ensures an error is raised if the number of knots is less than 2."""
    with pytest.raises(ValueError):
        _spline_utils._spline_knots(np.arange(10), num_knots)


@pytest.mark.parametrize('num_knots', (2, 20, 201))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('penalized', (True, False))
def test_spline_knots(data_fixture, num_knots, spline_degree, penalized):
    """Ensures the spline knot placement is correct."""
    x, y = data_fixture
    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, penalized)
    min_x = x.min()
    max_x = x.max()
    if penalized:
        dx = (max_x - min_x) / (num_knots - 1)
        inner_knots = np.linspace(min_x, max_x, num_knots)
        expected_knots = np.concatenate((
            np.linspace(min_x - dx * spline_degree, min_x - dx, spline_degree),
            inner_knots,
            np.linspace(max_x + dx, max_x + dx * spline_degree, spline_degree)
        ))
    else:
        inner_knots = np.percentile(x, np.linspace(0, 100, num_knots))
        expected_knots = np.concatenate((
            np.repeat(min_x, spline_degree),
            inner_knots,
            np.repeat(max_x, spline_degree)
        ))

    assert_allclose(knots, expected_knots, 1e-10)
    assert np.all(x >= knots[spline_degree])
    assert np.all(x <= knots[len(knots) - 1 - spline_degree])
    assert np.all(np.diff(knots) >= 0)


@pytest.mark.parametrize('num_knots', (2, 20, 201))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('source', ('simple', 'numba', 'scipy'))
def test_spline_basis(data_fixture, num_knots, spline_degree, source):
    """Tests the accuracy of the spline basis matrix."""
    if source == 'scipy' and not hasattr(BSpline, 'design_matrix'):
        # BSpline.design_matrix not available until scipy 1.8.0
        pytest.skip(
            "BSpline.design_matrix is not available in the tested scipy version (must be >= 1.8.0)"
        )

    x, y = data_fixture
    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)

    if source == 'scipy':
        basis_func = BSpline.design_matrix
    elif source == 'numba':
        basis_func = _spline_utils._make_design_matrix
    elif source == 'simple':
        basis_func = _spline_utils._slow_design_matrix

    basis = basis_func(x, knots, spline_degree)
    expected_basis = _nieve_basis_matrix(x, knots, spline_degree)

    assert basis.shape == (len(x), len(knots) - spline_degree - 1)

    assert issparse(basis)
    assert_allclose(basis.toarray(), expected_basis, 1e-14, 1e-14)
    # also test the main interface for the spline basis; only test for one
    # source to avoid unnecessary repetition
    if source == 'simple':
        basis_2 = _spline_utils._spline_basis(x, knots, spline_degree)
        assert issparse(basis_2)

        assert_allclose(basis.toarray(), expected_basis, 1e-14, 1e-14)
        assert_allclose(basis.toarray(), basis_2.toarray(), 1e-14, 1e-14)


def test_spline_basis_x_bounds():
    """
    Ensures an exception is raised when x-values are outside of the knots range.

    Hard to test all of the basis creation pathways using mock, so just rely on
    CI to test all pathways.
    """
    num_knots = 20
    spline_degree = 3
    x = np.linspace(-1, 1, 50)
    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)

    x[0] = knots[spline_degree] - 1  # lower than the knots bounds
    with pytest.raises(ValueError):
        _spline_utils._spline_basis(x, knots, spline_degree)
    x[0] = -1  # reset the value so that the upper bounds can be checked
    x[-1] = knots[-(spline_degree + 1)] + 1
    with pytest.raises(ValueError):
        _spline_utils._spline_basis(x, knots, spline_degree)


def test_bspline_has_extrapolate():
    """Validates the check for ``BSpline.design_matrix`` having an extrapolate keyword."""
    if not hasattr(BSpline, 'design_matrix'):
        # BSpline.design_matrix not available until scipy 1.8.0
        with pytest.raises(AttributeError):
            _spline_utils._bspline_has_extrapolate()
    else:
        spline_degree = 3
        x = np.linspace(-1, 1, 100)
        knots = _spline_utils._spline_knots(
            x, num_knots=50, spline_degree=spline_degree, penalized=True
        )
        # check if extrapolate is actually an allowable keyword argument
        has_extrapolate = True
        try:
            BSpline.design_matrix(x, knots, spline_degree, extrapolate=True)
        except TypeError:
            has_extrapolate = False

        assert _spline_utils._bspline_has_extrapolate() == has_extrapolate

        # Also check that the result is cached so that the actual check is only done once. The
        # cache hits would depend on the test run order, so just check that calling it twice
        # results in a non-zero hits value and that misses is 1 (the first call counts as a miss)
        assert _spline_utils._bspline_has_extrapolate() == has_extrapolate
        assert _spline_utils._bspline_has_extrapolate.cache_info().hits > 0
        assert _spline_utils._bspline_has_extrapolate.cache_info().misses == 1


@pytest.mark.parametrize('num_knots', (2, 20, 201))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
def test_numba_basis_len(data_fixture, num_knots, spline_degree):
    """
    Checks the length of the data attribute of the spline basis csr matrix.

    The `data` attribute of the csr matrix spline basis is used in a
    function to quickly compute ``B.T @ B`` and ``B.T @ y`` without havig to
    recreate the basis each iteration.

    Only the numba spline basis is tested, since if numba is not installed,
    the function is not used since it is much slower than other ways to compute
    ``B.T @ B`` and ``B.T @ y`` which do not require the data length to be a
    specific value.

    """
    x, y = data_fixture
    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._make_design_matrix(x, knots, spline_degree)

    assert len(basis.tocsr().data) == len(x) * (spline_degree + 1)


@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('has_numba', (True, False))
def test_pspline_solve(data_fixture, num_knots, spline_degree, diff_order, lower_only, has_numba):
    """
    Tests the accuracy of the penalized spline solvers.

    The penalized spline solver has two routes:
    1) use the custom numba function (preferred if numba is installed)
    2) compute ``B.T @ W @ B`` and ``B.T @ (w * y)`` using the sparse system (last resort)

    Both are tested here.

    """
    x, y = data_fixture
    # ensure x and y are floats
    x = x.astype(float)
    y = y.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty = _banded_utils.diff_penalty_diagonals(num_bases, diff_order, lower_only)
    penalty_matrix = _banded_utils.diff_penalty_matrix(num_bases, diff_order=diff_order)

    expected_coeffs = spsolve(
        basis.T @ diags(weights, format='csr') @ basis + penalty_matrix,
        basis.T @ (weights * y)
    )
    expected_spline = basis @ expected_coeffs
    with mock.patch.object(_spline_utils, '_HAS_NUMBA', has_numba):
        spline_basis = _spline_utils.SplineBasis(
            x, num_knots=num_knots, spline_degree=spline_degree
        )
        pspline = _spline_utils.PSpline(
            spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
        )
        assert_allclose(
            pspline.solve_pspline(y, weights=weights, penalty=penalty),
            expected_spline, 1e-10, 1e-12
        )
        assert_allclose(
            pspline.coef, expected_coeffs, 1e-10, 1e-12
        )


@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('lower_only', (True, False))
def test_pspline_factorize_solve(data_fixture, num_knots, spline_degree, diff_order, lower_only):
    """Tests the factorize and factorized_solve methods of a PSpline object."""
    x, y = data_fixture
    # ensure x and y are floats
    x = x.astype(float)
    y = y.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty_matrix = _banded_utils.diff_penalty_matrix(num_bases, diff_order=diff_order)

    lhs_sparse = basis.T @ diags(weights, format='csr') @ basis + penalty_matrix
    rhs = basis.T @ (weights * y)
    expected_coeffs = spsolve(lhs_sparse, rhs)

    lhs_banded = _banded_utils._sparse_to_banded(lhs_sparse)[0]
    if lower_only:
        lhs_banded = lhs_banded[len(lhs_banded) // 2:]

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
    )
    output_factorization = pspline.factorize(lhs_banded)
    if lower_only:
        expected_factorization = cholesky_banded(lhs_banded, lower=True)

        assert_allclose(
            output_factorization, expected_factorization, rtol=1e-14, atol=1e-14
        )
    else:
        assert callable(output_factorization)

    output = pspline.factorized_solve(output_factorization, rhs)
    assert_allclose(output, expected_coeffs, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('has_numba', (True, False))
def test_pspline_make_btwb(data_fixture, num_knots, spline_degree, diff_order, lower_only,
                           has_numba):
    """
    Tests the accuracy of the the PSpline ``B.T @ W @ B`` calculation.

    The PSpline has two routes:
    1) use the custom numba function (preferred if numba is installed)
    2) compute ``B.T @ W @ B`` using the sparse system (last resort)

    Both are tested here.

    """
    x, y = data_fixture
    # ensure x and y are floats
    x = x.astype(float)
    y = y.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)

    sparse_calc = basis.T @ diags(weights, format='csr') @ basis
    expected_output = _banded_utils._sparse_to_banded(sparse_calc)[0]
    if has_numba and len(expected_output) != 2 * spline_degree + 1:
        # the sparse calculation can truncate rows of just zeros, so refill them
        zeros = np.zeros(expected_output.shape[1])
        expected_output = np.vstack((zeros, expected_output, zeros))
    if lower_only:
        expected_output = expected_output[len(expected_output) // 2:]

    with mock.patch.object(_spline_utils, '_HAS_NUMBA', has_numba):
        spline_basis = _spline_utils.SplineBasis(
            x, num_knots=num_knots, spline_degree=spline_degree
        )
        pspline = _spline_utils.PSpline(
            spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
        )
        assert_allclose(
            pspline._make_btwb(weights=weights), expected_output, 1e-14, 1e-14
        )


@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('lower_only', (True, False))
def test_pspline_effective_dimension(data_fixture, num_knots, spline_degree, diff_order,
                                     lower_only):
    """
    Tests the effective_dimension method of a PSpline object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty_matrix = _banded_utils.diff_penalty_matrix(num_bases, diff_order=diff_order)

    btwb = basis.T @ diags(weights, format='csr') @ basis
    factorization = factorized(btwb + penalty_matrix)
    expected_ed = 0
    for i in range(num_bases):
        expected_ed += factorization(btwb[:, i].toarray())[i]

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
    )

    output = pspline.effective_dimension(weights, n_samples=0)
    assert_allclose(output, expected_ed, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('n_samples', (100, 201))
def test_pspline_stochastic_effective_dimension(data_fixture, num_knots, spline_degree, diff_order,
                                                lower_only, n_samples):
    """
    Tests the effective_dimension method of a PSpline object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
    )
    # true solution is already verified by other tests, so use that as "known" in
    # this test to only examine the relative difference from using stochastic estimation
    expected_ed = pspline.effective_dimension(weights, n_samples=0)

    output = pspline.effective_dimension(weights, n_samples=n_samples)
    assert_allclose(output, expected_ed, rtol=5e-2, atol=1e-5)


@pytest.mark.parametrize('n_samples', (-1, 50.5))
def test_pspline_stochastic_effective_dimension_invalid_samples(data_fixture, n_samples):
    """Ensures a non-zero, non-positive `n_samples` input raises an exception."""
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    spline_basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(spline_basis)
    with pytest.raises(TypeError):
        pspline.effective_dimension(weights, n_samples=n_samples)


def check_penalized_spline(penalized_system, expected_penalty, lam, diff_order,
                           allow_lower, reverse_diags, spline_degree, num_knots,
                           data_size):
    """
    Tests a PSpline object with the expected values.

    Also tests the `same_basis` method for the PSpline.

    """
    padding = spline_degree - diff_order
    expected_padded_penalty = lam * _banded_utils._pad_diagonals(
        expected_penalty, padding, lower_only=allow_lower
    )

    assert_array_equal(penalized_system.original_diagonals, expected_penalty)
    assert_array_equal(penalized_system.penalty, expected_padded_penalty)
    assert penalized_system.reversed == reverse_diags
    assert penalized_system.lower == allow_lower
    assert penalized_system.diff_order == diff_order
    assert penalized_system.num_bands == diff_order + max(0, padding)
    assert penalized_system.basis.num_knots == num_knots
    assert penalized_system.basis.spline_degree == spline_degree
    assert penalized_system.coef is None  # None since the solve method has not been called
    assert penalized_system.basis.basis.shape == (data_size, num_knots + spline_degree - 1)
    assert penalized_system.basis._num_bases == num_knots + spline_degree - 1
    assert penalized_system.basis.knots.shape == (num_knots + 2 * spline_degree,)
    assert isinstance(penalized_system.basis.x, np.ndarray)
    assert penalized_system.basis._x_len == len(penalized_system.basis.x)
    assert not penalized_system.using_penta
    if allow_lower:
        assert penalized_system.main_diagonal_index == 0
    else:
        assert penalized_system.main_diagonal_index == diff_order + max(0, padding)

    # check that PSpline.same_basis works as expected
    assert penalized_system.basis.same_basis(num_knots=num_knots, spline_degree=spline_degree)
    for new_num_knots in (1, 5, 1000):
        if new_num_knots == num_knots:
            continue
        for new_spline_degree in range(4):
            if new_spline_degree == spline_degree:
                continue
            assert not penalized_system.basis.same_basis(
                num_knots=new_num_knots, spline_degree=new_spline_degree
            )


@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('reverse_diags', (True, False))
def test_pspline_setup(data_fixture, num_knots, spline_degree, diff_order,
                       allow_lower, reverse_diags):
    """
    Ensure the PSpline setup is correct.

    Since `allow_penta` is always False for PSpline, the `lower` attribute of the
    PenalizedSystem will always equal the input `allow_lower` and the `reversed`
    attribute will be equal to the bool of the input `reverse_diags` input (ie. None
    will also be False).

    """
    x, y = data_fixture
    penalty_size = num_knots + spline_degree - 1
    data_size = len(x)
    lam = 5
    expected_penalty = _banded_utils.diff_penalty_diagonals(
        penalty_size, diff_order=diff_order, lower_only=allow_lower, padding=0
    )
    if reverse_diags:
        expected_penalty = expected_penalty[::-1]

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    if reverse_diags and allow_lower:
        # this configuration should never be used
        with pytest.raises(ValueError):
            pspline = _spline_utils.PSpline(
                spline_basis, lam=lam, diff_order=diff_order, allow_lower=allow_lower,
                reverse_diags=reverse_diags
            )
    else:
        pspline = _spline_utils.PSpline(
            spline_basis, lam=lam, diff_order=diff_order, allow_lower=allow_lower,
            reverse_diags=reverse_diags
        )
        check_penalized_spline(
            pspline, expected_penalty, lam, diff_order, allow_lower,
            bool(reverse_diags), spline_degree, num_knots, data_size
        )
        # also check that the reset_diagonal method performs similarly
        pspline.reset_penalty_diagonals(
            lam=lam, diff_order=diff_order, allow_lower=allow_lower, reverse_diags=reverse_diags
        )
        check_penalized_spline(
            pspline, expected_penalty, lam, diff_order, allow_lower,
            bool(reverse_diags), spline_degree, num_knots, data_size
        )


def test_spline_basis_non_finite_fails():
    """Ensure non-finite values raise an exception when check_finite is True."""
    x = np.linspace(-1, 1, 100)
    for value in (np.nan, np.inf, -np.inf):
        x[0] = value
        with pytest.raises(ValueError):
            _spline_utils.SplineBasis(x, check_finite=True)


def test_pspline_diff_order_zero_fails(data_fixture):
    """Ensures a difference order of 0 fails."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x)
    with pytest.raises(ValueError):
        _spline_utils.PSpline(basis, diff_order=0)


@pytest.mark.parametrize('spline_degree', (-2, -1, 0, 1))
def test_spline_basis_negative_spline_degree_fails(data_fixture, spline_degree):
    """Ensures a spline degree less than 0 fails."""
    x, y = data_fixture
    if spline_degree >= 0:
        _spline_utils.SplineBasis(x, spline_degree=spline_degree)
    else:
        with pytest.raises(ValueError):
            _spline_utils.SplineBasis(x, spline_degree=spline_degree)


@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('lam', (1e-2, 1e2))
def test_pspline_tck(data_fixture, num_knots, spline_degree, diff_order, lam):
    """Ensures the tck attribute can correctly recreate the solved spline."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x, num_knots=num_knots, spline_degree=spline_degree)
    pspline = _spline_utils.PSpline(basis, diff_order=diff_order, lam=lam)
    fit_spline = pspline.solve_pspline(y, weights=np.ones_like(y))

    # ensure tck is the knots, coefficients, and spline degree
    assert len(pspline.tck) == 3
    knots, coeffs, degree = pspline.tck

    assert_allclose(knots, pspline.basis.knots, rtol=1e-12)
    assert_allclose(coeffs, pspline.coef, rtol=1e-12)
    assert degree == spline_degree

    # now recreate the spline with scipy's BSpline and ensure it is the same
    recreated_spline = BSpline(*pspline.tck)(x)

    assert_allclose(recreated_spline, fit_spline, rtol=1e-10)


def test_pspline_tck_none(data_fixture):
    """Ensures an exception is raised when tck attribute is accessed without first solving once."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(basis)

    assert pspline.coef is None
    with pytest.raises(ValueError):
        tck = pspline.tck


def test_pspline_tck_readonly(data_fixture):
    """Ensures the tck attribute is read-only."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(basis)
    pspline.solve_pspline(y, np.ones_like(y))
    with pytest.raises(AttributeError):
        pspline.tck = (1, 2, 3)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('spline_degree', (1, 2, 3))
def test_pspline_update_lam(data_fixture, diff_order, allow_lower, num_knots, spline_degree):
    """Tests updating the lam value for PSpline."""
    x, y = data_fixture
    lam_init = 5
    basis = _spline_utils.SplineBasis(x, num_knots=num_knots, spline_degree=spline_degree)
    pspline = _spline_utils.PSpline(
        basis, diff_order=diff_order, lam=lam_init, allow_lower=allow_lower
    )
    data_size = pspline._num_bases

    expected_penalty = lam_init * _banded_utils.diff_penalty_diagonals(
        data_size, diff_order=diff_order, lower_only=pspline.lower,
        padding=spline_degree - diff_order
    )
    diag_index = pspline.main_diagonal_index

    assert_allclose(pspline.penalty, expected_penalty, rtol=1e-14, atol=1e-14)
    assert_allclose(
        pspline.main_diagonal, expected_penalty[diag_index], rtol=1e-14, atol=1e-14
    )
    assert_allclose(pspline.lam, lam_init, rtol=1e-15, atol=1e-15)
    for lam in (1e3, 5.2e1):
        expected_penalty = lam * _banded_utils.diff_penalty_diagonals(
            data_size, diff_order=diff_order, lower_only=pspline.lower,
            padding=spline_degree - diff_order
        )
        pspline.update_lam(lam)

        assert_allclose(pspline.penalty, expected_penalty, rtol=1e-14, atol=1e-14)
        assert_allclose(
            pspline.main_diagonal, expected_penalty[diag_index], rtol=1e-14, atol=1e-14
        )
        assert_allclose(pspline.lam, lam, rtol=1e-15, atol=1e-15)


def test_pspline_update_lam_invalid_lam(data_fixture):
    """Ensures PSpline.update_lam throws an exception when given a non-positive lam."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(basis)
    with pytest.raises(ValueError):
        pspline.update_lam(-1.)
    with pytest.raises(ValueError):
        pspline.update_lam(0)


def test_spline_basis_tk_readonly(data_fixture):
    """Ensures the tk attribute is read-only."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x)
    with pytest.raises(AttributeError):
        basis.tk = (1, 2)


@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (20, 101))
def test_spline_basis_tk(data_fixture, num_knots, spline_degree):
    """Ensures the tk attribute can correctly recreate the solved spline."""
    x, y = data_fixture
    basis = _spline_utils.SplineBasis(x, num_knots=num_knots, spline_degree=spline_degree)

    # ensure tk is the knots and spline degree
    assert len(basis.tk) == 2
    knots, degree = basis.tk

    assert_allclose(knots, basis.knots, rtol=1e-12)
    assert degree == spline_degree

    if hasattr(BSpline, 'design_matrix'):
        scipy_basis = BSpline.design_matrix(x, *basis.tk)
        assert_allclose(basis.basis.toarray(), scipy_basis.toarray(), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4))
def test_basis_midpoints(spline_degree):
    """Tests the _basis_midpoints function."""
    knots = np.arange(20)
    if spline_degree % 2:
        expected_points = knots[
            1 + spline_degree // 2:len(knots) - (spline_degree - spline_degree // 2)
        ]
    else:
        midpoints = 0.5 * (knots[1:] + knots[:-1])
        expected_points = midpoints[spline_degree // 2: len(midpoints) - spline_degree // 2]

    output_midpoints = _spline_utils._basis_midpoints(knots, spline_degree)

    assert_allclose(expected_points, output_midpoints, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('lam', (5, 1e2))
def test_compare_to_whittaker(data_fixture, lam, diff_order):
    """
    Ensures Whittaker and PSpline outputs are the same for specific condition.

    If the number of basis functions for splines is equal to the number of data points, and
    the spline degree is set to 0, then the spline basis becomes the identity function
    and should produce the same analytical equation as Whittaker smoothing.

    Since PSplines are more complicated for setting up in 1D than Whittaker smoothing, need to
    verify the PSPline implementation.

    """
    x, y = data_fixture

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=len(x) + 1, spline_degree=0, check_finite=False
    )

    pspline = _spline_utils.PSpline(spline_basis, lam=lam, diff_order=diff_order)

    # sanity check to ensure it was set up correctly
    assert_array_equal(pspline.basis.basis.shape, (len(x), len(x)))

    whittaker_system = _banded_utils.PenalizedSystem(len(y), lam=lam, diff_order=diff_order)

    weights = np.random.default_rng(0).normal(0.8, 0.05, len(y))
    weights = np.clip(weights, 0, 1).astype(float, copy=False)

    main_diag_idx = whittaker_system.main_diagonal_index
    main_diagonal = whittaker_system.penalty[main_diag_idx]
    whittaker_system.penalty[main_diag_idx] = main_diagonal + weights
    whittaker_output = whittaker_system.solve(
        whittaker_system.penalty, weights * y, overwrite_b=True
    )

    spline_output = pspline.solve_pspline(y, weights=weights)
    whittaker_output = whittaker_system.solve(whittaker_system.penalty, weights.ravel() * y.ravel())

    assert_allclose(spline_output, whittaker_output, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (10, 33, 100))
def test_numba_btb_bty(data_fixture, spline_degree, num_knots):
    """Ensures the B.T @ W @ B and B.T @ W @ y Numba implementations are correct."""
    # if not using the BSpline or Numba spline basis functions and instead creating the design
    # matrix row-by-row using BSpline.__call__, some basis elements get truncated to 0 and the
    # resulting csr matrix/array's data attribute will not have the correct number of elements;
    # this is fine since numba_bty_bty would not be called in that situation anyway, so can
    # safely skip this test
    if not (hasattr(BSpline, 'design_matrix') or _HAS_NUMBA):
        pytest.skip(reason='Code path is unused for this combination of dependencies')

    x, y = data_fixture
    x = x.astype(float)
    y = y.astype(float)
    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )

    weights = np.random.default_rng(1234).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float)

    expected_lhs = spline_basis.basis.T @ diags(weights, format='csr') @ spline_basis.basis
    expected_rhs = spline_basis.basis.T @ (weights * y)

    basis_data = spline_basis.basis.tocsr().data
    # sanity check that the data attribute contains the correct amount of points
    assert len(basis_data) == len(x) * (spline_basis.spline_degree + 1)

    ab_lower = np.zeros((spline_basis.spline_degree + 1, spline_basis._num_bases), order='F')
    rhs = np.zeros(spline_basis._num_bases)
    _spline_utils._numba_btb_bty(
        x, spline_basis.knots, spline_basis.spline_degree, y, weights, ab_lower, rhs,
        basis_data
    )

    ab_full = _banded_utils._lower_to_full(ab_lower)

    expected_ab, _ = _banded_utils._sparse_to_banded(expected_lhs)
    if expected_ab.shape[0] != ab_full.shape[0]:
        # diagonals became 0 and were truncated from the sparse object's data attritube
        expected_ab = _banded_utils._pad_diagonals(
            expected_ab, ab_full.shape[0] - expected_ab.shape[0], lower_only=False
        )
    expected_ab_lower = expected_ab[len(expected_ab) // 2:]

    assert_allclose(rhs, expected_rhs, rtol=1e-15, atol=1e-15)
    assert_allclose(ab_full, expected_ab, rtol=1e-15, atol=1e-15)
    assert_allclose(ab_lower, expected_ab_lower, rtol=1e-15, atol=1e-15)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (50, 501))
def test_pspline_lam_extremes(data_fixture, diff_order, allow_lower, spline_degree, num_knots):
    """
    Tests the result of P-spline smoothing for high and low limits of ``lam``.

    When ``lam`` is ~infinite, the solution to ``(B.T @ B + lam * D.T @ D) x = B.T @ y`` should
    approximate a polynomial of degree ``diff_order - 1`` as long as the spline degree is greater
    than or equal to ``diff_order`` according to [1]_. Likewise, as ``lam`` approaches 0, the
    solution should be the same as an interpolating spline of the same spline degree.

    References
    ----------
    .. [1] Eilers, P., et al. Flexible Smoothing with B-splines and Penalties. Statistical
           Science, 1996, 11(2), 89-121.

    """
    x, y = data_fixture
    weights = np.ones_like(y)

    spline_basis = _spline_utils.SplineBasis(x, num_knots=num_knots, spline_degree=spline_degree)
    if spline_degree >= diff_order:  # can only approximate a polynomial if the spline allows
        pspline = _spline_utils.PSpline(
            spline_basis, lam=1e13, diff_order=diff_order, allow_lower=allow_lower
        )
        output = pspline.solve_pspline(y, weights)

        polynomial_fit = np.polynomial.Polynomial.fit(x, y, deg=diff_order - 1)(x)
        # limited by how close to infinity lam can get before it causes numerical instability,
        # and both larger num_knots and larger diff_orders need larger lam for it to be a
        # polynomial, so have to reduce the relative tolerance; num_knots has a larger effect
        # than diff_order, so base the rtol on it
        rtol = {50: 5e-4, 501: 5e-3}[num_knots]
        assert_allclose(output, polynomial_fit, rtol=rtol, atol=1e-10)

    # for lam ~ 0, should just approximate the an interpolating spline
    pspline2 = _spline_utils.PSpline(
        spline_basis, lam=1e-10, diff_order=diff_order, allow_lower=allow_lower
    )
    output2 = pspline2.solve_pspline(y, weights)
    # cannot use interpolation from SciPy since the knot arrangement is going to be different
    expected_coeffs = spsolve(spline_basis.basis.T @ spline_basis.basis, spline_basis.basis.T @ y)
    expected = spline_basis.basis @ expected_coeffs
    assert_allclose(output2, expected, rtol=1e-9, atol=1e-10)
