# -*- coding: utf-8 -*-
"""Tests for pybaselines.results.

@author: Donald Erb
Created on January 8, 2026

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import kron
from scipy.sparse.linalg import factorized

from pybaselines import _banded_utils, _spline_utils, results
from pybaselines.two_d._spline_utils import PSpline2D, SplineBasis2D
from pybaselines.two_d._whittaker_utils import WhittakerSystem2D
from pybaselines._compat import _sparse_col_index, dia_object, diags, identity

from .base_tests import get_2dspline_inputs


@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('allow_penta', (True, False))
@pytest.mark.parametrize('size', (100, 401))
def test_whittaker_effective_dimension(diff_order, allow_lower, allow_penta, size):
    """
    Tests the effective_dimension method of a WhittakerResult object.

    The effective dimension for Whittaker smoothing should be
    ``trace((W + lam * D.T @ D)^-1 @ W)``, where `W` is the weight matrix, and
    ``D.T @ D`` is the penalty.

    """
    weights = np.random.default_rng(0).normal(0.8, 0.05, size)
    weights = np.clip(weights, 0, 1).astype(float)

    lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]
    expected_penalty = _banded_utils.diff_penalty_diagonals(
        size, diff_order=diff_order, lower_only=False
    )
    sparse_penalty = dia_object(
        (lam * expected_penalty, np.arange(diff_order, -(diff_order + 1), -1)),
        shape=(size, size)
    ).tocsr()
    weights_matrix = diags(weights, format='csc')
    factorization = factorized(weights_matrix + sparse_penalty)
    expected_ed = 0
    for i in range(size):
        expected_ed += factorization(_sparse_col_index(weights_matrix, i))[i]

    penalized_system = _banded_utils.PenalizedSystem(
        size, lam=lam, diff_order=diff_order, allow_lower=allow_lower,
        reverse_diags=False, allow_penta=allow_penta
    )
    result_obj = results.WhittakerResult(penalized_system, weights=weights)
    output = result_obj.effective_dimension(n_samples=0)

    assert_allclose(output, expected_ed, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('allow_penta', (True, False))
@pytest.mark.parametrize('size', (100, 401))
@pytest.mark.parametrize('n_samples', (100, 201))
def test_whittaker_effective_dimension_stochastic(diff_order, allow_lower, allow_penta, size,
                                                  n_samples):
    """
    Tests the stochastic effective_dimension calculation of a WhittakerResult object.

    The effective dimension for Whittaker smoothing should be
    ``trace((W + lam * D.T @ D)^-1 @ W)``, where `W` is the weight matrix, and
    ``D.T @ D`` is the penalty.

    """
    weights = np.random.default_rng(0).normal(0.8, 0.05, size)
    weights = np.clip(weights, 0, 1).astype(float)

    lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]

    penalized_system = _banded_utils.PenalizedSystem(
        size, lam=lam, diff_order=diff_order, allow_lower=allow_lower,
        reverse_diags=False, allow_penta=allow_penta
    )
    # true solution is already verified by other tests, so use that as "known" in
    # this test to only examine the relative difference from using stochastic estimation
    result_obj = results.WhittakerResult(penalized_system, weights=weights)
    expected_ed = result_obj.effective_dimension(n_samples=0)

    output = result_obj.effective_dimension(n_samples=n_samples)

    assert_allclose(output, expected_ed, rtol=5e-1, atol=1e-5)


@pytest.mark.parametrize('size', (100, 401))
@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('large_lam', (True, False))
@pytest.mark.parametrize('solver', (0, 1, 2))
def test_whittaker_effective_dimension_lam_extremes(size, diff_order, large_lam, solver):
    """
    Tests the effective dimension of Whittaker smoothing for high and low limits of ``lam``.

    When `lam` is ~infinite, the fit should approximate a polynomial of degree
    ``diff_order - 1``, so the effective dimension should be `diff_order` sinc ED for a
    polynomial is ``degree + 1``. Likewise, as `lam` approaches 0, the solution should
    be the same as an interpolating spline of the same spline degree, so its effective
    dimension should be the number of basis functions, i.e. the number of data points
    when using Whittaker smoothing.

    References
    ----------
    Eilers, P., et al. Flexible Smoothing with B-splines and Penalties. Statistical
    Science, 1996, 11(2), 89-121.

    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    if solver == 0:  # use full banded with solve_banded
        allow_lower = False
        allow_penta = False
    elif solver == 1:  # use lower banded with solveh_banded
        allow_lower = True
        allow_penta = False
    else:  # use full banded with pentadiagonal solver if diff_order=2
        allow_lower = False
        allow_penta = True

    if large_lam:
        lam = 1e14
        expected_ed = diff_order
        # limited by how close to infinity lam can get before it causes numerical instability,
        # and larger diff_orders need larger lam for it to be a polynomial, so have to reduce the
        # relative tolerance as diff_order increases
        rtol = {1: 8e-3, 2: 5e-3, 3: 3e-2}[diff_order]
    else:
        lam = 1e-16
        expected_ed = size
        rtol = 1e-15

    penalized_system = _banded_utils.PenalizedSystem(
        size, lam=lam, diff_order=diff_order, allow_lower=allow_lower,
        reverse_diags=False, allow_penta=allow_penta
    )
    result_obj = results.WhittakerResult(penalized_system)
    output = result_obj.effective_dimension(n_samples=0)
    assert_allclose(output, expected_ed, rtol=rtol, atol=1e-11)


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (20, 101))
@pytest.mark.parametrize('large_lam', (True, False))
@pytest.mark.parametrize('allow_lower', (True, False))
def test_pspline_effective_dimension_lam_extremes(data_fixture, diff_order, spline_degree,
                                                  num_knots, large_lam, allow_lower):
    """
    Tests the effective dimension of P-spline smoothing for high and low limits of ``lam``.

    When `lam` is ~infinite, the spline fit should approximate a polynomial of degree
    ``diff_order - 1``, so the effective dimension should be `diff_order` sinc ED for a
    polynomial is ``degree + 1``. Likewise, as ``lam`` approaches 0, the solution should
    be the same as an interpolating spline of the same spline degree, so its effective
    dimension should be the number of basis functions.

    References
    ----------
    Eilers, P., et al. Flexible Smoothing with B-splines and Penalties. Statistical
    Science, 1996, 11(2), 89-121.

    """
    x, y = data_fixture
    if large_lam:
        lam = 1e14
        expected_ed = diff_order
        # limited by how close to infinity lam can get before it causes numerical instability,
        # and both larger num_knots and larger diff_orders need larger lam for it to be a
        # polynomial, so have to reduce the relative tolerance; num_knots has a larger effect
        # than diff_order, so base the rtol on it
        if spline_degree >= diff_order:  # can approximate a polynomial
            rtol = {20: 6e-3, 101: 5e-2}[num_knots]
        else:
            # technically should skip since the spline cannot approximate the polynomial,
            # but just increase rtol instead
            rtol = {20: 1e-2, 101: 1e-1}[num_knots]
    else:
        lam = 1e-16
        expected_ed = num_knots + spline_degree - 1
        rtol = 1e-15

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=lam, diff_order=diff_order, allow_lower=allow_lower
    )
    result_obj = results.PSplineResult(pspline)
    output = result_obj.effective_dimension(n_samples=0)
    assert_allclose(output, expected_ed, rtol=rtol, atol=1e-11)


@pytest.mark.parametrize('n_samples', (-1, 50.5))
def test_whittaker_effective_dimension_stochastic_invalid_samples(data_fixture, n_samples):
    """Ensures a non-zero, non-positive `n_samples` input raises an exception."""
    x, y = data_fixture
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1).astype(float)

    penalized_system = _banded_utils.PenalizedSystem(x.size)
    result_obj = results.WhittakerResult(penalized_system, weights=weights)
    with pytest.raises(TypeError):
        result_obj.effective_dimension(n_samples=n_samples)


def test_whittaker_result_no_weights(data_fixture):
    """Ensures weights are initialized as ones if not given to WhittakerResult."""
    x, y = data_fixture

    penalized_system = _banded_utils.PenalizedSystem(x.size)
    result_obj = results.WhittakerResult(penalized_system)

    assert_allclose(result_obj._weights, np.ones(y.shape), rtol=1e-16, atol=0)


@pytest.mark.parametrize('num_knots', (20, 51))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3))
@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('lower_only', (True, False))
def test_pspline_effective_dimension(data_fixture, num_knots, spline_degree, diff_order,
                                     lower_only):
    """
    Tests the effective_dimension method of a PSplineResult object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1).astype(float)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty_matrix = _banded_utils.diff_penalty_matrix(num_bases, diff_order=diff_order)

    btwb = basis.T @ diags(weights, format='csr') @ basis
    factorization = factorized(btwb + penalty_matrix)
    expected_ed = 0
    for i in range(num_bases):
        expected_ed += factorization(_sparse_col_index(btwb, i))[i]

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
    )
    result_obj = results.PSplineResult(pspline, weights)
    output = result_obj.effective_dimension(n_samples=0)
    assert_allclose(output, expected_ed, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('num_knots', (20, 51))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3))
@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('lower_only', (True, False))
@pytest.mark.parametrize('n_samples', (100, 201))
def test_pspline_stochastic_effective_dimension(data_fixture, num_knots, spline_degree, diff_order,
                                                lower_only, n_samples):
    """
    Tests the effective_dimension method of a PSplineResult object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1).astype(float)

    spline_basis = _spline_utils.SplineBasis(
        x, num_knots=num_knots, spline_degree=spline_degree
    )
    pspline = _spline_utils.PSpline(
        spline_basis, lam=1, diff_order=diff_order, allow_lower=lower_only
    )
    # true solution is already verified by other tests, so use that as "known" in
    # this test to only examine the relative difference from using stochastic estimation
    result_obj = results.PSplineResult(pspline, weights)
    expected_ed = result_obj.effective_dimension(n_samples=0)

    output = result_obj.effective_dimension(n_samples=n_samples)
    assert_allclose(output, expected_ed, rtol=5e-2, atol=1e-5)


@pytest.mark.parametrize('n_samples', (-1, 50.5))
def test_pspline_stochastic_effective_dimension_invalid_samples(data_fixture, n_samples):
    """Ensures a non-zero, non-positive `n_samples` input raises an exception."""
    x, y = data_fixture
    x = x.astype(float)
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1).astype(float)

    spline_basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(spline_basis)
    result_obj = results.PSplineResult(pspline, weights)
    with pytest.raises(TypeError):
        result_obj.effective_dimension(n_samples=n_samples)


def test_pspline_result_no_weights(data_fixture):
    """Ensures weights are initialized as ones if not given to PSplineResult."""
    x, y = data_fixture

    spline_basis = _spline_utils.SplineBasis(x)
    pspline = _spline_utils.PSpline(spline_basis)
    result_obj = results.PSplineResult(pspline)

    assert_allclose(result_obj._weights, np.ones(y.shape), rtol=1e-16, atol=0)


@pytest.mark.parametrize('num_knots', (10, (11, 20)))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, (2, 3)))
@pytest.mark.parametrize('diff_order', (1, 2, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, (1e1, 1e2)))
def test_pspline_two_d_effective_dimension(data_fixture2d, num_knots, spline_degree, diff_order,
                                           lam):
    """
    Tests the effective_dimension method of a PSpline object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, z, y = data_fixture2d
    (
        num_knots_r, num_knots_c, spline_degree_r, spline_degree_c,
        lam_r, lam_c, diff_order_r, diff_order_c
    ) = get_2dspline_inputs(num_knots, spline_degree, lam, diff_order)

    knots_r = _spline_utils._spline_knots(x, num_knots_r, spline_degree_r, True)
    basis_r = _spline_utils._spline_basis(x, knots_r, spline_degree_r)

    knots_c = _spline_utils._spline_knots(z, num_knots_c, spline_degree_c, True)
    basis_c = _spline_utils._spline_basis(z, knots_c, spline_degree_c)

    num_bases = (basis_r.shape[1], basis_c.shape[1])
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    spline_basis = SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    # make B.T @ W @ B using generalized linear array model since it's much faster and
    # already verified in other tests
    btwb = spline_basis._make_btwb(weights).tocsc()
    P_r = kron(
        _banded_utils.diff_penalty_matrix(num_bases[0], diff_order=diff_order_r),
        identity(num_bases[1])
    )
    P_c = kron(
        identity(num_bases[0]),
        _banded_utils.diff_penalty_matrix(num_bases[1], diff_order=diff_order_c)
    )
    penalty = lam_r * P_r + lam_c * P_c

    lhs = btwb + penalty
    factorization = factorized(lhs.tocsc())
    expected_ed = 0
    for i in range(np.prod(num_bases)):
        expected_ed += factorization(_sparse_col_index(btwb, i))[i]

    pspline = PSpline2D(spline_basis, lam=lam, diff_order=diff_order)

    result_obj = results.PSplineResult2D(pspline, weights)
    output = result_obj.effective_dimension(n_samples=0)

    assert_allclose(output, expected_ed, rtol=1e-14, atol=1e-10)


@pytest.mark.parametrize('num_knots', (10, (11, 20)))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, (2, 3)))
@pytest.mark.parametrize('diff_order', (1, 2, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, (1e1, 1e2)))
@pytest.mark.parametrize('n_samples', (100, 201))
def test_pspline_two_d_effective_dimension_stochastic(data_fixture2d, num_knots, spline_degree,
                                                      diff_order, lam, n_samples):
    """
    Tests the effective_dimension method of a PSpline object.

    The effective dimension for penalized spline smoothing should be
    ``trace((B.T @ W @ B + lam * D.T @ D)^-1 @ B.T @ W @ B)``, where `W` is the weight matrix,
    ``D.T @ D`` is the penalty, and `B` is the spline basis.

    """
    x, z, y = data_fixture2d
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    spline_basis = SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    pspline = PSpline2D(spline_basis, lam=lam, diff_order=diff_order)
    # true solution is already verified by other tests, so use that as "known" in
    # this test to only examine the relative difference from using stochastic estimation
    result_obj = results.PSplineResult2D(pspline, weights)
    expected_ed = result_obj.effective_dimension(n_samples=0)

    output = result_obj.effective_dimension(n_samples=n_samples)
    assert_allclose(output, expected_ed, rtol=1e-1, atol=1e-5)


@pytest.mark.parametrize('n_samples', (-1, 50.5))
def test_pspline_two_d_stochastic_effective_dimension_invalid_samples(data_fixture2d, n_samples):
    """Ensures a non-zero, non-positive `n_samples` input raises an exception."""
    x, z, y = data_fixture2d
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    spline_basis = SplineBasis2D(x, z, num_knots=10)
    pspline = PSpline2D(spline_basis)
    result_obj = results.PSplineResult2D(pspline, weights)
    with pytest.raises(TypeError):
        result_obj.effective_dimension(n_samples=n_samples)


def test_pspline_result_two_d_weights(data_fixture2d):
    """Ensures both 1D and 2D weights are handled correctly by PSplineResult2D."""
    x, z, y = data_fixture2d
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    spline_basis = SplineBasis2D(x, z, num_knots=10)
    pspline = PSpline2D(spline_basis)

    result_obj = results.PSplineResult2D(pspline, weights)
    result_obj_1d = results.PSplineResult2D(pspline, weights.ravel())

    assert_allclose(result_obj._weights, result_obj_1d._weights, rtol=1e-16, atol=0)


def test_pspline_result_two_d_no_weights(data_fixture2d):
    """Ensures weights are initialized as ones if not given to PSplineResult2D."""
    x, z, y = data_fixture2d

    spline_basis = SplineBasis2D(x, z, num_knots=10)
    pspline = PSpline2D(spline_basis)
    result_obj = results.PSplineResult2D(pspline)

    assert_allclose(result_obj._weights, np.ones(y.shape), rtol=1e-16, atol=0)


@pytest.mark.parametrize('num_knots', (10, (11, 20)))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, (2, 3)))
@pytest.mark.parametrize('diff_order', (1, 2, (2, 3)))
@pytest.mark.parametrize('large_lam', (True, False))
def test_pspline_two_d_effective_dimension_lam_extremes(data_fixture2d, diff_order, spline_degree,
                                                        num_knots, large_lam):
    """
    Tests the effective dimension of 2D P-spline smoothing for high and low limits of ``lam``.

    Same reasoning as for `test_pspline_effective_dimension_lam_extremes`, except the total
    effective dimension is just the product of the row and column effective dimensions.

    """
    x, z, y = data_fixture2d
    (
        num_knots_r, num_knots_c, spline_degree_r, spline_degree_c,
        lam_r, lam_c, diff_order_r, diff_order_c
    ) = get_2dspline_inputs(num_knots, spline_degree, lam=1, diff_order=diff_order)

    if large_lam:
        lam = 1e14
        expected_ed = diff_order_r * diff_order_c
        # limited by how close to infinity lam can get before it causes numerical instability;
        # just picked rtol that passes all conditions
        rtol = 4e-2
    else:
        lam = 1e-16
        expected_ed = (num_knots_r + spline_degree_r - 1) * (num_knots_c + spline_degree_c - 1)
        rtol = 1e-13

    spline_basis = SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    pspline = PSpline2D(spline_basis, lam=lam, diff_order=diff_order)
    result_obj = results.PSplineResult2D(pspline)

    output = result_obj.effective_dimension(n_samples=0)

    assert_allclose(output, expected_ed, rtol=rtol, atol=1e-11)


@pytest.mark.parametrize('shape', ((20, 23), (51, 6)))
@pytest.mark.parametrize('diff_order', (1, 2, (2, 3)))
@pytest.mark.parametrize('lam', (1e2, (1e1, 1e2)))
@pytest.mark.parametrize('use_svd', (True, False))
def test_whittaker_two_d_effective_dimension(shape, diff_order, lam, use_svd):
    """
    Tests the effective_dimension method of WhittakerResult2D objects.

    The effective dimension for Whittaker smoothing should be
    ``trace((W + lam * D.T @ D)^-1 @ W)``, where `W` is the weight matrix, and
    ``D.T @ D`` is the penalty.

    Tests both the analytic and SVD-based solutions.

    """
    *_, lam_r, lam_c, diff_order_r, diff_order_c = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    weights = np.random.default_rng(0).normal(0.8, 0.05, shape)
    weights = np.clip(weights, 0, 1).astype(float)

    P_r = kron(
        _banded_utils.diff_penalty_matrix(shape[0], diff_order=diff_order_r),
        identity(shape[1])
    )
    P_c = kron(
        identity(shape[0]),
        _banded_utils.diff_penalty_matrix(shape[1], diff_order=diff_order_c)
    )
    penalty = lam_r * P_r + lam_c * P_c

    weights_matrix = diags(weights.ravel(), format='csc')
    factorization = factorized(weights_matrix + penalty)
    expected_ed = 0
    for i in range(np.prod(shape)):
        expected_ed += factorization(_sparse_col_index(weights_matrix, i))[i]

    # the relative error on the trace when using SVD decreases as the number of
    # eigenvalues approaches the data size, so just test with a value very close
    if use_svd:
        num_eigens = (shape[0] - 1, shape[1] - 1)
        atol = 1e-1
        rtol = 5e-2
    else:
        num_eigens = None
        atol = 1e-10
        rtol = 1e-14
    whittaker_system = WhittakerSystem2D(
        shape, lam=lam, diff_order=diff_order, num_eigens=num_eigens
    )
    result_obj = results.WhittakerResult2D(whittaker_system, weights=weights)
    output = result_obj.effective_dimension(n_samples=0)

    assert_allclose(output, expected_ed, rtol=rtol, atol=atol)


@pytest.mark.parametrize('shape', ((20, 23), (51, 6)))
@pytest.mark.parametrize('diff_order', (1, 2, (2, 3)))
@pytest.mark.parametrize('lam', (1e2, (1e1, 1e2)))
@pytest.mark.parametrize('use_svd', (True, False))
@pytest.mark.parametrize('n_samples', (100, 201))
def test_whittaker_two_d_effective_dimension_stochastic(shape, diff_order, lam, use_svd,
                                                        n_samples):
    """
    Tests the stochastic effective_dimension method of WhittakerResult2D objects.

    The effective dimension for Whittaker smoothing should be
    ``trace((W + lam * D.T @ D)^-1 @ W)``, where `W` is the weight matrix, and
    ``D.T @ D`` is the penalty.

    Tests both the analytic and SVD-based solutions.

    """
    weights = np.random.default_rng(0).normal(0.8, 0.05, shape)
    weights = np.clip(weights, 0, 1).astype(float)

    if use_svd:
        num_eigens = (shape[0] - 1, shape[1] - 1)
        rtol = 1e-2
    else:
        num_eigens = None
        rtol = 1e-1

    whittaker_system = WhittakerSystem2D(
        shape, lam=lam, diff_order=diff_order, num_eigens=num_eigens
    )

    # true solution is already verified by other tests, so use that as "known" in
    # this test to only examine the relative difference from using stochastic estimation
    result_obj = results.WhittakerResult2D(whittaker_system, weights=weights)
    expected_ed = result_obj.effective_dimension(n_samples=0)

    output = result_obj.effective_dimension(n_samples=n_samples)
    assert_allclose(output, expected_ed, rtol=rtol, atol=1e-4)


@pytest.mark.parametrize('n_samples', (-1, 50.5))
@pytest.mark.parametrize('num_eigens', (None, 5))
def test_whittaker_two_d_effective_dimension_stochastic_invalid_samples(small_data2d, n_samples,
                                                                        num_eigens):
    """Ensures a non-zero, non-positive `n_samples` input raises an exception."""
    weights = np.random.default_rng(0).normal(0.8, 0.05, small_data2d.shape)
    weights = np.clip(weights, 0, 1).astype(float)

    penalized_system = WhittakerSystem2D(small_data2d.shape, num_eigens=num_eigens)
    result_obj = results.WhittakerResult2D(penalized_system, weights)
    with pytest.raises(TypeError):
        result_obj.effective_dimension(n_samples=n_samples)


@pytest.mark.parametrize('num_eigens', (None, 5))
def test_whittaker_result_two_d_weights(data_fixture2d, num_eigens):
    """Ensures both 1D and 2D weights are handled correctly by WhittakerResult2D."""
    x, z, y = data_fixture2d
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    penalized_system = WhittakerSystem2D(y.shape, num_eigens=num_eigens)

    result_obj = results.WhittakerResult2D(penalized_system, weights)
    result_obj_1d = results.WhittakerResult2D(penalized_system, weights.ravel())

    if num_eigens is None:
        expected_shape = (y.size,)
    else:
        expected_shape = y.shape
    assert result_obj._weights.shape == expected_shape
    assert result_obj_1d._weights.shape == expected_shape
    assert_allclose(result_obj._weights, result_obj_1d._weights, rtol=1e-16, atol=0)


@pytest.mark.parametrize('num_eigens', (None, 5))
def test_whittaker_result_two_d_no_weights(data_fixture2d, num_eigens):
    """Ensures weights are initialized as ones if not given to WhittakerResult2D."""
    x, z, y = data_fixture2d
    if num_eigens is None:
        expected_shape = (y.size,)
    else:
        expected_shape = y.shape

    penalized_system = WhittakerSystem2D(y.shape, num_eigens=num_eigens)
    result_obj = results.WhittakerResult2D(penalized_system)

    assert_allclose(result_obj._weights, np.ones(expected_shape), rtol=1e-16, atol=0)


def test_whittaker_result_two_d_lhs_penalty_raises(data_fixture2d):
    """Ensures an exception is raised if both `lhs` and `penalty` are supplied."""
    x, z, y = data_fixture2d
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1, dtype=float)

    penalized_system = WhittakerSystem2D(y.shape)

    with pytest.raises(ValueError, match='both `lhs` and `penalty` cannot'):
        results.WhittakerResult2D(
            penalized_system, weights, lhs=penalized_system.penalty,
            penalty=penalized_system.penalty
        )


@pytest.mark.parametrize('shape', ((30, 21), (15, 40)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, (1, 2)))
@pytest.mark.parametrize('large_lam', (True, False))
def test_whittaker_two_d_effective_dimension_lam_extremes(shape, diff_order, large_lam):
    """
    Tests the effective dimension of 2D Whittaker smoothing for high and low limits of ``lam``.

    Same reasoning as for `test_whittaker_effective_dimension_lam_extremes`, except the total
    effective dimension is just the product of the row and column effective dimensions.

    """
    if large_lam:
        lam = 1e14
        if isinstance(diff_order, int):
            expected_ed = diff_order**2
            max_diff_order = diff_order
        else:
            expected_ed = np.prod(diff_order)
            max_diff_order = max(diff_order)
        # limited by how close to infinity lam can get before it causes numerical instability,
        # and larger diff_orders need larger lam for it to be a polynomial, so have to reduce the
        # relative tolerance as diff_order increases
        rtol = {1: 8e-3, 2: 2e-2, 3: 7e-2}[max_diff_order]
    else:
        lam = 1e-16
        expected_ed = np.prod(shape)
        rtol = 1e-15

    whittaker_system = WhittakerSystem2D(
        shape, lam=lam, diff_order=diff_order, num_eigens=None
    )
    result_obj = results.WhittakerResult2D(whittaker_system)

    output = result_obj.effective_dimension(n_samples=0)

    assert_allclose(output, expected_ed, rtol=rtol, atol=1e-11)
