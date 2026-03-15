# -*- coding: utf-8 -*-
"""Tests for pybaselines.two_d._spline_utils.

@author: Donald Erb
Created on January 8, 2024

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import interpolate
from scipy.sparse import issparse, kron
from scipy.sparse.linalg import spsolve

from pybaselines._compat import identity
from pybaselines._banded_utils import difference_matrix
from pybaselines.two_d import _spline_utils

from ..base_tests import get_2dspline_inputs


@pytest.mark.parametrize('num_knots', (10, (11, 20)))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5, (2, 3)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, (1e1, 1e2)))
def test_solve_psplines(data_fixture2d, num_knots, spline_degree, diff_order, lam):
    """
    Tests the accuracy of the penalized spline solvers.

    Uses the nieve way to solve 2D PSplines from Eilers's paper as the expected result, which
    uses the flattened `y` and weight values, while pybaselines uses the second, more efficient
    method in Eiler's paper which directly uses the 2D `y` and weights.

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    """
    x, z, y = data_fixture2d
    (
        num_knots_r, num_knots_c, spline_degree_x, spline_degree_z,
        lam_x, lam_z, diff_order_x, diff_order_z
    ) = get_2dspline_inputs(num_knots, spline_degree, lam, diff_order)

    knots_r = _spline_utils._spline_knots(x, num_knots_r, spline_degree_x, True)
    basis_r = _spline_utils._spline_basis(x, knots_r, spline_degree_x)

    knots_c = _spline_utils._spline_knots(z, num_knots_c, spline_degree_z, True)
    basis_c = _spline_utils._spline_basis(z, knots_c, spline_degree_z)

    num_bases = (basis_r.shape[1], basis_c.shape[1])
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.size)
    weights = np.clip(weights, 0, 1, dtype=float)

    # note: within Eiler's paper, the basis was defined as kron(basis_z, basis_x),
    # but the rows and columns were switched, ie. it should be kron(basis_rows, basis_columns),
    # so it is just a nomenclature difference
    basis = kron(basis_r, basis_c)
    CWT = basis.multiply(
        np.repeat(weights.flatten(), num_bases[0] * num_bases[1]).reshape(len(x) * len(z), -1)
    ).T
    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    expected_coeffs = spsolve(CWT @ basis + penalty, CWT @ y.flatten())
    expected_result = basis @ expected_coeffs

    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    pspline = _spline_utils.PSpline2D(spline_basis, lam=lam, diff_order=diff_order)

    output = pspline.solve(y, weights=weights.reshape(y.shape))

    assert_allclose(output.flatten(), expected_result, rtol=1e-8, atol=1e-8)
    assert_allclose(pspline.coef, expected_coeffs, rtol=1e-8, atol=1e-8)

    # also ensure that the pspline's basis can use the solved coefficients
    basis_output = spline_basis.basis @ pspline.coef
    assert_allclose(basis_output, expected_result, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('spline_degree', (1, 2, 3, [2, 3]))
@pytest.mark.parametrize('num_knots', (16, [21, 30]))
@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_pspline_setup(data_fixture2d, num_knots, spline_degree, diff_order, lam):
    """Ensure the PSpline2D setup is correct."""
    x, z, y = data_fixture2d
    (
        num_knots_r, num_knots_c, spline_degree_x, spline_degree_z,
        lam_x, lam_z, diff_order_x, diff_order_z
    ) = get_2dspline_inputs(num_knots, spline_degree, lam, diff_order)

    knots_r = _spline_utils._spline_knots(x, num_knots_r, spline_degree_x, True)
    basis_r = _spline_utils._spline_basis(x, knots_r, spline_degree_x)

    knots_c = _spline_utils._spline_knots(z, num_knots_c, spline_degree_z, True)
    basis_c = _spline_utils._spline_basis(z, knots_c, spline_degree_z)

    num_bases = (basis_r.shape[1], basis_c.shape[1])

    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    pspline = _spline_utils.PSpline2D(spline_basis, lam=lam, diff_order=diff_order)

    assert spline_basis.basis_r.shape == (len(x), len(knots_r) - spline_degree_x - 1)
    assert spline_basis.basis_c.shape == (len(z), len(knots_c) - spline_degree_z - 1)
    assert_array_equal(spline_basis._num_bases, num_bases)
    assert_array_equal(pspline._num_bases, num_bases)

    assert issparse(spline_basis.basis_r)
    assert issparse(spline_basis.basis_c)

    assert_allclose(spline_basis.basis_r.toarray(), basis_r.toarray(), rtol=1e-12, atol=1e-12)
    assert_allclose(spline_basis.basis_c.toarray(), basis_c.toarray(), rtol=1e-12, atol=1e-12)
    assert_allclose(pspline.penalty.toarray(), penalty.toarray(), rtol=1e-12, atol=1e-12)

    assert_array_equal(pspline.diff_order, (diff_order_x, diff_order_z))
    assert_array_equal(spline_basis.num_knots, (num_knots_r, num_knots_c))
    assert_array_equal(spline_basis.spline_degree, (spline_degree_x, spline_degree_z))
    assert_array_equal(pspline.lam, (lam_x, lam_z))
    assert pspline.coef is None  # None since the solve method has not been called
    assert spline_basis.basis_r.shape == (len(x), num_knots_r + spline_degree_x - 1)
    assert spline_basis.basis_c.shape == (len(z), num_knots_c + spline_degree_z - 1)
    assert_array_equal(
        spline_basis._num_bases,
        (num_knots_r + spline_degree_x - 1, num_knots_c + spline_degree_z - 1)
    )
    assert spline_basis.knots_r.shape == (num_knots_r + 2 * spline_degree_x,)
    assert spline_basis.knots_c.shape == (num_knots_c + 2 * spline_degree_z,)
    assert isinstance(spline_basis.x, np.ndarray)
    assert isinstance(spline_basis.z, np.ndarray)

    # _basis should be None since the basis attribute has not been accessed yet
    assert spline_basis._basis is None

    expected_basis = kron(basis_r, basis_c).toarray()

    assert_allclose(spline_basis.basis.toarray(), expected_basis, rtol=1e-12, atol=1e-12)
    assert_allclose(spline_basis._basis.toarray(), expected_basis, rtol=1e-12, atol=1e-12)


def test_spline_basis_same_basis(data_fixture2d):
    """Ensures SplineBasis2D.same_basis works correctly."""
    x, z, y = data_fixture2d

    num_knots = (20, 30)
    spline_degree = (2, 3)

    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )

    assert spline_basis.same_basis(num_knots, spline_degree)
    assert not spline_basis.same_basis(num_knots[::-1], spline_degree)
    assert not spline_basis.same_basis(num_knots, spline_degree[::-1])
    assert not spline_basis.same_basis(10, spline_degree)
    assert not spline_basis.same_basis(num_knots, 1)
    assert not spline_basis.same_basis(10, 1)


@pytest.mark.parametrize('diff_order', (0, -1, [0, 0], [1, 0], [0, 1], [-1, 1], [1, -1]))
def test_pspline_diff_order_zero_fails(data_fixture2d, diff_order):
    """Ensures a difference order of 0 fails."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(x, z)
    with pytest.raises(ValueError):
        _spline_utils.PSpline2D(spline_basis, diff_order=diff_order)


@pytest.mark.parametrize('spline_degree', (-2, -1, [-1, 1], [1, -1]))
def test_spline_basis_negative_spline_degree_fails(data_fixture2d, spline_degree):
    """Ensures a spline degree less than 0 fails."""
    x, z, y = data_fixture2d
    with pytest.raises(ValueError):
        _spline_utils.SplineBasis2D(x, z, spline_degree=spline_degree)


@pytest.mark.parametrize('lam', (-2, 0, [-1, 1], [1, -1], [1, 0], [0, 1]))
def test_pspline_negative_lam_fails(data_fixture2d, lam):
    """Ensures a lam value less than or equal to 0 fails."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(x, z)
    with pytest.raises(ValueError):
        _spline_utils.PSpline2D(spline_basis, lam=lam)


def test_spline_basis_non_finite_fails():
    """Ensure non-finite values raise an exception when check_finite is True."""
    x = np.linspace(-1, 1, 100)
    z = np.linspace(-1, 1, 50)
    original_x_value = x[0]
    original_z_value = z[0]
    for value in (np.nan, np.inf, -np.inf):
        x[0] = value
        with pytest.raises(ValueError):
            _spline_utils.SplineBasis2D(x, z, check_finite=True)
        x[0] = original_x_value

    for value in (np.nan, np.inf, -np.inf):
        z[0] = value
        with pytest.raises(ValueError):
            _spline_utils.SplineBasis2D(x, z, check_finite=True)
        z[0] = original_z_value


@pytest.mark.parametrize('spline_degree', (1, 2, 3, (2, 3)))
@pytest.mark.parametrize('num_knots', (10, 40, (20, 30)))
@pytest.mark.parametrize('diff_order', (1, 2, (1, 2)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
def test_pspline_tck(data_fixture2d, num_knots, spline_degree, diff_order, lam):
    """Ensures the tck attribute can correctly recreate the solved spline."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )
    pspline = _spline_utils.PSpline2D(spline_basis, lam=lam, diff_order=diff_order)
    fit_spline = pspline.solve(y, weights=np.ones_like(y))

    # ensure tck is the knots, coefficients, and spline degree
    assert len(pspline.tck) == 3
    (knots_r, knots_c), coeffs, (degree_x, degree_z) = pspline.tck

    assert_allclose(knots_r, spline_basis.knots_r, rtol=1e-12, atol=1e-12)
    assert_allclose(knots_c, spline_basis.knots_c, rtol=1e-12, atol=1e-12)
    assert_allclose(coeffs, pspline.coef.reshape(pspline._num_bases), rtol=1e-12, atol=1e-12)
    if isinstance(spline_degree, int):
        assert degree_x == spline_degree
        assert degree_z == spline_degree
    else:
        assert degree_x == spline_degree[0]
        assert degree_z == spline_degree[1]

    # Now recreate the spline with scipy's NdBSpline and ensure it is the same;
    # NdBSpline was introduced in scipy 1.12.0
    if hasattr(interpolate, 'NdBSpline'):
        # np.array(np.meshgrid(x, z)).T is the same as doing
        # np.array(np.meshgrid(x, z, indexing='ij')).transpose([1, 2, 0]), which
        # is just zipping the meshgrid of each x and z value
        recreated_spline = interpolate.NdBSpline(*pspline.tck)(np.array(np.meshgrid(x, z)).T)

        assert_allclose(recreated_spline, fit_spline, rtol=1e-10, atol=1e-12)


def test_pspline_tck_none(data_fixture2d):
    """Ensures an exception is raised when tck attribute is accessed without first solving once."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(x, z)
    pspline = _spline_utils.PSpline2D(spline_basis)

    assert pspline.coef is None
    with pytest.raises(ValueError):
        tck = pspline.tck


def test_pspline_tck_readonly(data_fixture2d):
    """Ensures the tck attribute is read-only."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(x, z, num_knots=10)
    pspline = _spline_utils.PSpline2D(spline_basis)

    with pytest.raises(AttributeError):
        pspline.tck = (1, 2, 3)

    pspline.solve(y, np.ones_like(y))
    with pytest.raises(AttributeError):
        pspline.tck = (1, 2, 3)


@pytest.mark.parametrize('spline_degree', (1, 2, 3, [2, 3]))
@pytest.mark.parametrize('num_knots', (10, 50, [20, 31]))
def test_spline_basis(data_fixture2d, spline_degree, num_knots):
    """Ensures spline basis setup is correct by comparing to SciPy."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=num_knots, spline_degree=spline_degree, check_finite=False
    )

    # ensure tk is the knots and spline degree
    assert len(spline_basis.tk) == 2
    (knots_r, knots_c), (degree_x, degree_z) = spline_basis.tk

    assert_allclose(knots_r, spline_basis.knots_r, rtol=1e-12, atol=1e-12)
    assert_allclose(knots_c, spline_basis.knots_c, rtol=1e-12, atol=1e-12)
    if isinstance(spline_degree, int):
        assert degree_x == spline_degree
        assert degree_z == spline_degree
    else:
        assert degree_x == spline_degree[0]
        assert degree_z == spline_degree[1]

    # Now compare with scipy's NdBSpline.design_matrix and ensure it is the same;
    # NdBSpline was introduced in scipy 1.12.0 and NdBspline.design_matrix was
    # introduced in scipy 1.13.0
    if hasattr(interpolate, 'NdBSpline') and hasattr(interpolate.NdBSpline, 'design_matrix'):
        # np.array(np.meshgrid(x, z)).T is the same as doing
        # np.array(np.meshgrid(x, z, indexing='ij')).transpose([1, 2, 0]), which
        # is just zipping the meshgrid of each x and z value; then reshape to
        # flatten into 2D for what scipy expects
        xz = np.array(np.meshgrid(x, z)).T.reshape(-1, 2)
        scipy_basis = interpolate.NdBSpline.design_matrix(xz, *spline_basis.tk)

        assert_allclose(spline_basis.basis.toarray(), scipy_basis.toarray(), rtol=1e-12, atol=1e-12)


def test_spline_basis_tk_readonly(data_fixture2d):
    """Ensures the tk attribute is read-only."""
    x, z, y = data_fixture2d
    spline_basis = _spline_utils.SplineBasis2D(x, z)

    with pytest.raises(AttributeError):
        spline_basis.tk = (1, 2)
