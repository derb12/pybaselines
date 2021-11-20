# -*- coding: utf-8 -*-
"""Tests for pybaselines._spline_utils.

@author: Donald Erb
Created on November 8, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.interpolate import BSpline, splev
from scipy.sparse import diags, issparse, spdiags
from scipy.sparse.linalg import spsolve

from pybaselines import _algorithm_setup, _spline_utils


def _nieve_basis_matrix(x, knots, spline_degree):
    """Simple function for creating the basis matrix for a spline."""
    num_bases = len(knots) - spline_degree - 1
    basis = np.empty((num_bases, len(x)))
    coeffs = np.zeros(num_bases)
    # evaluate each single basis
    for i in range(num_bases):
        coeffs[i] = 1  # evaluate the i-th basis within splev
        basis[i] = splev(x, (knots, coeffs, spline_degree))
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
    len_knots = len(knots)
    num_bases = len_knots - spline_degree - 1

    left_indices = []
    right_indices = []
    for x_val in x:
        # search starts at the left of the knot array
        left_indices.append(
            _spline_utils._find_interval(knots, spline_degree, x_val, 0, num_bases)
        )
        # search starts at the right of the knot array
        right_indices.append(
            _spline_utils._find_interval(knots, spline_degree, x_val, len_knots, num_bases)
        )

    assert_array_equal(left_indices, expected_indices)
    assert_array_equal(right_indices, expected_indices)


@pytest.mark.parametrize('num_knots', (2, 20, 1001))
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


@pytest.mark.parametrize('num_knots', (2, 20, 1001))
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
    assert_allclose(basis.toarray(), expected_basis, 1e-10, 1e-12)
    # also test the main interface for the spline basis; only test for one
    # source to avoid unnecessary repitition
    if source == 'simple':
        basis_2 = _spline_utils._spline_basis(x, knots, spline_degree)
        assert issparse(basis_2)

        assert_allclose(basis.toarray(), expected_basis, 1e-10, 1e-12)
        assert_allclose(basis.toarray(), basis_2.toarray(), 1e-10, 1e-12)


@pytest.mark.parametrize('num_knots', (2, 20, 1001))
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


def test_scipy_btb_bty(data_fixture):
    """
    Ensures the private function from Scipy works as intended.

    If numba is not installed, the private function scipy.interpolate._bspl._norm_eq_lsq
    is used to calculate ``B.T @ W @ B`` and ``B.T @ W @ y``.

    This test is a "canary in the coal mine", and if it ever fails, the scipy
    support will need to be looked at.

    """
    # import within this function in case this private file is ever renamed
    from scipy.interpolate import _bspl
    _scipy_btb_bty = _bspl._norm_eq_lsq

    x, y = data_fixture
    # ensure x and y are floats
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    weights = np.random.RandomState(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float, copy=False)

    spline_degree = 3
    num_knots = 100

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]

    ab = np.zeros((spline_degree + 1, num_bases), order='F')
    rhs = np.zeros((num_bases, 1), order='F')
    _scipy_btb_bty(x, knots, spline_degree, y.reshape(-1, 1), np.sqrt(weights), ab, rhs)
    rhs = rhs.reshape(-1)

    expected_rhs = basis.T @ (weights * y)
    expected_ab_full = (basis.T @ diags(weights, format='csr') @ basis).todia().data[::-1]
    expected_ab_lower = expected_ab_full[len(expected_ab_full) // 2:]

    assert_allclose(rhs, expected_rhs, 1e-10, 1e-12)
    assert_allclose(ab, expected_ab_lower, 1e-10, 1e-12)


@pytest.mark.parametrize('num_knots', (100, 1000))
@pytest.mark.parametrize('spline_degree', (0, 1, 2, 3, 4, 5))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
def test_solve_psplines(data_fixture, num_knots, spline_degree, diff_order):
    """
    Tests the accuracy of the penalized spline solvers.

    The penalized spline solver has three routes:
    1) use the custom numba function (preferred if numba is installed)
    2) use the scipy function scipy.interpolate._bspl._norm_eq_lsq (used if numba is
       not installed and the scipy import works correctly)
    3) compute ``B.T @ W @ B`` and ``B.T @ (w * y)`` using the sparse system (last resort)

    All three are tested here.

    """
    x, y = data_fixture
    # ensure x and y are floats
    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)
    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    weights = np.random.RandomState(0).normal(0.8, 0.05, x.size)
    weights = np.clip(weights, 0, 1).astype(float, copy=False)

    knots = _spline_utils._spline_knots(x, num_knots, spline_degree, True)
    basis = _spline_utils._spline_basis(x, knots, spline_degree)
    num_bases = basis.shape[1]
    penalty = _algorithm_setup.diff_penalty_diagonals(num_bases, diff_order)
    penalty_matrix = spdiags(
        _algorithm_setup.diff_penalty_diagonals(num_bases, diff_order, False),
        np.arange(diff_order, -(diff_order + 1), -1), num_bases, num_bases, 'csr'
    )

    expected_coeffs = spsolve(
        basis.T @ diags(weights, format='csr') @ basis + penalty_matrix,
        basis.T @ (weights * y)
    )

    with mock.patch.object(_spline_utils, '_HAS_NUMBA', True):
        # should use the numba calculation
        assert_allclose(
            _spline_utils._solve_pspline(x, y, weights, basis, penalty, knots, spline_degree),
            expected_coeffs, 1e-10, 1e-12
        )

    with mock.patch.object(_spline_utils, '_HAS_NUMBA', False):
        # should use the scipy calculation
        assert_allclose(
            _spline_utils._solve_pspline(x, y, weights, basis, penalty, knots, spline_degree),
            expected_coeffs, 1e-10, 1e-12
        )

        # mock that the scipy import failed; should use sparse calculation
        with mock.patch.object(_spline_utils, '_scipy_btb_bty', None):
            assert_allclose(
                _spline_utils._solve_pspline(x, y, weights, basis, penalty, knots, spline_degree),
                expected_coeffs, 1e-10, 1e-12
            )
