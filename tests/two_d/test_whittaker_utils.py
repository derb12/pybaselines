# -*- coding: utf-8 -*-
"""Tests for pybaselines.two_d._whittaker_utils.

@author: Donald Erb
Created on Dec. 11, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import eig_banded, solve
from scipy.sparse import issparse, kron
from scipy.sparse.linalg import spsolve

from pybaselines._banded_utils import diff_penalty_diagonals
from pybaselines._compat import dia_object, identity
from pybaselines.two_d import _spline_utils, _whittaker_utils
from pybaselines.utils import difference_matrix

from ..base_tests import get_2dspline_inputs


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
def test_solve_penalized_system(small_data2d, diff_order, lam):
    """
    Tests the accuracy of the penalized system solver.

    Not really useful at the moment, but will be more useful if the solver changes
    from the current basic sparse solver.

    """
    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    num_bases = small_data2d.shape

    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    penalized_system = _whittaker_utils.PenalizedSystem2D(
        small_data2d.shape, lam=lam, diff_order=diff_order
    )

    weights = np.random.default_rng(0).normal(0.8, 0.05, small_data2d.size)
    weights = np.clip(weights, 1e-12, 1).astype(float, copy=False).ravel()

    penalty.setdiag(penalty.diagonal() + weights)

    expected_result = spsolve(penalty, weights * small_data2d.flatten())
    output = penalized_system.solve(small_data2d.flatten(), weights)

    assert_allclose(output.flatten(), expected_result, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_penalized_system_setup(small_data2d, diff_order, lam):
    """Ensure the PenalizedSystem2D setup is correct."""
    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    num_bases = small_data2d.shape

    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    penalized_system = _whittaker_utils.PenalizedSystem2D(
        small_data2d.shape, lam=lam, diff_order=diff_order
    )

    assert_array_equal(penalized_system._num_bases, num_bases)

    assert issparse(penalized_system.penalty)
    assert_allclose(
        penalized_system.penalty.toarray(), penalty.toarray(), rtol=1e-12, atol=1e-12
    )

    assert_array_equal(penalized_system.diff_order, (diff_order_x, diff_order_z))
    assert_array_equal(penalized_system.lam, (lam_x, lam_z))


@pytest.mark.parametrize('diff_order', (0, -1, [0, 0], [1, 0], [0, 1], [-1, 1], [1, -1]))
def test_penalized_system_diff_order_fails(small_data2d, diff_order):
    """Ensures a difference order of less than 1 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.PenalizedSystem2D(small_data2d.shape, diff_order=diff_order)


@pytest.mark.parametrize('lam', (-2, 0, [-1, 1], [1, -1], [1, 0], [0, 1]))
def test_penalized_system_negative_lam_fails(small_data2d, lam):
    """Ensures a lam value less than or equal to 0 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.PenalizedSystem2D(small_data2d.shape, lam=lam)


@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_compare_to_psplines(data_fixture2d, lam, diff_order):
    """
    Ensures 2D Whittaker and PSpline outputs are the same for specific condition.

    If the number of basis functions for splines is equal to the number of data points, and
    the spline degree is set to 0, then the spline basis becomes the identity function
    and should produce the same analytical equation as Whittaker smoothing.

    Since the 2D PSpline case is known from Eiler's paper, and the implementation of
    2D Whittaker smoothing in pybaselines was adapted from that, need to verify the Whittaker
    smoothing implementation.

    """
    x, z, y = data_fixture2d

    spline_basis = _spline_utils.SplineBasis2D(
        x, z, num_knots=(len(x) + 1, len(z) + 1), spline_degree=0, check_finite=False
    )
    pspline = _spline_utils.PSpline2D(spline_basis, lam=lam, diff_order=diff_order)

    # sanity check to ensure it was set up correctly
    assert_array_equal(spline_basis.basis_r.shape, (len(x), len(x)))
    assert_array_equal(spline_basis.basis_c.shape, (len(z)), len(z))

    whittaker_system = _whittaker_utils.PenalizedSystem2D(
        y.shape, lam=lam, diff_order=diff_order
    )

    weights = np.random.default_rng(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 1e-12, 1).astype(float, copy=False)

    spline_output = pspline.solve(y, weights=weights)
    whittaker_output = whittaker_system.solve(y.ravel(), weights=weights.ravel())

    assert_allclose(whittaker_output.reshape(y.shape), spline_output, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
def test_penalized_system_add_penalty(diff_order):
    """Tests adding a penalty to a PenalizedSystem2D."""
    data_size = (40, 51)
    lam = 5

    whittaker_system = _whittaker_utils.PenalizedSystem2D(
        data_size, lam=lam, diff_order=diff_order
    )
    added_penalty = 5 * identity(np.prod(data_size))

    expected_output = (added_penalty + whittaker_system.penalty).toarray()
    expected_diagonal = expected_output.diagonal()

    output = whittaker_system.add_penalty(added_penalty)

    assert_allclose(output.toarray(), expected_output, rtol=1e-12, atol=1e-13)
    # should also modify the penalty attribute
    assert_allclose(whittaker_system.penalty.toarray(), expected_output, rtol=1e-12, atol=1e-13)
    # and the main diagonal
    assert_allclose(whittaker_system.main_diagonal, expected_diagonal, rtol=1e-12, atol=1e-13)


def test_face_splitting():
    """Ensures the face-splittng algorithms works as intended."""
    basis = np.array([
        [1., 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])

    output = _whittaker_utils._face_splitting(basis)

    assert output.shape == (basis.shape[0], basis.shape[1]**2)
    assert issparse(output)

    expected_output = kron(basis, np.ones((1, basis.shape[1]))).multiply(
        kron(np.ones((1, basis.shape[1])), basis)
    )
    assert_allclose(output.toarray(), expected_output.toarray(), rtol=0, atol=1e-12)


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
def test_solve_whittaker_system_no_eigenvalues(small_data2d, diff_order, lam):
    """
    Tests the accuracy of the Whittaker system solver when not using eigendecomposition.

    Not really useful at the moment, but will be more useful if the solver changes
    from the current basic sparse solver.

    """
    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    num_bases = small_data2d.shape

    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    penalized_system = _whittaker_utils.WhittakerSystem2D(
        small_data2d.shape, lam=lam, diff_order=diff_order, num_eigens=None
    )
    assert penalized_system.coef is None

    weights = np.random.default_rng(0).normal(0.8, 0.05, small_data2d.size)
    weights = np.clip(weights, 1e-12, 1).astype(float, copy=False).ravel()

    penalty.setdiag(penalty.diagonal() + weights)

    expected_result = spsolve(penalty, weights * small_data2d.flatten())
    output = penalized_system.solve(small_data2d.flatten(), weights)

    # coef should not be updated since not using eigendecomposition
    assert penalized_system.coef is None

    assert_allclose(output.flatten(), expected_result, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_whittaker_system_setup_no_eigenvalues(small_data2d, diff_order, lam):
    """Ensure the WhittakerSystem2D setup is correct when not using eigendecomposition."""
    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    num_bases = small_data2d.shape

    D1 = difference_matrix(num_bases[0], diff_order_x)
    D2 = difference_matrix(num_bases[1], diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases[1]))
    P2 = lam_z * kron(identity(num_bases[0]), D2.T @ D2)
    penalty = P1 + P2

    penalized_system = _whittaker_utils.WhittakerSystem2D(
        small_data2d.shape, lam=lam, diff_order=diff_order, num_eigens=None
    )

    assert_array_equal(penalized_system._num_bases, num_bases)

    assert issparse(penalized_system.penalty)
    assert_allclose(
        penalized_system.penalty.toarray(), penalty.toarray(), rtol=1e-12, atol=1e-12
    )

    assert_array_equal(penalized_system.diff_order, (diff_order_x, diff_order_z))
    assert_array_equal(penalized_system.lam, (lam_x, lam_z))


def check_orthogonality(eigenvectors):
    """
    Ensures each eigenvector is orthogonal to all other eigenvectors.

    Orthogonality should be ensured from the corresponding SciPy eigensolvers, but
    check to be sure.
    """
    for i, col in enumerate(eigenvectors.T):
        for j, col_2 in enumerate(eigenvectors.T):
            if i == j:
                continue
            assert_allclose(np.dot(col, col_2), 0, rtol=0, atol=1e-13)


@pytest.mark.parametrize('num_eigens', (5, 8, (5, 8)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_whittaker_system_setup_eigenvalues(data_fixture2d, num_eigens, diff_order, lam):
    """Ensure the WhittakerSystem2D setup is correct when using eigendecomposition."""
    x, z, y = data_fixture2d
    (
        num_eigens_r, num_eigens_c, _, _,
        lam_r, lam_c, diff_order_r, diff_order_c
    ) = get_2dspline_inputs(num_knots=num_eigens, lam=lam, diff_order=diff_order)

    whittaker_system = _whittaker_utils.WhittakerSystem2D(
        y.shape, lam=lam, diff_order=diff_order, num_eigens=num_eigens
    )

    assert_array_equal(whittaker_system._num_bases, num_eigens)

    eigenvalues_rows, expected_basis_rows = eig_banded(
        diff_penalty_diagonals(y.shape[0], diff_order_r, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_r - 1)
    )
    check_orthogonality(expected_basis_rows)

    eigenvalues_rows[:diff_order_r] = 0
    penalty_rows = kron(
        lam_r * dia_object((eigenvalues_rows, 0), shape=(num_eigens_r, num_eigens_r)),
        identity(num_eigens_c)
    )
    eigenvalues_cols, expected_basis_cols = eig_banded(
        diff_penalty_diagonals(y.shape[1], diff_order_c, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_c - 1)
    )
    check_orthogonality(expected_basis_cols)

    eigenvalues_cols[:diff_order_c] = 0
    penalty_cols = kron(
        identity(num_eigens_r),
        lam_c * dia_object((eigenvalues_cols, 0), shape=(num_eigens_c, num_eigens_c))
    )
    expected_basis = kron(expected_basis_rows, expected_basis_cols)

    assert whittaker_system.penalty.shape == (num_eigens_r * num_eigens_c,)
    assert_allclose(
        whittaker_system.penalty, (penalty_rows + penalty_cols).diagonal(), rtol=1e-12, atol=1e-12
    )
    # TODO comparing the non-absolute values of the coefficients fails for diff_order=1 since
    # that uses eigh_tridiagonal; the only difference is that the eigenvectors are
    # -1 * eigenvectors of eig_banded for the first `diff_order` eigenvectors; this does
    # not affect the calculated baselines, nor the degree of freedom calculation, so is
    # it worth "fixing"?
    if 1 in (diff_order_r, diff_order_c):
        assert_allclose(
            np.abs(whittaker_system.basis_r), np.abs(expected_basis_rows),
            rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            np.abs(whittaker_system.basis_c), np.abs(expected_basis_cols),
            rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            np.abs(whittaker_system.basis.toarray()), np.abs(expected_basis.toarray()),
            rtol=1e-12, atol=1e-12
        )
    else:
        assert_allclose(
            whittaker_system.basis_r, expected_basis_rows,
            rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            whittaker_system.basis_c, expected_basis_cols,
            rtol=1e-12, atol=1e-12
        )
        assert_allclose(
            whittaker_system.basis.toarray(), expected_basis.toarray(),
            rtol=1e-12, atol=1e-12
        )

    assert_array_equal(whittaker_system.diff_order, (diff_order_r, diff_order_c))
    assert_array_equal(whittaker_system.lam, (lam_r, lam_c))


@pytest.mark.parametrize('diff_order', (0, -1, [0, 0], [1, 0], [0, 1], [-1, 1], [1, -1]))
def test_whittaker_system_diff_order_fails(small_data2d, diff_order):
    """Ensures a difference order of less than 1 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, diff_order=diff_order, num_eigens=None
        )
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, diff_order=diff_order, num_eigens=(5, 5)
        )


@pytest.mark.parametrize('lam', (-2, 0, [-1, 1], [1, -1], [1, 0], [0, 1]))
def test_whittaker_system_negative_lam_fails(small_data2d, lam):
    """Ensures a lam value less than or equal to 0 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(small_data2d.shape, lam=lam, num_eigens=None)
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, lam=lam, num_eigens=(5, 5)
        )


@pytest.mark.parametrize('num_eigens', (-2, 0, [-1, 10], [10, -1], [10, 0], [0, 10]))
def test_whittaker_system_negative_num_eigens_fails(small_data2d, num_eigens):
    """Ensures a num_eigens value less than or equal to 0 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, num_eigens=num_eigens
        )


@pytest.mark.parametrize('num_eigens', ([None, 5], [3, None], np.array([None, 6])))
def test_whittaker_system_None_and_nonNone_num_eigens_fails(small_data2d, num_eigens):
    """Ensures a num_eigens cannot mix None with a non-None value."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, num_eigens=num_eigens
        )


def test_whittaker_system_too_many_num_eigens_fails():
    """Ensures a num_eigens value greater than the data size fails."""
    data_size = (30, 40)
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(data_size, num_eigens=(5, 41))
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(data_size, num_eigens=(31, 5))
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(data_size, num_eigens=(31, 41))


def test_whittaker_system_too_few_num_eigens_fails():
    """Ensures a num_eigens value less than or equal to diff_order fails."""
    data_size = (30, 40)
    diff_order = (3, 2)
    for num_eigens in ((3, 2), (4, 2), (4, 1), (3, 3), (2, 3)):
        with pytest.raises(ValueError):
            _whittaker_utils.WhittakerSystem2D(
                data_size, diff_order=diff_order, num_eigens=num_eigens
            )

    diff_order = (1, 2)
    for num_eigens in ((1, 2), (4, 2), (4, 1), (1, 3)):
        with pytest.raises(ValueError):
            _whittaker_utils.WhittakerSystem2D(
                data_size, diff_order=diff_order, num_eigens=num_eigens
            )


def test_whittaker_system_same_basis():
    """Ensures WhittakerSystem2D.same_basis works correctly."""
    data_size = (30, 35)
    num_eigens = (10, 15)
    diff_order = (2, 3)

    whittaker_system = _whittaker_utils.WhittakerSystem2D(
        data_size, num_eigens=num_eigens, diff_order=diff_order
    )

    assert whittaker_system.same_basis(diff_order, num_eigens)
    assert not whittaker_system.same_basis(diff_order[::-1], num_eigens)
    assert not whittaker_system.same_basis(diff_order, num_eigens[::-1])
    assert not whittaker_system.same_basis(2, num_eigens)
    assert not whittaker_system.same_basis(diff_order, 10)


@pytest.mark.parametrize('num_eigens', (5, 8, (5, 8)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
def test_solve_whittaker_system_eigenvalues(data_fixture2d, num_eigens, diff_order, lam):
    """
    Tests the accuracy of the Whittaker system solver when using eigendecomposition.

    Uses the nieve way to solve 2D Whittaker system as the expected result, which
    uses the flattened `y` and weight values, while pybaselines uses the second, more efficient
    method in Eiler's paper which directly uses the 2D `y` and weights.

    References
    ----------
    Eilers, P., et al. Fast and compact smoothing on large multidimensional grids. Computational
    Statistics and Data Analysis, 2006, 50(1), 61-76.

    """
    x, z, y = data_fixture2d
    (
        num_eigens_r, num_eigens_c, _, _,
        lam_x, lam_z, diff_order_x, diff_order_z
    ) = get_2dspline_inputs(num_knots=num_eigens, lam=lam, diff_order=diff_order)

    eigenvalues_rows, basis_r = eig_banded(
        diff_penalty_diagonals(y.shape[0], diff_order_x, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_r - 1)
    )
    penalty_rows = kron(
        lam_x * dia_object((eigenvalues_rows, 0), shape=(num_eigens_r, num_eigens_r)),
        identity(num_eigens_c)
    )

    eigenvalues_cols, basis_c = eig_banded(
        diff_penalty_diagonals(y.shape[1], diff_order_z, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_c - 1)
    )
    penalty_cols = kron(
        identity(num_eigens_r),
        lam_z * dia_object((eigenvalues_cols, 0), shape=(num_eigens_c, num_eigens_c))
    )
    penalty = penalty_rows + penalty_cols

    num_bases = (basis_r.shape[1], basis_c.shape[1])
    weights = np.random.default_rng(0).normal(0.8, 0.05, y.size)
    weights = np.clip(weights, 0, 1, dtype=float)

    basis = kron(basis_r, basis_c)
    CWT = basis.multiply(
        np.repeat(weights.flatten(), num_bases[0] * num_bases[1]).reshape(len(x) * len(z), -1)
    ).T

    expected_coeffs = solve(
        (CWT @ basis + penalty).toarray(), CWT @ y.flatten(),
        lower=True, overwrite_a=True, overwrite_b=True, check_finite=False, assume_a='pos'
    )
    expected_result = basis @ expected_coeffs

    whittaker_system = _whittaker_utils.WhittakerSystem2D(
        y.shape, lam=lam, diff_order=diff_order, num_eigens=num_eigens
    )

    assert whittaker_system.coef is None
    output = whittaker_system.solve(y, weights=weights.reshape(y.shape))

    assert_allclose(output.flatten(), expected_result, rtol=1e-8, atol=1e-8)
    # TODO comparing the non-absolute values of the coefficients fails for diff_order=1 since
    # that uses eigh_tridiagonal; the only difference is that the eigenvectors are
    # -1 * eigenvectors of eig_banded for the first `diff_order` eigenvectors; this does
    # not affect the calculated baselines, nor the degree of freedom calculation, so is
    # it worth "fixing"?
    if 1 in (diff_order_x, diff_order_z):
        assert_allclose(
            np.abs(whittaker_system.coef), np.abs(expected_coeffs), rtol=1e-8, atol=1e-8
        )
    else:
        assert_allclose(whittaker_system.coef, expected_coeffs, rtol=1e-8, atol=1e-8)

    # also ensure that the whittaker_systems's basis can use the solved coefficients
    basis_output = whittaker_system.basis @ whittaker_system.coef
    assert_allclose(basis_output, expected_result, rtol=1e-8, atol=1e-8)

    BT_W_B = CWT @ basis
    expected_dof = solve(
        (BT_W_B + penalty).toarray(), BT_W_B.toarray(), lower=True, overwrite_a=True,
        overwrite_b=True, check_finite=False, assume_a='gen'
    ).diagonal().reshape(num_bases)

    # and check the effective degrees of freedom are correct
    dof = whittaker_system._calc_dof(weights.reshape(y.shape))

    assert_allclose(dof, expected_dof, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('num_eigens', (5, 8, (5, 8)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
@pytest.mark.parametrize('lam_new', (1e-2, 1e2, (1e1, 1e2)))
def test_whittaker_system_update_penalty(data_fixture2d, num_eigens, diff_order, lam, lam_new):
    """Ensures the update_penalty method correctly assigns the new lam values."""
    x, z, y = data_fixture2d
    _, _, _, _, lam_r_new, lam_c_new, _, _ = get_2dspline_inputs(lam=lam_new)

    (
        num_eigens_r, num_eigens_c, _, _,
        lam_r, lam_c, diff_order_r, diff_order_c
    ) = get_2dspline_inputs(num_knots=num_eigens, lam=lam, diff_order=diff_order)

    eigenvalues_rows, expected_basis_rows = eig_banded(
        diff_penalty_diagonals(y.shape[0], diff_order_r, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_r - 1)
    )
    eigenvalues_rows[:diff_order_r] = 0
    penalty_rows = kron(
        lam_r * dia_object((eigenvalues_rows, 0), shape=(num_eigens_r, num_eigens_r)),
        identity(num_eigens_c)
    )

    eigenvalues_cols, expected_basis_cols = eig_banded(
        diff_penalty_diagonals(y.shape[1], diff_order_c, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, num_eigens_c - 1)
    )
    eigenvalues_cols[:diff_order_c] = 0
    penalty_cols = kron(
        identity(num_eigens_r),
        lam_c * dia_object((eigenvalues_cols, 0), shape=(num_eigens_c, num_eigens_c))
    )

    whittaker_system = _whittaker_utils.WhittakerSystem2D(
        y.shape, lam=lam, diff_order=diff_order, num_eigens=num_eigens
    )

    assert_allclose(
        whittaker_system.penalty, (penalty_rows + penalty_cols).diagonal(),
        rtol=1e-12, atol=1e-12
    )

    whittaker_system.update_penalty(lam=lam_new)

    new_penalty_rows = (lam_r_new / lam_r) * penalty_rows
    new_penalty_cols = (lam_c_new / lam_c) * penalty_cols
    assert_allclose(
        whittaker_system.penalty, (new_penalty_rows + new_penalty_cols).diagonal(),
        rtol=1e-12, atol=1e-12
    )
