# -*- coding: utf-8 -*-
"""Tests for pybaselines._banded_utils.

@author: Donald Erb
Created on Dec. 11, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import eig_banded
from scipy.sparse import issparse, kron
from scipy.sparse.linalg import spsolve

from pybaselines._banded_utils import diff_penalty_diagonals
from pybaselines._compat import identity, dia_object
from pybaselines.two_d import _spline_utils, _whittaker_utils
from pybaselines.utils import difference_matrix

from ..conftest import get_2dspline_inputs


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

    pspline = _spline_utils.PSpline2D(
        x, z, num_knots=(len(x) + 1, len(z) + 1), spline_degree=0, lam=lam, diff_order=diff_order,
        check_finite=False
    )

    # sanity check to ensure it was set up correctly
    assert_array_equal(pspline.basis_r.shape, (len(x), len(x)))
    assert_array_equal(pspline.basis_c.shape, (len(z)), len(z))

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
def test_solve_whittaker_system(small_data2d, diff_order, lam):
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
        small_data2d.shape, lam=lam, diff_order=diff_order, max_eigens=None
    )

    weights = np.random.default_rng(0).normal(0.8, 0.05, small_data2d.size)
    weights = np.clip(weights, 1e-12, 1).astype(float, copy=False).ravel()

    penalty.setdiag(penalty.diagonal() + weights)

    expected_result = spsolve(penalty, weights * small_data2d.flatten())
    output = penalized_system.solve(small_data2d.flatten(), weights)

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
        small_data2d.shape, lam=lam, diff_order=diff_order, max_eigens=None
    )

    assert_array_equal(penalized_system._num_bases, num_bases)

    assert issparse(penalized_system.penalty)
    assert_allclose(
        penalized_system.penalty.toarray(), penalty.toarray(), rtol=1e-12, atol=1e-12
    )

    assert_array_equal(penalized_system.diff_order, (diff_order_x, diff_order_z))
    assert_array_equal(penalized_system.lam, (lam_x, lam_z))


@pytest.mark.parametrize('diff_order', (1, 2, 3, [1, 3]))
@pytest.mark.parametrize('lam', (5, (3, 5)))
def test_whittaker_system_setup_eigenvalues(small_data2d, diff_order, lam):
    """Ensure the WhittakerSystem2D setup is correct when using eigendecomposition."""
    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )
    max_eigens = np.array([5, 10])

    penalized_system = _whittaker_utils.WhittakerSystem2D(
        small_data2d.shape, lam=lam, diff_order=diff_order, max_eigens=max_eigens
    )

    assert_array_equal(penalized_system._num_bases, max_eigens)

    eigenvalues_rows, expected_basis_rows = eig_banded(
        diff_penalty_diagonals(small_data2d.shape[0], diff_order_x, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, max_eigens[0] - 1)
    )
    penalty_rows = kron(
        lam_x * dia_object((eigenvalues_rows, 0), shape=(max_eigens[0], max_eigens[0])),
        identity(max_eigens[1])
    )

    eigenvalues_cols, expected_basis_cols = eig_banded(
        diff_penalty_diagonals(small_data2d.shape[1], diff_order_z, lower_only=True),
        lower=True, overwrite_a_band=True, select='i', select_range=(0, max_eigens[1] - 1)
    )
    penalty_cols = kron(
        identity(max_eigens[0]),
        lam_z * dia_object((eigenvalues_cols, 0), shape=(max_eigens[1], max_eigens[1]))
    )

    assert penalized_system.penalty.shape == (np.prod(max_eigens),)
    assert_allclose(
        penalized_system.penalty, (penalty_rows + penalty_cols).diagonal(), rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        penalized_system.basis_r, expected_basis_rows, rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        penalized_system.basis_c, expected_basis_cols, rtol=1e-12, atol=1e-12
    )

    assert_array_equal(penalized_system.diff_order, (diff_order_x, diff_order_z))
    assert_array_equal(penalized_system.lam, (lam_x, lam_z))


@pytest.mark.parametrize('diff_order', (0, -1, [0, 0], [1, 0], [0, 1], [-1, 1], [1, -1]))
def test_whittaker_system_diff_order_fails(small_data2d, diff_order):
    """Ensures a difference order of less than 1 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, diff_order=diff_order, max_eigens=None
        )
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, diff_order=diff_order, max_eigens=(5, 5)
        )


@pytest.mark.parametrize('lam', (-2, 0, [-1, 1], [1, -1], [1, 0], [0, 1]))
def test_whittaker_system_negative_lam_fails(small_data2d, lam):
    """Ensures a lam value less than or equal to 0 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(small_data2d.shape, lam=lam, max_eigens=None)
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, lam=lam, max_eigens=(5, 5)
        )


@pytest.mark.parametrize('max_eigens', (-2, 0, [-1, 1], [1, -1], [1, 0], [0, 1]))
def test_whittaker_system_negative_maxeigens_fails(small_data2d, max_eigens):
    """Ensures a max_eigens value less than or equal to 0 fails."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, max_eigens=max_eigens
        )


@pytest.mark.parametrize('max_eigens', ([None, 5], [3, None], np.array([None, 6])))
def test_whittaker_system_None_and_nonNone_maxeigens_fails(small_data2d, max_eigens):
    """Ensures a max_eigens cannot mix None with a non-None value."""
    with pytest.raises(ValueError):
        _whittaker_utils.WhittakerSystem2D(
            small_data2d.shape, max_eigens=max_eigens
        )
