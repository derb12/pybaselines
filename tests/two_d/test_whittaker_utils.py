# -*- coding: utf-8 -*-
"""Tests for pybaselines._banded_utils.

@author: Donald Erb
Created on Dec. 11, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import identity, issparse, kron
from scipy.sparse.linalg import spsolve

from pybaselines.two_d import _spline_utils, _whittaker_utils
from pybaselines.utils import difference_matrix

from ..conftest import get_2dspline_inputs


@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('lam', (1e-2, 1e2, (1e1, 1e2)))
def test_solve_penalized_system(small_data2d, diff_order, lam):
    """
    Tests the accuracy of the penalized system solver.

    Not really useful at the moment, but will be mroe useful if the solver changes
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

    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    weights = np.random.RandomState(0).normal(0.8, 0.05, small_data2d.size)
    weights = np.clip(weights, 0, 1).astype(float, copy=False).ravel()

    penalty.setdiag(penalty.diagonal() + weights)
    penalized_system.penalty.setdiag(penalized_system.penalty.diagonal() + weights)

    expected_result = spsolve(penalty, weights * small_data2d.flatten())
    output = penalized_system.solve(penalized_system.penalty, weights * small_data2d.flatten())

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

    assert_allclose(penalized_system.penalty.toarray(), penalty.toarray(), rtol=1e-12, atol=1e-12)

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
    assert_array_equal(pspline.basis_x.shape, (len(x), len(x)))
    assert_array_equal(pspline.basis_z.shape, (len(z)), len(z))

    whittaker_system = _whittaker_utils.PenalizedSystem2D(y.shape, lam=lam, diff_order=diff_order)

    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    weights = np.random.RandomState(0).normal(0.8, 0.05, y.shape)
    weights = np.clip(weights, 0, 1).astype(float, copy=False)

    whittaker_system.penalty.setdiag(whittaker_system.penalty.diagonal() + weights.ravel())

    spline_output = pspline.solve_pspline(y, weights=weights)
    whittaker_output = whittaker_system.solve(whittaker_system.penalty, weights.ravel() * y.ravel())

    assert_allclose(whittaker_output.reshape(y.shape), spline_output, rtol=1e-12, atol=1e-12)
