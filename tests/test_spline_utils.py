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
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve

from pybaselines import _banded_utils, _spline_utils
from pybaselines._compat import diags, dia_object


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
    assert np.all(x >= knots[spline_degree])
    assert np.all(x <= knots[len(knots) - 1 - spline_degree])
    assert np.all(np.diff(knots) >= 0)


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
@pytest.mark.parametrize('lower_only', (True, False))
def test_solve_psplines(data_fixture, num_knots, spline_degree, diff_order, lower_only):
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
    penalty = _banded_utils.diff_penalty_diagonals(num_bases, diff_order, lower_only)
    penalty_matrix = dia_object(
        (_banded_utils.diff_penalty_diagonals(num_bases, diff_order, False),
        np.arange(diff_order, -(diff_order + 1), -1)), shape=(num_bases, num_bases)
    ).tocsr()

    expected_coeffs = spsolve(
        basis.T @ diags(weights, format='csr') @ basis + penalty_matrix,
        basis.T @ (weights * y)
    )

    with mock.patch.object(_spline_utils, '_HAS_NUMBA', False):
        # mock that the scipy import failed, so should use sparse calculation; tested
        # first since it should be most stable
        with mock.patch.object(_spline_utils, '_scipy_btb_bty', None):
            assert_allclose(
                _spline_utils._solve_pspline(
                    x, y, weights, basis, penalty, knots, spline_degree, lower_only=lower_only
                ),
                expected_coeffs, 1e-10, 1e-12
            )

        # should use the scipy calculation
        assert_allclose(
            _spline_utils._solve_pspline(
                x, y, weights, basis, penalty, knots, spline_degree, lower_only=lower_only
            ),
            expected_coeffs, 1e-10, 1e-12
        )

    with mock.patch.object(_spline_utils, '_HAS_NUMBA', True):
        # should use the numba calculation
        assert_allclose(
            _spline_utils._solve_pspline(
                x, y, weights, basis, penalty, knots, spline_degree, lower_only=lower_only
            ),
            expected_coeffs, 1e-10, 1e-12
        )


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
    assert penalized_system.num_knots == num_knots
    assert penalized_system.spline_degree == spline_degree
    assert penalized_system.coef is None  # None since the solve method has not been called
    assert penalized_system.basis.shape == (data_size, num_knots + spline_degree - 1)
    assert penalized_system._num_bases == num_knots + spline_degree - 1
    assert penalized_system.knots.shape == (num_knots + 2 * spline_degree,)
    assert isinstance(penalized_system.x, np.ndarray)
    assert penalized_system._x_len == len(penalized_system.x)
    assert not penalized_system.using_pentapy
    if allow_lower:
        assert penalized_system.main_diagonal_index == 0
    else:
        assert penalized_system.main_diagonal_index == diff_order + max(0, padding)

    # check that PSpline.same_basis works as expected
    assert penalized_system.same_basis(num_knots=num_knots, spline_degree=spline_degree)
    for new_num_knots in (1, 5, 1000):
        if new_num_knots == num_knots:
            continue
        for new_spline_degree in range(4):
            if new_spline_degree == spline_degree:
                continue
            assert not penalized_system.same_basis(
                num_knots=new_num_knots, spline_degree=new_spline_degree
            )


@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (10, 100))
@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('reverse_diags', (None, True, False))
def test_pspline_setup(data_fixture, num_knots, spline_degree, diff_order,
                       allow_lower, reverse_diags):
    """
    Ensure the PSpline setup is correct.

    Since `allow_pentapy` is always False for PSpline, the `lower` attribute of the
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

    pspline = _spline_utils.PSpline(
        x, num_knots=num_knots, spline_degree=spline_degree, check_finite=False,
        lam=lam, diff_order=diff_order, allow_lower=allow_lower, reverse_diags=reverse_diags
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


def test_pspline_non_finite_fails():
    """Ensure non-finite values raise an exception when check_finite is True."""
    x = np.linspace(-1, 1, 100)
    for value in (np.nan, np.inf, -np.inf):
        x[0] = value
        with pytest.raises(ValueError):
            _spline_utils.PSpline(x, check_finite=True)


def test_pspline_diff_order_zero_fails(data_fixture):
    """Ensures a difference order of 0 fails."""
    x, y = data_fixture
    with pytest.raises(ValueError):
        _spline_utils.PSpline(x, diff_order=0)


@pytest.mark.parametrize('spline_degree', (-2, -1, 0, 1))
def test_pspline_negative_spline_degree_fails(data_fixture, spline_degree):
    """Ensures a spline degree less than 0 fails."""
    x, y = data_fixture
    if spline_degree >= 0:
        _spline_utils.PSpline(x, spline_degree=spline_degree)
    else:
        with pytest.raises(ValueError):
            _spline_utils.PSpline(x, spline_degree=spline_degree)


@pytest.mark.parametrize('spline_degree', (1, 2, 3))
@pytest.mark.parametrize('num_knots', (10, 100))
@pytest.mark.parametrize('diff_order', (1, 2))
@pytest.mark.parametrize('lam', (1e-2, 1e2))
def test_pspline_tck(data_fixture, num_knots, spline_degree, diff_order, lam):
    """Ensures the tck attribute can correctly recreate the solved spline."""
    x, y = data_fixture
    pspline = _spline_utils.PSpline(
        x, num_knots=num_knots, spline_degree=spline_degree, diff_order=diff_order, lam=lam
    )
    fit_spline = pspline.solve_pspline(y, weights=np.ones_like(y))

    # ensure tck is the knots, coefficients, and spline degree
    assert len(pspline.tck) == 3
    knots, coeffs, degree = pspline.tck

    assert_allclose(knots, pspline.knots, rtol=1e-12)
    assert_allclose(coeffs, pspline.coef, rtol=1e-12)
    assert degree == spline_degree

    # now recreate the spline with scipy's BSpline and ensure it is the same
    recreated_spline = BSpline(*pspline.tck)(x)

    assert_allclose(recreated_spline, fit_spline, rtol=1e-10)


def test_pspline_tck_none(data_fixture):
    """Ensures an exception is raised when tck attribute is accessed without first solving once."""
    x, y = data_fixture
    pspline = _spline_utils.PSpline(x)

    assert pspline.coef is None
    with pytest.raises(ValueError):
        pspline.tck


def test_pspline_tck_readonly(data_fixture):
    """Ensures the tck attribute is read-only."""
    x, y = data_fixture
    pspline = _spline_utils.PSpline(x)
    pspline.solve_pspline(y, np.ones_like(y))
    with pytest.raises(AttributeError):
        pspline.tck = (1, 2, 3)


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

    pspline = _spline_utils.PSpline(
        x, num_knots=len(x) + 1, spline_degree=0, lam=lam, diff_order=diff_order,
        check_finite=False
    )

    # sanity check to ensure it was set up correctly
    assert_array_equal(pspline.basis.shape, (len(x), len(x)))

    whittaker_system = _banded_utils.PenalizedSystem(len(y), lam=lam, diff_order=diff_order)

    # TODO replace with np.random.default_rng when min numpy version is >= 1.17
    weights = np.random.RandomState(0).normal(0.8, 0.05, len(y))
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
