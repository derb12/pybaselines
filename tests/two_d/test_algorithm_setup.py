# -*- coding: utf-8 -*-
"""Tests for pybaselines.two_d._algorithm_setup.

@author: Donald Erb
Created on January 5, 2024

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import kron

from pybaselines._compat import identity
from pybaselines.two_d import Baseline2D, _algorithm_setup, _spline_utils, _whittaker_utils
from pybaselines.results import PSplineResult2D, WhittakerResult2D
from pybaselines.utils import ParameterWarning, SortingWarning, difference_matrix, estimate_window
from pybaselines._validation import _check_scalar

from ..base_tests import ensure_deprecation, get_2dspline_inputs, get_data2d


@pytest.fixture
def algorithm(small_data2d):
    """
    An _Algorithm2D class with x-data set to np.arange(10) and z-data set to np.arange(20).

    Returns
    -------
    pybaselines.two_d._algorithm_setup._Algorithm2D
        An _Algorithm2D class for testing.
    """
    num_x, num_z = small_data2d.shape
    return _algorithm_setup._Algorithm2D(
        x_data=np.arange(num_x), z_data=np.arange(num_z), assume_sorted=True, check_finite=False
    )


@pytest.mark.parametrize('diff_order', (1, 2, 3, (2, 3)))
@pytest.mark.parametrize('lam', (1, 20, (2, 5)))
def test_setup_whittaker_diff_matrix(data_fixture2d, lam, diff_order):
    """Ensures output difference matrix diagonal data is in desired format."""
    x, z, y = data_fixture2d

    algorithm = _algorithm_setup._Algorithm2D(x, z)

    _, _, whittaker_system = algorithm._setup_whittaker(y, lam=lam, diff_order=diff_order)

    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    D1 = difference_matrix(len(x), diff_order_x)
    D2 = difference_matrix(len(z), diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(len(z)))
    P2 = lam_z * kron(identity(len(x)), D2.T @ D2)
    expected_penalty = P1 + P2

    assert_allclose(
        whittaker_system.penalty.toarray(),
        expected_penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize('num_eigens', (None, 3))
@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
@pytest.mark.parametrize('has_mask', (True, False))
def test_setup_whittaker_weights(small_data2d, algorithm, num_eigens, weight_enum, has_mask):
    """Ensures output weight array is correct."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d)
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
        desired_weights = np.arange(small_data2d.size)
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape).tolist()
        desired_weights = np.arange(small_data2d.size)

    if has_mask:
        mask = np.zeros(algorithm._shape, dtype=bool)
        mask[3:9] = True
        algorithm.mask = mask
        desired_weights = np.where(mask.ravel(), 0., desired_weights)

    if num_eigens is not None:
        desired_weights = desired_weights.reshape(small_data2d.shape)
        expected_y = small_data2d
    else:
        expected_y = small_data2d.ravel()

    y, weight_array, _ = algorithm._setup_whittaker(
        small_data2d, lam=1, diff_order=2, weights=weights, num_eigens=num_eigens
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert_allclose(y, expected_y, rtol=1e-14, atol=1e-14)
    assert weight_array.dtype == float


def test_setup_whittaker_wrong_weight_shape(small_data2d, algorithm):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(np.array(small_data2d.shape) + 1)
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data2d, lam=1, diff_order=2, weights=weights)


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_whittaker_diff_matrix_fails(small_data2d, algorithm, diff_order):
    """Ensures using a diff_order < 1 with _setup_whittaker raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data2d, lam=1, diff_order=diff_order)


@pytest.mark.parametrize('diff_order', (4, 5))
def test_setup_whittaker_diff_matrix_warns(small_data2d, algorithm, diff_order):
    """Ensures using a diff_order > 3 with _setup_whittaker raises a warning."""
    with pytest.warns(ParameterWarning):
        algorithm._setup_whittaker(small_data2d, lam=1, diff_order=diff_order)


def test_setup_whittaker_negative_lam_fails(small_data2d, algorithm):
    """Ensures a negative lam value fails."""
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data2d, lam=-1)


def test_setup_whittaker_array_lam(small_data2d):
    """Ensures a lam that is a single array of one or two values passes while larger arrays fail."""
    num_x, num_z = small_data2d.shape
    _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_whittaker(
        small_data2d, lam=[1]
    )
    _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_whittaker(
        small_data2d, lam=[1, 2]
    )
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_whittaker(
            small_data2d, lam=[1, 2, 3]
        )


@pytest.mark.parametrize('has_mask', (True, False))
@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_polynomial_weights(small_data2d, algorithm, weight_enum, has_mask):
    """Ensures output weight array is correctly handled."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d)
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
        desired_weights = np.arange(small_data2d.size)
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape).tolist()
        desired_weights = np.arange(small_data2d.size)

    if has_mask:
        mask = np.zeros(algorithm._shape, dtype=bool)
        mask[3:9] = True
        algorithm.mask = mask
        desired_weights = np.where(mask.ravel(), 0., desired_weights)

    y, weight_array = algorithm._setup_polynomial(small_data2d, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert_allclose(y, small_data2d.ravel(), rtol=1e-14, atol=1e-14)
    assert weight_array.dtype == float


def test_setup_polynomial_wrong_weight_shape(small_data2d, algorithm):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(np.array(small_data2d.shape) + 1)
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, weights=weights)


@pytest.mark.parametrize('poly_order', (0, 2, 4, (2, 4)))
@pytest.mark.parametrize('vander_enum', (0, 1, 2, 3))
@pytest.mark.parametrize('include_pinv', (True, False))
def test_setup_polynomial_vandermonde(small_data2d, algorithm, vander_enum, include_pinv,
                                      poly_order):
    """Ensures that the Vandermonde matrix and the pseudo-inverse matrix are correct."""
    if vander_enum == 0:
        # no weights specified
        weights = None
    elif vander_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d)
    elif vander_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
    elif vander_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape).tolist()

    output = algorithm._setup_polynomial(
        small_data2d, weights=weights, poly_order=poly_order, calc_vander=True,
        calc_pinv=include_pinv
    )
    if include_pinv:
        _, weight_array, pinv_matrix = output
    else:
        _, weight_array = output

    if isinstance(poly_order, int):
        x_order = poly_order
        z_order = poly_order
    else:
        x_order, z_order = poly_order

    mapped_x = np.polynomial.polyutils.mapdomain(algorithm.x, algorithm.x_domain, [-1, 1])
    mapped_z = np.polynomial.polyutils.mapdomain(algorithm.z, algorithm.z_domain, [-1, 1])
    desired_vander = np.polynomial.polynomial.polyvander2d(
        *np.meshgrid(mapped_x, mapped_z, indexing='ij'), (x_order, z_order)
    ).reshape((-1, (x_order + 1) * (z_order + 1)))
    assert_allclose(algorithm._polynomial.vandermonde, desired_vander, 1e-12)

    if include_pinv:
        desired_pinv = np.linalg.pinv(np.sqrt(weight_array)[:, np.newaxis] * desired_vander)
        assert_allclose(pinv_matrix, desired_pinv, 1e-10)
        if weights is None:
            assert_allclose(pinv_matrix, algorithm._polynomial.pseudo_inverse, 1e-10)


def test_setup_polynomial_negative_polyorder_fails(small_data2d, algorithm):
    """Ensures a negative poly_order raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=-1)

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=[1, -1])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=[-1, 1])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=[-1, -1])


def test_setup_polynomial_too_large_polyorder_fails(small_data2d, algorithm):
    """Ensures an exception is raised if poly_order has more than two values."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=[1, 2, 3])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=[1, 2, 3, 4])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, poly_order=np.array([1, 2, 3]))


def test_setup_polynomial_maxcross(small_data2d):
    """Ensures the _max_cross attribute is updated after calling _setup_polynomial."""
    num_x, num_z = small_data2d.shape
    algorithm = _algorithm_setup._Algorithm2D(
        x_data=np.arange(num_x), z_data=np.arange(num_z), assume_sorted=True, check_finite=False
    )
    algorithm._setup_polynomial(small_data2d, max_cross=[1], calc_vander=True)
    assert algorithm._polynomial.max_cross == 1

    algorithm._setup_polynomial(small_data2d, max_cross=1, calc_vander=True)
    assert algorithm._polynomial.max_cross == 1

    algorithm._setup_polynomial(small_data2d, max_cross=0, calc_vander=False)
    # should not update the _polynomial since Vandermonde is not calculated
    assert algorithm._polynomial.max_cross == 1

    algorithm._setup_polynomial(small_data2d, max_cross=0, calc_vander=True)
    assert algorithm._polynomial.max_cross == 0

    algorithm._setup_polynomial(small_data2d, max_cross=None, calc_vander=True)
    assert algorithm._polynomial.max_cross is None


def test_setup_polynomial_too_large_maxcross_fails(small_data2d, algorithm):
    """Ensures an exception is raised if max_cross has more than one value."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, max_cross=[1, 2], calc_vander=True)

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, max_cross=[1, 2, 3], calc_vander=True)

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, max_cross=np.array([1, 2]), calc_vander=True)


def test_setup_polynomial_negative_maxcross_fails(small_data2d, algorithm):
    """Ensures an exception is raised if max_cross is negative."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, max_cross=[-1], calc_vander=True)

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data2d, max_cross=-2, calc_vander=True)


@pytest.mark.parametrize('half_window', (None, 2, (2, 2)))
def test_setup_morphology(data_fixture2d, algorithm, half_window):
    """
    Ensures setup_morphology works as expected.

    Note that a half window of 2 was selected since it should not be the output
    of estimate_window; setup_morphology should just pass the half window back
    out if it was not None.
    """
    x, z, y = data_fixture2d
    y_out, half_window_out = algorithm._setup_morphology(y, half_window)
    if half_window is None:
        half_window_expected = estimate_window(y)
    else:
        half_window_expected = _check_scalar(half_window, 2, fill_scalar=True, dtype=int)[0]
        # sanity check that the calculated half window does not match the test case one
        assert not np.array_equal(half_window, estimate_window(y))

    assert np.array_equal(half_window_out, half_window_expected)
    assert y is y_out  # should not be modified by setup_morphology


@pytest.mark.parametrize('half_window', (-1, 0))
def test_setup_morphology_bad_hw_fails(small_data2d, algorithm, half_window):
    """Ensures half windows less than 1 raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_morphology(small_data2d, half_window=half_window)


@ensure_deprecation(1, 4)
def test_setup_morphology_kwargs_warns(small_data2d, algorithm):
    """Ensures passing keyword arguments is deprecated."""
    with pytest.warns(DeprecationWarning):
        algorithm._setup_morphology(small_data2d, min_half_window=2)

    # also ensure both window_kwargs and **kwargs are passed to estimate_window
    with pytest.raises(TypeError):
        with pytest.warns(DeprecationWarning):
            algorithm._setup_morphology(
                small_data2d, window_kwargs={'min_half_window': 2}, min_half_window=2
            )


def test_setup_smooth_shape(small_data2d, algorithm):
    """Ensures output y is correctly padded."""
    pad_length = 4
    y, hw = algorithm._setup_smooth(small_data2d, pad_length, pad_kwargs={'mode': 'edge'})
    assert_array_equal(
        y.shape, (small_data2d.shape[0] + 2 * pad_length, small_data2d.shape[1] + 2 * pad_length)
    )
    assert_array_equal(hw, [pad_length, pad_length])


@pytest.mark.parametrize('half_window', (-1, 0))
def test_setup_smooth_bad_hw_fails(small_data2d, algorithm, half_window):
    """Ensures half windows less than 1 raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_smooth(small_data2d, half_window=half_window)


@ensure_deprecation(1, 4)
def test_setup_smooth_kwargs_warns(small_data2d, algorithm):
    """Ensures passing keyword arguments is deprecated."""
    with pytest.warns(DeprecationWarning):
        algorithm._setup_smooth(small_data2d, extrapolate_window=2)

    # also ensure both pad_kwargs and **kwargs are passed to pad_edges
    with pytest.raises(TypeError):
        with pytest.warns(DeprecationWarning):
            algorithm._setup_smooth(
                small_data2d, pad_kwargs={'extrapolate_window': 2}, extrapolate_window=2
            )


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
@pytest.mark.parametrize('has_mask', (True, False))
def test_setup_classification_weights(small_data2d, algorithm, weight_enum, has_mask):
    """Ensures output weight array is correctly handled in classification setup."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data2d, dtype=bool)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d, dtype=bool)
        desired_weights = np.ones_like(small_data2d, dtype=bool)
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).astype(bool).reshape(small_data2d.shape)
        desired_weights = np.arange(small_data2d.size).astype(bool).reshape(small_data2d.shape)
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).astype(bool).reshape(small_data2d.shape).tolist()
        desired_weights = np.arange(small_data2d.size).astype(bool).reshape(small_data2d.shape)

    if has_mask:
        mask = np.zeros(algorithm._shape, dtype=bool)
        mask[3:9] = True
        algorithm.mask = mask
        desired_weights = np.where(mask, False, desired_weights)

    _, weight_array = algorithm._setup_classification(small_data2d, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert weight_array.dtype == bool


@pytest.mark.parametrize('num_knots', (10, 30, (20, 30)))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4, (2, 3)))
def test_setup_spline_spline_basis(data_fixture2d, num_knots, spline_degree):
    """Ensures the spline basis function is correctly created."""
    x, z, y = data_fixture2d
    fitter = _algorithm_setup._Algorithm2D(x, z)
    assert fitter._spline_basis is None

    fitter._setup_spline(
        y, weights=None, spline_degree=spline_degree, num_knots=num_knots
    )

    if isinstance(num_knots, int):
        num_knots_r = num_knots
        num_knots_c = num_knots
    else:
        num_knots_r, num_knots_c = num_knots
    if isinstance(spline_degree, int):
        spline_degree_x = spline_degree
        spline_degree_z = spline_degree
    else:
        spline_degree_x, spline_degree_z = spline_degree

    assert_array_equal(
        fitter._spline_basis.basis_r.shape,
        (len(x), num_knots_r + spline_degree_x - 1)
    )
    assert_array_equal(
        fitter._spline_basis.basis_c.shape,
        (len(z), num_knots_c + spline_degree_z - 1)
    )


@pytest.mark.parametrize('lam', (1, 20, (3, 10)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('num_knots', (20, (21, 30)))
def test_setup_spline_diff_matrix(data_fixture2d, lam, diff_order, spline_degree, num_knots):
    """Ensures output difference matrix diagonal data is in desired format."""
    x, z, y = data_fixture2d

    algorithm = _algorithm_setup._Algorithm2D(x, z)
    _, _, pspline = algorithm._setup_spline(
        y, weights=None, spline_degree=spline_degree, num_knots=num_knots,
        diff_order=diff_order, lam=lam
    )

    (
        num_knots_r, num_knots_c, spline_degree_x, spline_degree_z,
        lam_x, lam_z, diff_order_x, diff_order_z
    ) = get_2dspline_inputs(
        num_knots=num_knots, spline_degree=spline_degree, lam=lam, diff_order=diff_order
    )

    num_bases_x = num_knots_r + spline_degree_x - 1
    num_bases_z = num_knots_c + spline_degree_z - 1

    D1 = difference_matrix(num_bases_x, diff_order_x)
    D2 = difference_matrix(num_bases_z, diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases_z))
    P2 = lam_z * kron(identity(num_bases_x), D2.T @ D2)
    expected_penalty = P1 + P2

    assert_allclose(
        pspline.penalty.toarray(),
        expected_penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (5, 50, 100))
def test_setup_spline_too_high_diff_order(small_data2d, spline_degree, num_knots):
    """
    Ensures an exception is raised when the difference order is >= number of basis functions.

    The number of basis functions is equal to the number of knots + the spline degree - 1.
    Tests both difference order equal to and greater than the number of basis functions.

    """
    num_z, num_x = small_data2d.shape
    diff_order = num_knots + spline_degree - 1
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, weights=None, spline_degree=spline_degree, num_knots=num_knots,
            penalized=True, diff_order=diff_order
        )

    diff_order += 1
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, weights=None, spline_degree=spline_degree, num_knots=num_knots,
            penalized=True, diff_order=diff_order
        )


@pytest.mark.parametrize('num_knots', (0, 1))
def test_setup_spline_too_few_knots(small_data2d, num_knots):
    """Ensures an error is raised if the number of knots is less than 2."""
    num_x, num_z = small_data2d.shape
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, weights=None, spline_degree=3, num_knots=num_knots,
            penalized=True, diff_order=1
        )


def test_setup_spline_wrong_weight_shape(small_data2d):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(np.array(small_data2d.shape) + 1)
    num_x, num_z = small_data2d.shape
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, weights=weights
        )


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_spline_diff_matrix_fails(small_data2d, diff_order):
    """Ensures using a diff_order < 1 with _setup_spline raises an exception."""
    num_x, num_z = small_data2d.shape
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, diff_order=diff_order
        )


@pytest.mark.parametrize('diff_order', (5, 6))
def test_setup_spline_diff_matrix_warns(small_data2d, diff_order):
    """Ensures using a diff_order > 4 with _setup_spline raises a warning."""
    num_x, num_z = small_data2d.shape
    with pytest.warns(ParameterWarning):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, diff_order=diff_order
        )


def test_setup_spline_negative_lam_fails(small_data2d):
    """Ensures a negative lam value fails."""
    num_x, num_z = small_data2d.shape
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, lam=-1
        )


def test_setup_spline_array_lam(small_data2d):
    """Ensures a lam that is a single array of one or two values passes while larger arrays fail."""
    num_x, num_z = small_data2d.shape
    _algorithm_setup._Algorithm2D(
        np.arange(num_x), np.arange(num_z)
    )._setup_spline(small_data2d, lam=[1])
    _algorithm_setup._Algorithm2D(
        np.arange(num_x), np.arange(num_z)
    )._setup_spline(small_data2d, lam=[1, 2])
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm2D(np.arange(num_x), np.arange(num_z))._setup_spline(
            small_data2d, lam=[1, 2, 3]
        )


@pytest.mark.parametrize('has_mask', (True, False))
@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_spline_weights(small_data2d, algorithm, weight_enum, has_mask):
    """Ensures output weight array is correct."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data2d)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d)
        desired_weights = np.ones_like(small_data2d)
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
        desired_weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape).tolist()
        desired_weights = np.arange(small_data2d.size).reshape(small_data2d.shape)

    if has_mask:
        mask = np.zeros(algorithm._shape, dtype=bool)
        mask[3:9] = True
        algorithm.mask = mask
        desired_weights = np.where(mask, 0., desired_weights)

    y, weight_array, _ = algorithm._setup_spline(
        small_data2d, lam=1, diff_order=2, weights=weights
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert_allclose(y, small_data2d, rtol=1e-14, atol=1e-14)
    assert weight_array.dtype == float


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('input_z', (True, False))
@pytest.mark.parametrize('check_finite', (True, False))
@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_class_init(input_x, input_z, check_finite, assume_sorted, change_order):
    """Tests the initialization of _Algorithm2D objects."""
    sort_order = slice(0, 10)
    expected_x = None
    expected_z = None
    x = None
    z = None
    if input_x or input_z:
        x_, z_, _ = get_data2d()
        if input_x:
            x = x_
        if input_z:
            z = z_

    if input_x:
        expected_x = x.copy()
        if change_order:
            x[sort_order] = x[sort_order][::-1]
            # sanity check that a true copy was made
            assert (expected_x != x).any()

    if input_z:
        expected_z = z.copy()
        if change_order:
            z[sort_order] = z[sort_order][::-1]
            # sanity check that a true copy was made
            assert (expected_z != z).any()

    if assume_sorted and change_order and (input_x or input_z):
        with pytest.warns(SortingWarning):
            algorithm = _algorithm_setup._Algorithm2D(
                x, z, check_finite=check_finite, assume_sorted=assume_sorted,
            )
    else:
        algorithm = _algorithm_setup._Algorithm2D(
            x, z, check_finite=check_finite, assume_sorted=assume_sorted
        )
    assert_array_equal(algorithm.x, expected_x)
    assert_array_equal(algorithm.z, expected_z)
    assert algorithm._check_finite == check_finite

    expected_shape = [None, None]
    if input_x:
        expected_shape[0] = len(x)
    if input_z:
        expected_shape[1] = len(z)
    assert isinstance(algorithm._shape, tuple)
    assert algorithm._shape == tuple(expected_shape)
    if None in expected_shape:
        assert algorithm._size is None
    else:
        assert algorithm._size == len(x) * len(z)

    if change_order and (input_x or input_z):
        if input_x and input_z:
            x_order = np.arange(len(x))
            z_order = np.arange(len(z))
            for order in (x_order, z_order):
                order[sort_order] = order[sort_order][::-1]

            for actual, expected in zip(
                algorithm._sort_order, (x_order[:, None], z_order[None, :])
            ):
                assert_array_equal(actual, expected)
            for actual, expected in zip(
                algorithm._inverted_order, (x_order.argsort()[:, None], z_order.argsort()[None, :])
            ):
                assert_array_equal(actual, expected)
        elif input_x:
            order = np.arange(len(x))
            order[sort_order] = order[sort_order][::-1]
            assert_array_equal(algorithm._sort_order, order)
            assert_array_equal(algorithm._inverted_order, order.argsort())
        else:
            order = np.arange(len(z))
            order[sort_order] = order[sort_order][::-1]
            assert_array_equal(algorithm._sort_order[1], order)
            assert_array_equal(algorithm._inverted_order[1], order.argsort())
            assert algorithm._sort_order[0] is Ellipsis
            assert algorithm._inverted_order[0] is Ellipsis
    else:
        assert algorithm._sort_order is None
        assert algorithm._inverted_order is None

    # ensure attributes are correctly initialized
    assert algorithm._polynomial is None
    assert algorithm._spline_basis is None
    if input_x:
        assert not algorithm._validated_x
    else:
        assert algorithm._validated_x
    if input_z:
        assert not algorithm._validated_z
    else:
        assert algorithm._validated_z


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('input_z', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_class_init_mask(input_x, input_z, change_order):
    """Tests the initialization of _Algorithm2D objects when given a mask."""
    x_, z_, y = get_data2d()
    mask = np.zeros(y.shape, dtype=bool)
    mask[:mask.shape[0] // 2, :mask.shape[1] // 2] = True
    expected_mask = mask.copy()
    if input_x:
        x = x_
        expected_x = x.copy()
        if change_order:
            x = x[::-1]
            expected_mask = expected_mask[::-1]
    else:
        x = None
        expected_x = np.linspace(-1, 1, y.shape[0])

    if input_z:
        z = z_
        expected_z = z.copy()
        if change_order:
            z = z[::-1]
            expected_mask = expected_mask[:, ::-1]
    else:
        z = None
        expected_z = np.linspace(-1, 1, y.shape[1])

    algorithm = _algorithm_setup._Algorithm2D(x, z, mask=mask)
    assert_allclose(algorithm.x, expected_x, rtol=1e-15, atol=1e-15)
    assert_allclose(algorithm.z, expected_z, rtol=1e-15, atol=1e-15)
    assert_array_equal(algorithm.mask, expected_mask)

    assert isinstance(algorithm._shape, tuple)
    assert algorithm._shape == y.shape
    assert algorithm._size == y.size

    if input_x:
        assert not algorithm._validated_x
    else:
        assert algorithm._validated_x
    if input_z:
        assert not algorithm._validated_z
    else:
        assert algorithm._validated_z


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('input_z', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_class_init_mask_attr(input_x, input_z, change_order):
    """Like test_algorithm_class_init_mask, but sets mask after initialization."""
    x_, z_, y = get_data2d()
    mask = np.zeros(y.shape, dtype=bool)
    mask[:mask.shape[0] // 2, :mask.shape[1] // 2] = True
    expected_mask = mask.copy()
    if input_x:
        x = x_
        expected_x = x.copy()
        if change_order:
            x = x[::-1]
            expected_mask = expected_mask[::-1]
    else:
        x = None
        expected_x = np.linspace(-1, 1, y.shape[0])

    if input_z:
        z = z_
        expected_z = z.copy()
        if change_order:
            z = z[::-1]
            expected_mask = expected_mask[:, ::-1]
    else:
        z = None
        expected_z = np.linspace(-1, 1, y.shape[1])

    algorithm = _algorithm_setup._Algorithm2D(x, z)
    assert algorithm.mask is None
    if input_x:
        assert_allclose(algorithm.x, expected_x, rtol=1e-15, atol=1e-15)
    else:
        assert algorithm.x is None
    if input_z:
        assert_allclose(algorithm.z, expected_z, rtol=1e-15, atol=1e-15)
    else:
        assert algorithm.z is None

    algorithm.mask = mask
    assert_allclose(algorithm.x, expected_x, rtol=1e-15, atol=1e-15)
    assert_allclose(algorithm.z, expected_z, rtol=1e-15, atol=1e-15)
    assert_array_equal(algorithm.mask, expected_mask)

    assert isinstance(algorithm._shape, tuple)
    assert algorithm._shape == y.shape
    assert algorithm._size == y.size

    if input_x:
        assert not algorithm._validated_x
    else:
        assert algorithm._validated_x
    if input_z:
        assert not algorithm._validated_z
    else:
        assert algorithm._validated_z


@pytest.mark.parametrize('shape_mismatch', ((1, 0), (0, 1), (1, 1)))
def test_algorithm_mask_incorrect_size(shape_mismatch):
    """Ensures an exception is raised if x, z, and mask sizes differ."""
    x = np.arange(10)
    z = np.arange(11)
    mask = np.zeros((x.size + shape_mismatch[0], z.size + shape_mismatch[1]))
    with pytest.raises(ValueError, match='length mismatch for mask'):
        _algorithm_setup._Algorithm2D(x, z, mask=mask)

    # also ensure it fails when setting the attribute
    algorithm = _algorithm_setup._Algorithm2D(x, z)
    with pytest.raises(ValueError, match='length mismatch for mask'):
        algorithm.mask = mask


@pytest.mark.parametrize('one_d', (True, False))
def test_algorithm_mask_non2d_fails(one_d):
    """Ensures an exception is raised if mask is not 2D."""
    x = np.arange(10)
    z = np.arange(11)
    mask = np.zeros((x.size, z.size), dtype=bool)
    if one_d:
        mask = mask.ravel()
    else:
        mask = np.repeat(mask[None, :], 2, axis=0)
    with pytest.raises(ValueError, match='input data must be a two dimensional array'):
        _algorithm_setup._Algorithm2D(x, z, mask=mask)

    # also ensure it fails when setting the attribute
    algorithm = _algorithm_setup._Algorithm2D(x, z)
    with pytest.raises(ValueError, match='input data must be a two dimensional array'):
        algorithm.mask = mask


@ensure_deprecation(1, 5)  # remove output_dtype from _Algorithm2D in v1.5
@pytest.mark.parametrize('output_dtype', ('deprecated', int, float, np.float64))
def test_algorithm_class_init_dtype(output_dtype):
    """Ensures specifying output_dtype gives DeprecationWarning."""
    x, z, _ = get_data2d()

    if output_dtype != 'deprecated':
        with pytest.warns(DeprecationWarning, match='specifying "output_dtype" is deprecated'):
            algorithm = _algorithm_setup._Algorithm2D(x, z, output_dtype=output_dtype)
    else:
        algorithm = _algorithm_setup._Algorithm2D(x, z, output_dtype=output_dtype)

    if output_dtype == 'deprecated':
        expected_attribute = 'deprecated'
    else:
        expected_attribute = output_dtype
    assert algorithm._dtype == expected_attribute


@ensure_deprecation(1, 5)  # remove dtype from _Algorithm2D._return_results in v1.5
@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('reshape_baseline', (True, False))
@pytest.mark.parametrize('three_d', (True, False))
def test_algorithm_return_results(assume_sorted, change_order, reshape_baseline, three_d):
    """Ensures the _return_results method returns the correctly sorted outputs."""
    x, z, y = get_data2d()
    baseline = np.arange(y.size).reshape(y.shape)
    # 'a' values will be sorted, 'b' values will be kept the same, 'c' will be reshaped,
    # and 'd' will be reshaped and then sorted
    params = {
        'a': np.arange(y.size).reshape(y.shape),
        'b': np.arange(len(x)),
        'c': np.arange(y.size),
        'd': np.arange(y.size),
    }
    if change_order:
        x = x[::-1]
        z = z[::-1]
        y = y[::-1, ::-1]

    expected_params = {
        'a': np.arange(y.size).reshape(y.shape),
        'b': np.arange(len(x)),
        'c': np.arange(y.size).reshape(y.shape),
        'd': np.arange(y.size).reshape(y.shape),
    }
    if three_d:
        baseline = np.array([baseline, baseline])
    expected_baseline = baseline.copy()
    if reshape_baseline:
        baseline = baseline.reshape(baseline.shape[0], -1)

    if change_order:
        expected_baseline = expected_baseline[..., ::-1, ::-1]
        expected_params['a'] = expected_params['a'][::-1, ::-1]
        expected_params['d'] = expected_params['d'][::-1, ::-1]

    if assume_sorted and change_order:
        with pytest.warns(SortingWarning):
            algorithm = _algorithm_setup._Algorithm2D(x, z, assume_sorted=assume_sorted)
    else:
        algorithm = _algorithm_setup._Algorithm2D(x, z, assume_sorted=assume_sorted)
    output, output_params = algorithm._return_results(
        baseline, params, dtype='deprecated', sort_keys=('a', 'd'),
        reshape_keys=('c', 'd'), ensure_dims=not three_d
    )

    assert_allclose(output, expected_baseline, 1e-14, 1e-14)
    for key, value in expected_params.items():
        assert_array_equal(value, output_params[key])


@ensure_deprecation(1, 5)  # remove dtype from _Algorithm2D._return_results in v1.5
@pytest.mark.parametrize('output_dtype', ('deprecated', int, float, np.float64))
def test_algorithm_return_results_dtype(output_dtype):
    """Ensures the _return_results method respects specified dtypes."""
    x, z, y = get_data2d()
    baseline = np.arange(y.size).reshape(y.shape)

    if output_dtype != 'deprecated':
        with pytest.warns(DeprecationWarning, match='specifying "output_dtype" is deprecated'):
            algorithm = _algorithm_setup._Algorithm2D(x, z, output_dtype=output_dtype)
    else:
        algorithm = _algorithm_setup._Algorithm2D(x, z, output_dtype=output_dtype)

    output, _ = algorithm._return_results(baseline, {}, dtype=output_dtype)

    if (output_dtype == 'deprecated' or output.dtype == baseline.dtype):
        assert np.shares_memory(output, baseline)  # should be the same object
    else:
        assert baseline is not output

    if output_dtype != 'deprecated':
        assert output.dtype == output_dtype
    else:
        assert output.dtype == baseline.dtype


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('skip_sorting', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
def test_algorithm_handle_io(assume_sorted, change_order, skip_sorting, list_input):
    """
    Ensures the _handle_io wrapper method returns the correctly sorted and shaped outputs.

    The input y-values within the wrapped function should be correctly sorted
    if `assume_sorted` is False, while the output baseline should always match
    the ordering of the input y-values. The output params should have an inverted
    sort order to also match the ordering of the input y-values if `assume_sorted`
    is False.

    """
    x, z, y = get_data2d()

    class SubClass(_algorithm_setup._Algorithm2D):
        # 'a' values will be sorted and 'b' values will be kept the same
        @_algorithm_setup._Algorithm2D._handle_io(sort_keys=('a', 'd'), reshape_keys=('c', 'd'))
        def func(self, data, *args, **kwargs):
            """For checking sorting and reshaping output parameters."""
            expected_x, expected_z, expected_y = get_data2d()

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            params = {
                'a': np.arange(data.size).reshape(data.shape),
                'b': np.arange(len(self.x)),
                'c': np.arange(data.size),
                'd': np.arange(data.size)
            }
            return 1 * data, params

        @_algorithm_setup._Algorithm2D._handle_io
        def func2(self, data, *args, **kwargs):
            """For checking reshaping output baseline."""
            expected_x, expected_z, expected_y = get_data2d()

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return 1 * data.ravel(), {}

        @_algorithm_setup._Algorithm2D._handle_io
        def func3(self, data, *args, **kwargs):
            """For checking empty decorator."""
            expected_x, expected_z, expected_y = get_data2d()

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._handle_io(
            sort_keys=('a', 'd'), reshape_keys=('c', 'd'), skip_sorting=skip_sorting
        )
        def func4(self, data, *args, **kwargs):
            """For checking skip_sorting key."""
            expected_x, expected_z, expected_y = get_data2d()
            if change_order and skip_sorting:
                expected_y = expected_y[::-1, ::-1]

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            params = {
                'a': np.arange(data.size).reshape(data.shape),
                'b': np.arange(len(self.x)),
                'c': np.arange(data.size),
                'd': np.arange(data.size)
            }

            return 1 * data, params

        @_algorithm_setup._Algorithm2D._handle_io(require_unique=False)
        def func5(self, data, *args, **kwargs):
            """For ensuring require_unique works as intended."""
            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._handle_io(require_unique=True)
        def func6(self, data, *args, **kwargs):
            """For ensuring require_unique works as intended."""
            return 1 * data, {}

    if change_order:
        x = x[::-1]
        z = z[::-1]
        y = y[::-1, ::-1]
    expected_params = {
        'a': np.arange(y.size).reshape(y.shape),
        'b': np.arange(len(x)),
        'c': np.arange(y.size).reshape(y.shape),
        'd': np.arange(y.size).reshape(y.shape),
    }
    expected_baseline = 1. * y
    expected_dtype = float
    if list_input:
        x = x.tolist()
        z = z.tolist()
        y = y.tolist()

    if change_order:
        expected_params['a'] = expected_params['a'][::-1, ::-1]
        expected_params['d'] = expected_params['d'][::-1, ::-1]

    if assume_sorted and change_order:
        with pytest.warns(SortingWarning):
            algorithm = SubClass(x, z, assume_sorted=assume_sorted)
    else:
        algorithm = SubClass(x, z, assume_sorted=assume_sorted)

    output, output_params = algorithm.func(y)

    # baseline should always match y-order on the output; only sorted within the
    # function
    assert_allclose(output, expected_baseline, 1e-14, 1e-14)
    assert isinstance(output, np.ndarray)
    assert output.dtype == expected_dtype
    for key, value in expected_params.items():
        assert_array_equal(value, output_params[key], err_msg=f'{key} failed')

    output2, _ = algorithm.func2(y)
    assert_allclose(output2, expected_baseline, 1e-14, 1e-14)
    assert isinstance(output2, np.ndarray)
    assert output2.dtype == expected_dtype

    output3, _ = algorithm.func3(y)
    assert_allclose(output3, expected_baseline, 1e-14, 1e-14)
    assert isinstance(output3, np.ndarray)
    assert output3.dtype == expected_dtype

    output4, output_params4 = algorithm.func4(y)
    assert_allclose(output4, expected_baseline, 1e-14, 1e-14)
    assert isinstance(output4, np.ndarray)
    assert output4.dtype == expected_dtype
    for key, value in expected_params.items():
        assert_array_equal(value, output_params4[key], err_msg=f'{key} failed')

    assert not algorithm._validated_x  # has not had a need to validate x or z yet
    assert not algorithm._validated_z
    output = algorithm.func6(y)
    assert algorithm._validated_x
    assert algorithm._validated_z

    x[5] = x[4]
    new_algorithm = SubClass(x)
    # ensure calling a method that does not require unique x does not validate or raise an error
    out = new_algorithm.func5(y)
    assert not new_algorithm._validated_x
    assert new_algorithm._validated_z  # not given z
    with pytest.raises(ValueError):
        out = new_algorithm.func6(y)

    z[5] = z[4]
    new_algorithm = SubClass(z_data=z)
    # ensure calling a method that does not require unique z does not validate or raise an error
    out = new_algorithm.func5(y)
    assert new_algorithm._validated_x  # not given x
    assert not new_algorithm._validated_z
    with pytest.raises(ValueError):
        out = new_algorithm.func6(y)

    new_algorithm = SubClass(x, z)
    out = new_algorithm.func5(y)
    assert not new_algorithm._validated_x
    assert not new_algorithm._validated_z
    with pytest.raises(ValueError):
        out = new_algorithm.func6(y)


def test_algorithm_handle_io_no_data_fails():
    """Ensures an error is raised if the input data is None."""

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._handle_io
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._Algorithm2D._handle_io()
        def func2(self, data, *args, **kwargs):
            """For checking closed decorator."""
            return data, {}

    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func()
    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func2()


def test_algorithm_handle_io_1d_fails(data_fixture):
    """Ensures an error is raised if 1D data is used for 2D algorithms."""

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._handle_io
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._Algorithm2D._handle_io()
        def func2(self, data, *args, **kwargs):
            """For checking closed decorator."""
            return data, {}

    x, y = data_fixture
    algorithm = SubClass()
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y)

    # also test when given x values
    algorithm = SubClass(None, x)  # x would correspond to the columns in 2D y
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y)

    # and when y is 2D but only has one row
    y_2d = np.atleast_2d(y)
    algorithm = SubClass()
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y_2d)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y_2d)

    algorithm = SubClass(None, x)  # x would correspond to the columns in 2D y
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y_2d)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y_2d)

    # and when y is 2D but only has one column
    y_2d_transposed = np.atleast_2d(y).T
    algorithm = SubClass()
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y_2d_transposed)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y_2d_transposed)

    algorithm = SubClass(x)  # x now correspond to the rows in 2D y
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(y_2d_transposed)
    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func2(y_2d_transposed)


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('input_z', (True, False))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_handle_io_3d(input_x, input_z, change_order):
    """Ensures 3D data is allowed for 2D algorithms only when specified.

    Also checks _Algorithm2D setup when given 3D data as the first call.

    """
    x_vals, z_vals, input_y_2d = get_data2d()
    x_slice = slice(None)
    z_slice = slice(None)
    if input_x:
        expected_x = x_vals
    else:
        expected_x = np.linspace(-1, 1, input_y_2d.shape[0])
        if change_order:
            x_slice = slice(None, None, -1)
    if input_z:
        expected_z = z_vals
    else:
        expected_z = np.linspace(-1, 1, input_y_2d.shape[1])
        if change_order:
            z_slice = slice(None, None, -1)
    stacks = 2
    expected_y = np.repeat(input_y_2d[None, :], stacks, axis=0)

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._handle_io
        def func(self, data, *args, **kwargs):
            """Errors if input is not 2D."""
            assert data.ndim == 2
            assert data.shape == expected_y.shape
            return data, {}

        @_algorithm_setup._Algorithm2D._handle_io(ensure_dims=False)
        def func2(self, data, *args, **kwargs):
            """Allows 3D data."""
            assert data.ndim == 3
            assert data.shape == expected_y.shape

            expected = expected_y.copy()
            if change_order:
                expected = expected[:, x_slice, z_slice]

            assert_allclose(data, expected, 1e-14, 1e-14)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return data * 1, {}

        @_algorithm_setup._Algorithm2D._handle_io(ensure_dims=False)
        def func3(self, data, *args, **kwargs):
            """For checking reshaping output baseline for 3D input raveled on last axis."""
            assert data.ndim == 3
            assert data.shape == expected_y.shape

            return 1 * data.reshape(data.shape[0], -1), {}

        @_algorithm_setup._Algorithm2D._handle_io(ensure_dims=False, skip_sorting=True)
        def func4(self, data, *args, **kwargs):
            """Allows 3D data and skips sorting."""
            assert data.ndim == 3
            assert data.shape == expected_y.shape

            expected = expected_y.copy()
            if change_order:
                expected = expected[:, ::-1, ::-1]

            assert_allclose(data, expected, 1e-14, 1e-14)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return data * 1, {}

    x_, z_, y_2d = get_data2d()
    if change_order:
        x_ = x_[::-1]
        z_ = z_[::-1]
        y_2d = y_2d[::-1, ::-1]
    x = None
    z = None
    initial_shape = [None, None]
    if input_x:
        x = x_
        initial_shape[0] = len(x)
    if input_z:
        z = z_
        initial_shape[1] = len(z)
    initial_shape = tuple(initial_shape)
    initial_size = None if None in initial_shape else y_2d.size

    input_y = np.repeat(y_2d[None, :], stacks, axis=0)
    assert input_y.shape == (stacks, *y_2d.shape)  # sanity check for correct setup

    algorithm = SubClass(x, z)
    assert algorithm._shape == initial_shape
    assert algorithm._size == initial_size

    with pytest.raises(ValueError, match='input data must be a two dimensional'):
        algorithm.func(input_y)
    assert algorithm._shape == initial_shape

    # should run without issues and set stored shape correctly
    output, _ = algorithm.func2(input_y)
    assert algorithm._shape == y_2d.shape
    assert algorithm._size == y_2d.size
    assert output.shape == input_y.shape
    assert_allclose(output, input_y, 1e-14, 1e-14)

    output2, _ = algorithm.func3(input_y)
    assert output2.shape == input_y.shape
    assert_allclose(output2, input_y, 1e-14, 1e-14)

    output3, _ = algorithm.func4(input_y)
    assert output3.shape == input_y.shape
    assert_allclose(output2, input_y, 1e-14, 1e-14)


@pytest.mark.parametrize('three_d', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
@pytest.mark.parametrize('check_finite', (True, False))
@pytest.mark.parametrize('strict_mask', (True, False))
def test_algorithm_handle_io_mask(list_input, check_finite, strict_mask, three_d):
    """Ensures the _handle_io wrapper method works correctly with masks."""
    x, z, y = get_data2d(num_points=(30, 41))
    mask = np.zeros(y.shape, dtype=bool)
    mask[[1, 5], [1, 7]] = True
    mask_inv = np.logical_not(mask)

    expected_zero_fill = np.where(mask, 0., y)
    X, Z = np.meshgrid(x, z, indexing='ij')
    expected_interp = LinearNDInterpolator(
        np.stack((X[mask_inv], Z[mask_inv]), axis=-1), y[mask_inv], fill_value=0
    )(np.stack((X, Z), axis=-1)).reshape(y.shape)
    y[mask] = np.nan
    if three_d:
        stacks = 3
        y = np.repeat(y[None, :], stacks, axis=0)
        expected_zero_fill = np.repeat(expected_zero_fill[None, :], stacks, axis=0)
        expected_interp = np.repeat(expected_interp[None, :], stacks, axis=0)

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._handle_io(ensure_dims=not three_d)
        def func(self, data, *args, **kwargs):
            """Masking not supported; strict_mask=True will interpolate."""
            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_interp, 1e-16, 1e-16)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, x, 1e-16, 1e-16)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, z, 1e-16, 1e-16)

            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._handle_io(mask_support=0, ensure_dims=not three_d)
        def func2(self, data, *args, **kwargs):
            """Ignores mask."""
            assert isinstance(data, np.ndarray)
            assert_allclose(data, y, 1e-16, 1e-16)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, x, 1e-16, 1e-16)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, z, 1e-16, 1e-16)

            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._handle_io(mask_support=1, ensure_dims=not three_d)
        def func3(self, data, *args, **kwargs):
            """Replaces masked values with 0."""
            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_zero_fill, 1e-16, 1e-16)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, x, 1e-16, 1e-16)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, z, 1e-16, 1e-16)

            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._handle_io(mask_support=2, ensure_dims=not three_d)
        def func4(self, data, *args, **kwargs):
            """Interpolates following the mask."""
            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_interp, 1e-16, 1e-16)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, x, 1e-16, 1e-16)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, z, 1e-16, 1e-16)

            return 1 * data, {}

    if list_input:
        x = x.tolist()
        z = z.tolist()
        y = y.tolist()

    algorithm = SubClass(x, z, mask=mask, strict_mask=strict_mask, check_finite=check_finite)

    if strict_mask:
        with pytest.raises(NotImplementedError, match='masking is not supported'):
            algorithm.func(y)
    else:
        output, _ = algorithm.func(y)
        assert_allclose(output, expected_interp, 1e-16, 1e-16)

    output2, _ = algorithm.func2(y)
    assert_allclose(output2, y, 1e-16, 1e-16)

    output3, _ = algorithm.func3(y)
    assert_allclose(output3, expected_zero_fill, 1e-16, 1e-16)

    output4, _ = algorithm.func4(y)
    assert_allclose(output4, expected_interp, 1e-16, 1e-16)


@pytest.mark.parametrize('three_d', (True, False))
@pytest.mark.parametrize('check_finite', (True, False))
def test_mask_check_finite(check_finite, three_d):
    """Ensures proper behavior with both a mask and check_finite.

    If a mask is supplied, only non-masked regions should be sugjected to the finite check.

    """
    x, z, y = get_data2d(num_points=(20, 21))
    mask = np.zeros(y.shape, dtype=bool)
    mask[[1, 11], [0, 5]] = True

    y[mask] = np.nan
    if three_d:
        y = np.repeat(y[None, :], 3, axis=0)
    y2 = y.copy()
    y2[..., 5] = np.nan

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._handle_io(mask_support=0, ensure_dims=not three_d)
        def func(self, data, *args, **kwargs):
            return 1 * data, {}

    algorithm = SubClass(x, z, check_finite=check_finite, mask=mask)

    # check on y should always pass since it contains invalid points only in masked regions
    algorithm.func(y)
    if check_finite:
        with pytest.raises(ValueError, match='array must not contain infs or NaNs'):
            algorithm.func(y2)
    else:
        algorithm.func(y2)


def test_override_x(algorithm):
    """Ensures the `override_x` method correctly initializes with the new x values."""
    new_len = 20
    new_x = np.arange(new_len)
    with pytest.raises(NotImplementedError):
        new_algorithm = algorithm._override_x(new_x)


@pytest.mark.parametrize('method', ('collab_pls', 'modpoly', 'asls'))
@pytest.mark.parametrize('ensure_new', (True, False))
def test_spawn_fitter(method, ensure_new):
    """Ensures _spawn_fitter gets the correct method and creates new object when appropriate."""
    algorithm = Baseline2D(
        x_data=np.arange(10), z_data=np.arange(20), assume_sorted=True, check_finite=False
    )
    class_object = algorithm._spawn_fitter(method, ensure_new=ensure_new)
    assert isinstance(class_object, _algorithm_setup._Algorithm2D)
    if ensure_new:
        assert class_object is not algorithm
    else:
        assert class_object is algorithm


def test_spawn_fitter_fails_wrong_method(algorithm):
    """Ensures _get_function fails when an no function with the input name is available."""
    with pytest.raises(AttributeError):
        algorithm._spawn_fitter('unknown function')


def test_get_function_fails_no_module(algorithm):
    """Ensures _get_function fails when not given any modules to search."""
    with pytest.raises(AttributeError):
        algorithm._get_function('collab_pls', [])


@pytest.mark.parametrize('ensure_new', (True, False))
@pytest.mark.parametrize('has_mask', (True, False))
def test_spawn_fitter_sorting_x(ensure_new, has_mask):
    """Ensures the sort order is correct for the output class object when x is reversed."""
    num_points = 10
    x = np.arange(num_points)
    ordering = np.arange(num_points)
    if has_mask:
        mask = np.zeros((num_points, num_points), dtype=bool)
        mask[:num_points // 2, :num_points // 2] = True
    else:
        mask = None
    algorithm = Baseline2D(x[::-1], assume_sorted=False, mask=mask)
    class_object = algorithm._spawn_fitter('asls', ensure_new=ensure_new)

    assert_array_equal(class_object.x, x)
    assert_array_equal(class_object._sort_order, ordering[::-1])
    assert_array_equal(class_object._inverted_order, ordering[::-1])
    assert_array_equal(class_object._sort_order, algorithm._sort_order)
    assert_array_equal(class_object._inverted_order, algorithm._inverted_order)
    if ensure_new:
        assert class_object is not algorithm
    else:
        assert class_object is algorithm
    if has_mask:
        assert_array_equal(class_object.mask, mask[::-1])
    else:
        assert class_object.mask is None


@pytest.mark.parametrize('ensure_new', (True, False))
@pytest.mark.parametrize('has_mask', (True, False))
def test_spawn_fitter_sorting_z(ensure_new, has_mask):
    """Ensures the sort order is correct for the output class object when z is reversed."""
    num_points = 10
    z = np.arange(num_points)
    ordering = np.arange(num_points)
    if has_mask:
        mask = np.zeros((num_points, num_points), dtype=bool)
        mask[:num_points // 2, :num_points // 2] = True
    else:
        mask = None
    algorithm = Baseline2D(None, z[::-1], assume_sorted=False, mask=mask)
    class_object = algorithm._spawn_fitter('asls', ensure_new=ensure_new)

    assert_array_equal(class_object.z, z)
    assert class_object._sort_order[0] is Ellipsis
    assert class_object._inverted_order[0] is Ellipsis
    assert algorithm._sort_order[0] is Ellipsis
    assert algorithm._inverted_order[0] is Ellipsis
    assert_array_equal(class_object._sort_order[1], ordering[::-1])
    assert_array_equal(class_object._inverted_order[1], ordering[::-1])
    assert_array_equal(class_object._sort_order[1], algorithm._sort_order[1])
    assert_array_equal(class_object._inverted_order[1], algorithm._inverted_order[1])
    if ensure_new:
        assert class_object is not algorithm
    else:
        assert class_object is algorithm
    if has_mask:
        assert_array_equal(class_object.mask, mask[:, ::-1])
    else:
        assert class_object.mask is None


@pytest.mark.parametrize('ensure_new', (True, False))
@pytest.mark.parametrize('has_mask', (True, False))
def test_spawn_fitter_sorting_xz(ensure_new, has_mask):
    """Ensures the sort order is correct for the output class object when x and z are reversed."""
    num_x_points = 10
    num_z_points = 11
    x = np.arange(num_x_points)
    x_ordering = np.arange(num_x_points)
    z = np.arange(num_z_points)
    z_ordering = np.arange(num_z_points)
    if has_mask:
        mask = np.zeros((num_x_points, num_z_points), dtype=bool)
        mask[:num_x_points // 2, :num_z_points // 2] = True
    else:
        mask = None

    algorithm = Baseline2D(x[::-1], z[::-1], assume_sorted=False, mask=mask)
    class_object = algorithm._spawn_fitter('asls', ensure_new=ensure_new)

    assert_array_equal(class_object.x, x)
    assert_array_equal(class_object.z, z)
    assert_array_equal(class_object._sort_order[0], x_ordering[::-1][:, None])
    assert_array_equal(class_object._sort_order[1], z_ordering[::-1][None, :])
    assert_array_equal(class_object._inverted_order[0], x_ordering[::-1][:, None])
    assert_array_equal(class_object._inverted_order[1], z_ordering[::-1][None, :])
    assert_array_equal(class_object._sort_order[0], algorithm._sort_order[0])
    assert_array_equal(class_object._sort_order[1], algorithm._sort_order[1])
    assert_array_equal(class_object._inverted_order[0], algorithm._inverted_order[0])
    assert_array_equal(class_object._inverted_order[1], algorithm._inverted_order[1])
    if ensure_new:
        assert class_object is not algorithm
    else:
        assert class_object is algorithm
    if has_mask:
        assert_array_equal(class_object.mask, mask[::-1, ::-1])
    else:
        assert class_object.mask is None


@pytest.mark.parametrize('method_kwargs', (None, {'a': 2}))
@pytest.mark.parametrize('ensure_new', (True, False))
def test_setup_optimizer(small_data2d, method_kwargs, ensure_new):
    """Ensures output of _setup_optimizer is correct."""
    num_x, num_z = small_data2d.shape
    algorithm = Baseline2D(
        x_data=np.arange(num_x), z_data=np.arange(num_z), assume_sorted=True, check_finite=False
    )
    y, optimizer_obj, output_kwargs = algorithm._setup_optimizer(
        small_data2d, 'asls', method_kwargs=method_kwargs, ensure_new=ensure_new
    )

    assert isinstance(y, np.ndarray)
    assert_allclose(y, small_data2d)
    assert callable(optimizer_obj.method_call)
    assert optimizer_obj.method_call.__name__ == 'asls'
    assert optimizer_obj.module == 'whittaker'
    assert isinstance(output_kwargs, dict)
    if method_kwargs is not None:
        assert output_kwargs == method_kwargs
    else:
        assert output_kwargs == {}
    assert isinstance(optimizer_obj.fitter, _algorithm_setup._Algorithm2D)
    if ensure_new:
        assert optimizer_obj.fitter is not algorithm
    else:
        assert optimizer_obj.fitter is algorithm


@pytest.mark.parametrize('copy_kwargs', (True, False))
def test_setup_optimizer_copy_kwargs(small_data2d, algorithm, copy_kwargs):
    """Ensures the copy behavior of the input keyword argument dictionary."""
    input_kwargs = {'a': 1}
    _, _, output_kwargs = algorithm._setup_optimizer(
        small_data2d, 'asls', method_kwargs=input_kwargs, copy_kwargs=copy_kwargs
    )

    output_kwargs['a'] = 2
    if copy_kwargs:
        assert input_kwargs['a'] == 1
    else:
        assert input_kwargs['a'] == 2


@ensure_deprecation(1, 4)
def test_deprecated_pentapy_solver(algorithm):
    """Ensures setting and getting the pentapy_solver attribute is deprecated."""
    with pytest.warns(DeprecationWarning):
        algorithm.pentapy_solver = 2
    with pytest.warns(DeprecationWarning):
        solver = algorithm.pentapy_solver


@pytest.mark.parametrize('banded_solver', (1, 2, 3, 4))
def test_banded_solver(algorithm, banded_solver):
    """Ensures setting banded_solver works as intended."""
    algorithm.banded_solver = banded_solver
    assert algorithm.banded_solver == banded_solver


@pytest.mark.parametrize('banded_solver', (0, -1, 5, '1', True, False))
def test_wrong_banded_solver_fails(algorithm, banded_solver):
    """Ensures only valid integers between 0 and 4 are allowed as banded_solver inputs."""
    with pytest.raises(ValueError):
        algorithm.banded_solver = banded_solver


@pytest.mark.parametrize('diff_order', (1, 2, 3, (2, 3)))
@pytest.mark.parametrize('lam', (1, 20, (2, 5)))
def test_setup_pls_whittaker_diff_matrix(data_fixture2d, lam, diff_order):
    """Ensures output difference matrix diagonal data is in desired format for _setup_pls."""
    x, z, y = data_fixture2d

    algorithm = _algorithm_setup._Algorithm2D(x, z)

    # intentionally do not input spline_degree here to ensure default behavior is
    # spline_degree=None -> Whittaker smoothing
    _, _, whittaker_system, result_class = algorithm._setup_pls(y, lam=lam, diff_order=diff_order)
    _, _, expected_system = algorithm._setup_whittaker(y, lam=lam, diff_order=diff_order)

    *_, lam_x, lam_z, diff_order_x, diff_order_z = get_2dspline_inputs(
        lam=lam, diff_order=diff_order
    )

    D1 = difference_matrix(len(x), diff_order_x)
    D2 = difference_matrix(len(z), diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(len(z)))
    P2 = lam_z * kron(identity(len(x)), D2.T @ D2)
    expected_penalty = P1 + P2

    assert_allclose(
        whittaker_system.penalty.toarray(),
        expected_penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )
    assert_allclose(
        whittaker_system.penalty.toarray(),
        expected_system.penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )
    assert isinstance(whittaker_system, _whittaker_utils.WhittakerSystem2D)
    assert result_class is WhittakerResult2D


@pytest.mark.parametrize('has_mask', (True, False))
@pytest.mark.parametrize('spline_degree', (None, 3))
@pytest.mark.parametrize('num_eigens', (None, 3))
@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_pls_weights(small_data2d, algorithm, spline_degree, num_eigens, weight_enum,
                           has_mask):
    """Ensures output weight array is correct when using _setup_pls."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data2d)
        desired_weights = np.ones(small_data2d.size)
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape)
        desired_weights = np.arange(small_data2d.size)
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data2d.size).reshape(small_data2d.shape).tolist()
        desired_weights = np.arange(small_data2d.size)

    if has_mask:
        mask = np.zeros(algorithm._shape, dtype=bool)
        mask[3:9] = True
        algorithm.mask = mask
        desired_weights = np.where(mask.ravel(), 0., desired_weights)

    if spline_degree is None and num_eigens is None:
        expected_y = small_data2d.ravel()
    else:
        desired_weights = desired_weights.reshape(small_data2d.shape)
        expected_y = small_data2d

    y, weight_array, penalized_system, result_class = algorithm._setup_pls(
        small_data2d, lam=1, diff_order=2, weights=weights, spline_degree=spline_degree,
        num_eigens=num_eigens
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)
    assert weight_array.dtype == float
    assert_allclose(y, expected_y, rtol=1e-14, atol=1e-14)
    assert isinstance(
        penalized_system,
        _whittaker_utils.WhittakerSystem2D if spline_degree is None else _spline_utils.PSpline2D
    )
    assert result_class is WhittakerResult2D if spline_degree is None else PSplineResult2D


@pytest.mark.parametrize('num_knots', (10, 30, (20, 30)))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4, (2, 3), None))
def test_setup_pls_spline_basis(data_fixture2d, num_knots, spline_degree):
    """Ensures the spline basis function is correctly created through _setup_pls."""
    x, z, y = data_fixture2d
    fitter = _algorithm_setup._Algorithm2D(x, z)
    assert fitter._spline_basis is None

    fitter._setup_pls(
        y, weights=None, spline_degree=spline_degree, num_knots=num_knots
    )

    if spline_degree is None:
        assert fitter._spline_basis is None
        return

    if isinstance(num_knots, int):
        num_knots_r = num_knots
        num_knots_c = num_knots
    else:
        num_knots_r, num_knots_c = num_knots
    if isinstance(spline_degree, int):
        spline_degree_x = spline_degree
        spline_degree_z = spline_degree
    else:
        spline_degree_x, spline_degree_z = spline_degree

    assert_array_equal(
        fitter._spline_basis.basis_r.shape,
        (len(x), num_knots_r + spline_degree_x - 1)
    )
    assert_array_equal(
        fitter._spline_basis.basis_c.shape,
        (len(z), num_knots_c + spline_degree_z - 1)
    )


@pytest.mark.parametrize('lam', (1, 20, (3, 10)))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4, (2, 3)))
@pytest.mark.parametrize('num_knots', (20, (21, 30)))
def test_setup_pls_spline_diff_matrix(data_fixture2d, lam, diff_order, spline_degree, num_knots):
    """Ensures output difference matrix diagonal data is in desired format for setup_pls."""
    x, z, y = data_fixture2d

    algorithm = _algorithm_setup._Algorithm2D(x, z)
    _, _, pspline, result_class = algorithm._setup_pls(
        y, weights=None, spline_degree=spline_degree, num_knots=num_knots,
        diff_order=diff_order, lam=lam
    )

    (
        num_knots_r, num_knots_c, spline_degree_x, spline_degree_z,
        lam_x, lam_z, diff_order_x, diff_order_z
    ) = get_2dspline_inputs(
        num_knots=num_knots, spline_degree=spline_degree, lam=lam, diff_order=diff_order
    )

    num_bases_x = num_knots_r + spline_degree_x - 1
    num_bases_z = num_knots_c + spline_degree_z - 1

    D1 = difference_matrix(num_bases_x, diff_order_x)
    D2 = difference_matrix(num_bases_z, diff_order_z)

    P1 = lam_x * kron(D1.T @ D1, identity(num_bases_z))
    P2 = lam_z * kron(identity(num_bases_x), D2.T @ D2)
    expected_penalty = P1 + P2

    assert_allclose(
        pspline.penalty.toarray(),
        expected_penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )
    assert isinstance(pspline, _spline_utils.PSpline2D)
    assert result_class is PSplineResult2D

    _, _, expected_system = algorithm._setup_spline(
        y, weights=None, spline_degree=spline_degree, num_knots=num_knots,
        diff_order=diff_order, lam=lam
    )
    assert_allclose(
        pspline.penalty.toarray(),
        expected_system.penalty.toarray(),
        rtol=1e-12, atol=1e-12
    )
