# -*- coding: utf-8 -*-
"""Tests for pybaselines.two_d._algorithm_setup.

@author: Donald Erb
Created on January 5, 2024

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import kron

from pybaselines._compat import identity
from pybaselines.two_d import _algorithm_setup, optimizers, polynomial, whittaker
from pybaselines.utils import ParameterWarning, difference_matrix, optimize_window
from pybaselines._validation import _check_scalar

from ..conftest import ensure_deprecation, get_2dspline_inputs, get_data2d


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


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_whittaker_weights(small_data2d, algorithm, weight_enum):
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

    _, weight_array, _ = algorithm._setup_whittaker(
        small_data2d, lam=1, diff_order=2, weights=weights
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


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


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_polynomial_weights(small_data2d, algorithm, weight_enum):
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

    _, weight_array = algorithm._setup_polynomial(small_data2d, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


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


def test_setup_polynomial_maxcross(small_data2d, algorithm):
    """Ensures the _max_cross attribute is updated after calling _setup_polynomial."""
    algorithm._setup_polynomial(small_data2d, max_cross=[1], calc_vander=True)
    assert algorithm._polynomial.max_cross == 1

    algorithm._setup_polynomial(small_data2d, max_cross=1, calc_vander=True)
    assert algorithm._polynomial.max_cross == 1

    algorithm._setup_polynomial(small_data2d, max_cross=0)
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
    of optimize_window; setup_morphology should just pass the half window back
    out if it was not None.
    """
    x, z, y = data_fixture2d
    y_out, half_window_out = algorithm._setup_morphology(y, half_window)
    if half_window is None:
        half_window_expected = optimize_window(y)
    else:
        half_window_expected = _check_scalar(half_window, 2, fill_scalar=True, dtype=int)[0]
        # sanity check that the calculated half window does not match the test case one
        assert not np.array_equal(half_window, optimize_window(y))

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

    # also ensure both window_kwargs and **kwargs are passed to optimize_window
    with pytest.raises(TypeError):
        with pytest.warns(DeprecationWarning):
            algorithm._setup_morphology(
                small_data2d, window_kwargs={'min_half_window': 2}, min_half_window=2
            )


def test_setup_smooth_shape(small_data2d, algorithm):
    """Ensures output y is correctly padded."""
    pad_length = 4
    y, hw = algorithm._setup_smooth(small_data2d, pad_length, mode='edge')
    assert_array_equal(
        y.shape, (small_data2d.shape[0] + 2 * pad_length, small_data2d.shape[1] + 2 * pad_length)
    )
    assert_array_equal(hw, [pad_length, pad_length])


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
@pytest.mark.parametrize('num_knots', (20, 51, (20, 30)))
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


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_spline_weights(small_data2d, algorithm, weight_enum):
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

    _, weight_array, _ = algorithm._setup_spline(
        small_data2d, lam=1, diff_order=2, weights=weights
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('input_z', (True, False))
@pytest.mark.parametrize('check_finite', (True, False))
@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_class_init(input_x, input_z, check_finite, assume_sorted, output_dtype,
                              change_order):
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
            if assume_sorted:
                expected_x[sort_order] = expected_x[sort_order][::-1]
    if input_z:
        expected_z = z.copy()
        if change_order:
            z[sort_order] = z[sort_order][::-1]
            if assume_sorted:
                expected_z[sort_order] = expected_z[sort_order][::-1]

    algorithm = _algorithm_setup._Algorithm2D(
        x, z, check_finite=check_finite, assume_sorted=assume_sorted, output_dtype=output_dtype
    )
    assert_array_equal(algorithm.x, expected_x)
    assert_array_equal(algorithm.z, expected_z)
    assert algorithm._check_finite == check_finite
    assert algorithm._dtype == output_dtype

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

    if not assume_sorted and change_order and (input_x or input_z):
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


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('reshape_baseline', (True, False))
@pytest.mark.parametrize('three_d', (True, False))
def test_algorithm_return_results(assume_sorted, output_dtype, change_order, reshape_baseline,
                                  three_d):
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

    should_change_order = change_order and not assume_sorted
    if should_change_order:
        expected_baseline = expected_baseline[..., ::-1, ::-1]
        expected_params['a'] = expected_params['a'][::-1, ::-1]
        expected_params['d'] = expected_params['d'][::-1, ::-1]

    algorithm = _algorithm_setup._Algorithm2D(
        x, z, assume_sorted=assume_sorted, output_dtype=output_dtype, check_finite=False
    )
    output, output_params = algorithm._return_results(
        baseline, params, dtype=output_dtype, sort_keys=('a', 'd'),
        reshape_baseline=reshape_baseline, reshape_keys=('c', 'd'),
        ensure_2d=not three_d
    )

    if not should_change_order and (output_dtype is None or baseline.dtype == output_dtype):
        assert np.shares_memory(output, baseline)  # should be the same object
    else:
        assert baseline is not output

    if output_dtype is not None:
        assert output.dtype == output_dtype
    else:
        assert output.dtype == baseline.dtype

    assert_allclose(output, expected_baseline, 1e-14, 1e-14)
    for key, value in expected_params.items():
        assert_array_equal(value, output_params[key])


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('skip_sorting', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
def test_algorithm_register(assume_sorted, output_dtype, change_order, skip_sorting, list_input):
    """
    Ensures the _register wrapper method returns the correctly sorted and shaped outputs.

    The input y-values within the wrapped function should be correctly sorted
    if `assume_sorted` is False, while the output baseline should always match
    the ordering of the input y-values. The output params should have an inverted
    sort order to also match the ordering of the input y-values if `assume_sorted`
    is False.

    """
    x, z, y = get_data2d()

    class SubClass(_algorithm_setup._Algorithm2D):
        # 'a' values will be sorted and 'b' values will be kept the same
        @_algorithm_setup._Algorithm2D._register(sort_keys=('a', 'd'), reshape_keys=('c', 'd'))
        def func(self, data, *args, **kwargs):
            """For checking sorting and reshaping output parameters."""
            expected_x, expected_z, expected_y = get_data2d()
            if change_order and assume_sorted:
                expected_y = expected_y[::-1, ::-1]
                expected_x = expected_x[::-1]
                expected_z = expected_z[::-1]

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

        @_algorithm_setup._Algorithm2D._register(reshape_baseline=True)
        def func2(self, data, *args, **kwargs):
            """For checking reshaping output baseline."""
            expected_x, expected_z, expected_y = get_data2d()
            if change_order and assume_sorted:
                expected_y = expected_y[::-1, ::-1]
                expected_x = expected_x[::-1]
                expected_z = expected_z[::-1]

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return 1 * data.flatten(), {}

        @_algorithm_setup._Algorithm2D._register
        def func3(self, data, *args, **kwargs):
            """For checking empty decorator."""
            expected_x, expected_z, expected_y = get_data2d()
            if change_order and assume_sorted:
                expected_y = expected_y[::-1, ::-1]
                expected_x = expected_x[::-1]
                expected_z = expected_z[::-1]

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return 1 * data, {}

        @_algorithm_setup._Algorithm2D._register(
            sort_keys=('a', 'd'), reshape_keys=('c', 'd'), skip_sorting=skip_sorting
        )
        def func4(self, data, *args, **kwargs):
            """For checking skip_sorting key."""
            expected_x, expected_z, expected_y = get_data2d()
            if change_order and (assume_sorted or skip_sorting):
                expected_y = expected_y[::-1, ::-1]
            if change_order and assume_sorted:
                expected_x = expected_x[::-1]
                expected_z = expected_z[::-1]

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
    expected_baseline = (1 * y).astype(output_dtype)
    if output_dtype is None:
        expected_dtype = y.dtype
    else:
        expected_dtype = expected_baseline.dtype
    if list_input:
        x = x.tolist()
        z = z.tolist()
        y = y.tolist()

    if change_order and not assume_sorted:
        # if assume_sorted is False, the param order should be inverted to match
        # the input y-order
        expected_params['a'] = expected_params['a'][::-1, ::-1]
        expected_params['d'] = expected_params['d'][::-1, ::-1]

    algorithm = SubClass(
        x, z, assume_sorted=assume_sorted, output_dtype=output_dtype, check_finite=False
    )
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


def test_algorithm_register_no_data_fails():
    """Ensures an error is raised if the input data is None."""

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._register
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._Algorithm2D._register()
        def func2(self, data, *args, **kwargs):
            """For checking closed decorator."""
            return data, {}

    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func()
    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func2()


def test_algorithm_register_1d_fails(data_fixture):
    """Ensures an error is raised if 1D data is used for 2D algorithms."""

    class SubClass(_algorithm_setup._Algorithm2D):

        @_algorithm_setup._Algorithm2D._register
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._Algorithm2D._register()
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


def test_override_x(algorithm):
    """Ensures the `override_x` method correctly initializes with the new x values."""
    new_len = 20
    new_x = np.arange(new_len)
    with pytest.raises(NotImplementedError):
        new_algorithm = algorithm._override_x(new_x)


@pytest.mark.parametrize(
    'method_and_outputs', (
        ('collab_pls', 'collab_pls', 'optimizers'),
        ('COLLAB_pls', 'collab_pls', 'optimizers'),
        ('modpoly', 'modpoly', 'polynomial'),
        ('asls', 'asls', 'whittaker')
    )
)
def test_get_function(algorithm, method_and_outputs):
    """Ensures _get_function gets the correct method, regardless of case."""
    method, expected_func, expected_module = method_and_outputs
    tested_modules = [optimizers, polynomial, whittaker]
    selected_func, module, class_object = algorithm._get_function(
        method, tested_modules
    )
    assert selected_func.__name__ == expected_func
    assert module == expected_module
    assert isinstance(class_object, _algorithm_setup._Algorithm2D)


def test_get_function_fails_wrong_method(algorithm):
    """Ensures _get_function fails when an no function with the input name is available."""
    with pytest.raises(AttributeError):
        algorithm._get_function('unknown function', [optimizers])


def test_get_function_fails_no_module(algorithm):
    """Ensures _get_function fails when not given any modules to search."""
    with pytest.raises(AttributeError):
        algorithm._get_function('collab_pls', [])


def test_get_function_sorting_x():
    """Ensures the sort order is correct for the output class object when x is reversed."""
    num_points = 10
    x = np.arange(num_points)
    ordering = np.arange(num_points)
    algorithm = _algorithm_setup._Algorithm2D(x[::-1], assume_sorted=False)
    func, func_module, class_object = algorithm._get_function('asls', [whittaker])

    assert_array_equal(class_object.x, x)
    assert_array_equal(class_object._sort_order, ordering[::-1])
    assert_array_equal(class_object._inverted_order, ordering[::-1])
    assert_array_equal(class_object._sort_order, algorithm._sort_order)
    assert_array_equal(class_object._inverted_order, algorithm._inverted_order)


def test_get_function_sorting_z():
    """Ensures the sort order is correct for the output class object when z is reversed."""
    num_points = 10
    z = np.arange(num_points)
    ordering = np.arange(num_points)
    algorithm = _algorithm_setup._Algorithm2D(None, z[::-1], assume_sorted=False)
    func, func_module, class_object = algorithm._get_function('asls', [whittaker])

    assert_array_equal(class_object.z, z)
    assert class_object._sort_order[0] is Ellipsis
    assert class_object._inverted_order[0] is Ellipsis
    assert algorithm._sort_order[0] is Ellipsis
    assert algorithm._inverted_order[0] is Ellipsis
    assert_array_equal(class_object._sort_order[1], ordering[::-1])
    assert_array_equal(class_object._inverted_order[1], ordering[::-1])
    assert_array_equal(class_object._sort_order[1], algorithm._sort_order[1])
    assert_array_equal(class_object._inverted_order[1], algorithm._inverted_order[1])


def test_get_function_sorting_xz():
    """Ensures the sort order is correct for the output class object when x and z are reversed."""
    num_x_points = 10
    num_z_points = 11
    x = np.arange(num_x_points)
    x_ordering = np.arange(num_x_points)
    z = np.arange(num_z_points)
    z_ordering = np.arange(num_z_points)

    algorithm = _algorithm_setup._Algorithm2D(x[::-1], z[::-1], assume_sorted=False)
    func, func_module, class_object = algorithm._get_function('asls', [whittaker])

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


@pytest.mark.parametrize('method_kwargs', (None, {'a': 2}))
def test_setup_optimizer(small_data2d, algorithm, method_kwargs):
    """Ensures output of _setup_optimizer is correct."""
    y, fit_func, func_module, output_kwargs, class_object = algorithm._setup_optimizer(
        small_data2d, 'asls', [whittaker], method_kwargs
    )

    assert isinstance(y, np.ndarray)
    assert_allclose(y, small_data2d)
    assert fit_func.__name__ == 'asls'
    assert func_module == 'whittaker'
    assert isinstance(output_kwargs, dict)
    assert isinstance(class_object, _algorithm_setup._Algorithm2D)


@pytest.mark.parametrize('copy_kwargs', (True, False))
def test_setup_optimizer_copy_kwargs(small_data2d, algorithm, copy_kwargs):
    """Ensures the copy behavior of the input keyword argument dictionary."""
    input_kwargs = {'a': 1}
    y, _, _, output_kwargs, _ = algorithm._setup_optimizer(
        small_data2d, 'asls', [whittaker], input_kwargs, copy_kwargs
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
