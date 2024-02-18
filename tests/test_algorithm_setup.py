# -*- coding: utf-8 -*-
"""Tests for pybaselines._algorithm_setup.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import _algorithm_setup, optimizers, polynomial, whittaker
from pybaselines._compat import dia_object
from pybaselines.utils import ParameterWarning

from .conftest import get_data


@pytest.fixture
def algorithm():
    """
    An _Algorithm class with x-data set to np.arange(10).

    Returns
    -------
    pybaselines._algorithm_setup._Algorithm
        An _Algorithm class for testing.
    """
    return _algorithm_setup._Algorithm(
        x_data=np.arange(10), assume_sorted=True, check_finite=False
    )


@pytest.mark.parametrize('diff_order', (1, 2, 3))
@pytest.mark.parametrize('lam', (1, 20))
@pytest.mark.parametrize('allow_lower', (True, False))
@pytest.mark.parametrize('reverse_diags', (True, False, None))
def test_setup_whittaker_diff_matrix(small_data, algorithm, lam, diff_order,
                                     allow_lower, reverse_diags):
    """Ensures output difference matrix diagonal data is in desired format."""
    if reverse_diags and allow_lower:
        # this configuration is never used
        return

    _ = algorithm._setup_whittaker(
        small_data, lam, diff_order, allow_lower=allow_lower, reverse_diags=reverse_diags
    )

    numpy_diff = np.diff(np.eye(small_data.shape[0]), diff_order, 0)
    desired_diagonals = dia_object(lam * (numpy_diff.T @ numpy_diff)).data[::-1]
    if allow_lower and not algorithm.whittaker_system.using_pentapy:
        # only include the lower diagonals
        desired_diagonals = desired_diagonals[diff_order:]

    # the diagonals should be in the opposite order as the diagonal matrix's data
    # if reverse_diags is False
    if reverse_diags or (algorithm.whittaker_system.using_pentapy and reverse_diags is not False):
        desired_diagonals = desired_diagonals[::-1]

    assert_allclose(algorithm.whittaker_system.penalty, desired_diagonals, 1e-10)


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_whittaker_weights(small_data, algorithm, weight_enum):
    """Ensures output weight array is correct."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0])
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        desired_weights = np.arange(small_data.shape[0])

    _, weight_array = algorithm._setup_whittaker(
        small_data, lam=1, diff_order=2, weights=weights
    )

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


def test_setup_whittaker_wrong_weight_shape(small_data, algorithm):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data, lam=1, diff_order=2, weights=weights)


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_whittaker_diff_matrix_fails(small_data, algorithm, diff_order):
    """Ensures using a diff_order < 1 with _setup_whittaker raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data, lam=1, diff_order=diff_order)


@pytest.mark.parametrize('diff_order', (4, 5))
def test_setup_whittaker_diff_matrix_warns(small_data, algorithm, diff_order):
    """Ensures using a diff_order > 3 with _setup_whittaker raises a warning."""
    with pytest.warns(ParameterWarning):
        algorithm._setup_whittaker(small_data, lam=1, diff_order=diff_order)


def test_setup_whittaker_negative_lam_fails(small_data, algorithm):
    """Ensures a negative lam value fails."""
    with pytest.raises(ValueError):
        algorithm._setup_whittaker(small_data, lam=-1)


def test_setup_whittaker_array_lam(small_data):
    """Ensures a lam that is a single array passes while larger arrays fail."""
    _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_whittaker(
        small_data, lam=[1]
    )
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_whittaker(
            small_data, lam=[1, 2]
        )


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_polynomial_weights(small_data, algorithm, weight_enum):
    """Ensures output weight array is correctly handled."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0])
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        desired_weights = np.arange(small_data.shape[0])

    _, weight_array = algorithm._setup_polynomial(small_data, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


def test_setup_polynomial_wrong_weight_shape(small_data, algorithm):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data, weights=weights)


@pytest.mark.parametrize('vander_enum', (0, 1, 2, 3))
@pytest.mark.parametrize('include_pinv', (True, False))
def test_setup_polynomial_vandermonde(small_data, algorithm, vander_enum, include_pinv):
    """Ensures that the Vandermonde matrix and the pseudo-inverse matrix are correct."""
    if vander_enum == 0:
        # no weights specified
        weights = None
        poly_order = 2
    elif vander_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        poly_order = 4
    elif vander_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        poly_order = 2
    elif vander_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        poly_order = 4

    output = algorithm._setup_polynomial(
        small_data, weights=weights, poly_order=poly_order, calc_vander=True,
        calc_pinv=include_pinv
    )
    if include_pinv:
        _, weight_array, pinv_matrix = output
    else:
        _, weight_array = output

    desired_vander = np.polynomial.polynomial.polyvander(
        np.polynomial.polyutils.mapdomain(
            algorithm.x, algorithm.x_domain, np.array([-1., 1.])
        ), poly_order
    )
    assert_allclose(desired_vander, algorithm.vandermonde, 1e-12)

    if include_pinv:
        desired_pinv = np.linalg.pinv(np.sqrt(weight_array)[:, np.newaxis] * desired_vander)
        assert_allclose(desired_pinv, pinv_matrix, 1e-10)


def test_setup_polynomial_negative_polyorder_fails(small_data, algorithm):
    """Ensures a negative poly_order raises an exception."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data, poly_order=-1)


def test_setup_polynomial_too_large_polyorder_fails(small_data, algorithm):
    """Ensures an exception is raised if poly_order has more than one value."""
    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data, poly_order=[1, 2])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data, poly_order=[1, 2, 3])

    with pytest.raises(ValueError):
        algorithm._setup_polynomial(small_data, poly_order=np.array([1, 2]))


def test_setup_smooth_shape(small_data, algorithm):
    """Ensures output y is correctly padded."""
    pad_length = 4
    y = algorithm._setup_smooth(small_data, pad_length, mode='edge')
    assert y.shape[0] == small_data.shape[0] + 2 * pad_length


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_classification_weights(small_data, algorithm, weight_enum):
    """Ensures output weight array is correctly handled in classification setup."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones(small_data.shape[0], bool)
    elif weight_enum == 1:
        # uniform 1 weighting, input as boolean dtype
        weights = np.ones(small_data.shape[0], bool)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points, input as ints
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0]).astype(bool)
    elif weight_enum == 3:
        # different weights for all points, weights input as a list of floats
        weights = np.arange(small_data.shape[0], dtype=float).tolist()
        desired_weights = np.arange(small_data.shape[0]).astype(bool)

    _, weight_array = algorithm._setup_classification(small_data, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


@pytest.mark.parametrize('num_knots', (5, 15, 100))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('penalized', (True, False))
def test_setup_spline_spline_basis(small_data, num_knots, spline_degree, penalized):
    """Ensures the spline basis function is correctly created."""
    fitter = _algorithm_setup._Algorithm(np.arange(len(small_data)))
    _ = fitter._setup_spline(
        small_data, weights=None, spline_degree=spline_degree, num_knots=num_knots,
        penalized=True
    )

    assert fitter.pspline.basis.shape[0] == len(small_data)
    # num_knots == number of inner knots with min and max points counting as
    # the first and last inner knots; then add `degree` extra knots
    # on each end to accomodate the final polynomial on each end; therefore,
    # total number of knots = num_knots + 2 * degree; the number of basis
    # functions is total knots - (degree + 1), so the ultimate
    # shape of the basis matrix should be num_knots + degree - 1
    assert fitter.pspline.basis.shape[1] == num_knots + spline_degree - 1


@pytest.mark.parametrize('lam', (1, 20))
@pytest.mark.parametrize('diff_order', (1, 2, 3, 4))
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (5, 50, 100))
def test_setup_spline_diff_matrix(small_data, lam, diff_order, spline_degree, num_knots):
    """Ensures output difference matrix diagonal data is in desired format."""
    fitter = _algorithm_setup._Algorithm(np.arange(len(small_data)))
    _ = fitter._setup_spline(
        small_data, weights=None, spline_degree=spline_degree, num_knots=num_knots,
        penalized=True, diff_order=diff_order, lam=lam
    )

    num_bases = num_knots + spline_degree - 1
    numpy_diff = np.diff(np.eye(num_bases), diff_order, axis=0)
    desired_diagonals = lam * dia_object(numpy_diff.T @ numpy_diff).data[::-1][diff_order:]
    if diff_order < spline_degree:
        padding = np.zeros((spline_degree - diff_order, desired_diagonals.shape[1]))
        desired_diagonals = np.concatenate((desired_diagonals, padding))

    assert_allclose(fitter.pspline.penalty, desired_diagonals, 1e-10, 1e-12)


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('spline_degree', (1, 2, 3, 4))
@pytest.mark.parametrize('num_knots', (5, 50, 100))
def test_setup_spline_too_high_diff_order(small_data, spline_degree, num_knots):
    """
    Ensures an exception is raised when the difference order is >= number of basis functions.

    The number of basis functions is equal to the number of knots + the spline degree - 1.
    Tests both difference order equal to and greater than the number of basis functions.

    """
    diff_order = num_knots + spline_degree - 1
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, weights=None, spline_degree=spline_degree, num_knots=num_knots,
            penalized=True, diff_order=diff_order
        )

    diff_order += 1
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, weights=None, spline_degree=spline_degree, num_knots=num_knots,
            penalized=True, diff_order=diff_order
        )


@pytest.mark.parametrize('num_knots', (0, 1))
def test_setup_spline_too_few_knots(small_data, num_knots):
    """Ensures an error is raised if the number of knots is less than 2."""
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, weights=None, spline_degree=3, num_knots=num_knots,
            penalized=True, diff_order=1
        )


def test_setup_spline_wrong_weight_shape(small_data):
    """Ensures that an exception is raised if input weights and data are different shapes."""
    weights = np.ones(small_data.shape[0] + 1)
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, weights=weights
        )


@pytest.mark.parametrize('diff_order', (0, -1))
def test_setup_spline_diff_matrix_fails(small_data, diff_order):
    """Ensures using a diff_order < 1 with _setup_spline raises an exception."""
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, diff_order=diff_order
        )


@pytest.mark.parametrize('diff_order', (5, 6))
def test_setup_spline_diff_matrix_warns(small_data, diff_order):
    """Ensures using a diff_order > 4 with _setup_spline raises a warning."""
    with pytest.warns(ParameterWarning):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, diff_order=diff_order
        )


def test_setup_spline_negative_lam_fails(small_data):
    """Ensures a negative lam value fails."""
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, lam=-1
        )


@pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
def test_setup_spline_weights(small_data, algorithm, weight_enum):
    """Ensures output weight array is correct."""
    if weight_enum == 0:
        # no weights specified
        weights = None
        desired_weights = np.ones_like(small_data)
    elif weight_enum == 1:
        # uniform 1 weighting
        weights = np.ones_like(small_data)
        desired_weights = weights.copy()
    elif weight_enum == 2:
        # different weights for all points
        weights = np.arange(small_data.shape[0])
        desired_weights = np.arange(small_data.shape[0])
    elif weight_enum == 3:
        # different weights for all points, and weights input as a list
        weights = np.arange(small_data.shape[0]).tolist()
        desired_weights = np.arange(small_data.shape[0])

    _, weight_array = algorithm._setup_spline(small_data, lam=1, diff_order=2, weights=weights)

    assert isinstance(weight_array, np.ndarray)
    assert_array_equal(weight_array, desired_weights)


def test_setup_spline_array_lam(small_data):
    """Ensures a lam that is a single array passes while larger arrays fail."""
    _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(small_data, lam=[1])
    with pytest.raises(ValueError):
        _algorithm_setup._Algorithm(np.arange(len(small_data)))._setup_spline(
            small_data, lam=[1, 2]
        )


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
    assert isinstance(class_object, _algorithm_setup._Algorithm)


def test_get_function_fails_wrong_method(algorithm):
    """Ensures _get_function fails when an no function with the input name is available."""
    with pytest.raises(AttributeError):
        algorithm._get_function('unknown function', [optimizers])


def test_get_function_fails_no_module(algorithm):
    """Ensures _get_function fails when not given any modules to search."""
    with pytest.raises(AttributeError):
        algorithm._get_function('collab_pls', [])


def test_get_function_sorting():
    """Ensures the sort order is correct for the output class object."""
    num_points = 10
    x = np.arange(num_points)
    ordering = np.arange(num_points)
    algorithm = _algorithm_setup._Algorithm(x[::-1], assume_sorted=False)
    func, func_module, class_object = algorithm._get_function('asls', [whittaker])

    assert_array_equal(class_object.x, x)
    assert_array_equal(class_object._sort_order, ordering[::-1])
    assert_array_equal(class_object._inverted_order, ordering[::-1])
    assert_array_equal(class_object._sort_order, algorithm._sort_order)
    assert_array_equal(class_object._inverted_order, algorithm._inverted_order)


@pytest.mark.parametrize('method_kwargs', (None, {'a': 2}))
def test_setup_optimizer(small_data, algorithm, method_kwargs):
    """Ensures output of _setup_optimizer is correct."""
    y, fit_func, func_module, output_kwargs, class_object = algorithm._setup_optimizer(
        small_data, 'asls', [whittaker], method_kwargs
    )

    assert isinstance(y, np.ndarray)
    assert_allclose(y, small_data)
    assert fit_func.__name__ == 'asls'
    assert func_module == 'whittaker'
    assert isinstance(output_kwargs, dict)
    assert isinstance(class_object, _algorithm_setup._Algorithm)


@pytest.mark.parametrize('copy_kwargs', (True, False))
def test_setup_optimizer_copy_kwargs(small_data, algorithm, copy_kwargs):
    """Ensures the copy behavior of the input keyword argument dictionary."""
    input_kwargs = {'a': 1}
    y, _, _, output_kwargs, _ = algorithm._setup_optimizer(
        small_data, 'asls', [whittaker], input_kwargs, copy_kwargs
    )

    output_kwargs['a'] = 2
    if copy_kwargs:
        assert input_kwargs['a'] == 1
    else:
        assert input_kwargs['a'] == 2


@pytest.mark.parametrize('input_x', (True, False))
@pytest.mark.parametrize('check_finite', (True, False))
@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_class_init(input_x, check_finite, assume_sorted, output_dtype, change_order):
    """Tests the initialization of _Algorithm objects."""
    sort_order = slice(0, 10)
    if input_x:
        x, _ = get_data()
        expected_x = x.copy()
        if change_order:
            x[sort_order] = x[sort_order][::-1]
            if assume_sorted:
                expected_x[sort_order] = expected_x[sort_order][::-1]
    else:
        x = None
        expected_x = None

    algorithm = _algorithm_setup._Algorithm(
        x, check_finite=check_finite, assume_sorted=assume_sorted, output_dtype=output_dtype
    )
    assert_array_equal(algorithm.x, expected_x)
    assert algorithm._check_finite == check_finite
    assert algorithm._dtype == output_dtype

    if input_x:
        assert algorithm._len == len(x)
    else:
        assert algorithm._len is None

    if not assume_sorted and change_order and input_x:
        order = np.arange(len(x))
        order[sort_order] = order[sort_order][::-1]
        assert_array_equal(algorithm._sort_order, order)
        assert_array_equal(algorithm._inverted_order, order.argsort())
    else:
        assert algorithm._sort_order is None
        assert algorithm._inverted_order is None

    # ensure attributes are correctly initialized
    assert algorithm.poly_order == -1
    assert algorithm.pspline is None
    assert algorithm.whittaker_system is None
    assert algorithm.vandermonde is None


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
def test_algorithm_return_results(assume_sorted, output_dtype, change_order):
    """Ensures the _return_results method returns the correctly sorted outputs."""
    x, _ = get_data()
    baseline = np.arange(len(x))
    # 'a' values will be sorted and 'b' values will be kept the same
    params = {
        'a': np.arange(len(x)),
        'b': np.arange(len(x))
    }
    sort_indices = slice(0, 100)
    params['b'][sort_indices] = params['b'][sort_indices]
    if change_order:
        x[sort_indices] = x[sort_indices][::-1]

    expected_params = {
        'a': np.arange(len(x)),
        'b': np.arange(len(x))
    }
    expected_params['b'][sort_indices] = expected_params['b'][sort_indices]
    expected_baseline = baseline.copy()
    if change_order and not assume_sorted:
        expected_baseline[sort_indices] = expected_baseline[sort_indices][::-1]
        expected_params['a'][sort_indices] = expected_params['a'][sort_indices][::-1]

    algorithm = _algorithm_setup._Algorithm(
        x, assume_sorted=assume_sorted, output_dtype=output_dtype, check_finite=False
    )
    output, output_params = algorithm._return_results(
        baseline, params, dtype=output_dtype, sort_keys=('a',)
    )

    assert_allclose(output, expected_baseline, 1e-16, 1e-16)
    assert output.dtype == output_dtype
    for key, value in expected_params.items():
        assert_array_equal(value, output_params[key])


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
@pytest.mark.parametrize('skip_sorting', (True, False))
def test_algorithm_register(assume_sorted, output_dtype, change_order, list_input, skip_sorting):
    """
    Ensures the _register wrapper method returns the correctly sorted outputs.

    The input y-values within the wrapped function should be correctly sorted
    if `assume_sorted` is False, while the output baseline should always match
    the ordering of the input y-values. The output params should have an inverted
    sort order to also match the ordering of the input y-values if `assume_sorted`
    is False.

    """
    x = np.arange(20)
    y = 5 * x
    y_dtype = y.dtype
    sort_indices = slice(0, 10)

    class SubClass(_algorithm_setup._Algorithm):
        # 'a' values will be sorted and 'b' values will be kept the same
        @_algorithm_setup._Algorithm._register(sort_keys=('a',))
        def func(self, data, *args, **kwargs):
            """For checking sorting of output parameters."""
            expected_x = np.arange(20)
            if change_order and assume_sorted:
                expected_x[sort_indices] = expected_x[sort_indices][::-1]
            expected_input = 5 * expected_x

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_input, 1e-16, 1e-16)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-16, 1e-16)

            params = {
                'a': np.arange(len(x)),
                'b': np.arange(len(x))
            }
            return 1 * data, params

        @_algorithm_setup._Algorithm._register(sort_keys=('a',), skip_sorting=skip_sorting)
        def func2(self, data, *args, **kwargs):
            expected_x = np.arange(20)
            expected_input = 5 * expected_x
            if change_order and assume_sorted:
                expected_x[sort_indices] = expected_x[sort_indices][::-1]
            if change_order and (assume_sorted or skip_sorting):
                expected_input[sort_indices] = expected_input[sort_indices][::-1]

            assert_allclose(data, expected_input, 1e-14, 1e-14)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)

            params = {
                'a': np.arange(len(x)),
                'b': np.arange(len(x))
            }
            return 1 * data, params

    if change_order:
        x[sort_indices] = x[sort_indices][::-1]
        y[sort_indices] = y[sort_indices][::-1]
    expected_baseline = (1 * y).astype(output_dtype)
    if output_dtype is None:
        expected_dtype = y_dtype
    else:
        expected_dtype = expected_baseline.dtype
    if list_input:
        x = x.tolist()
        y = y.tolist()

    expected_params = {
        'a': np.arange(len(x)),
        'b': np.arange(len(x))
    }
    if change_order and not assume_sorted:
        # if assume_sorted is False, the param order should be inverted to match
        # the input y-order
        expected_params['a'][sort_indices] = expected_params['a'][sort_indices][::-1]

    algorithm = SubClass(
        x, assume_sorted=assume_sorted, output_dtype=output_dtype, check_finite=False
    )
    output, output_params = algorithm.func(y)

    # baseline should always match y-order on the output; only sorted within the
    # function
    assert_allclose(output, expected_baseline, 1e-16, 1e-16)
    assert isinstance(output, np.ndarray)
    assert output.dtype == expected_dtype
    for key, value in expected_params.items():
        assert_array_equal(value, output_params[key])

    output2, output_params2 = algorithm.func2(y)

    # baseline should always match y-order on the output; only sorted within the
    # function
    assert_allclose(output2, expected_baseline, 1e-16, 1e-16)
    assert isinstance(output2, np.ndarray)
    for key, value in expected_params.items():
        assert_array_equal(value, output_params2[key])


def test_class_wrapper():
    """Ensures the class wrapper function correctly processes inputs for _Algorithm classes."""
    default_b = 2
    default_c = 3

    class Dummy:

        def __init__(self, x_data=None):
            self.x = x_data

        def func(self, a, b=default_b, c=default_c):
            return a, b, c, self.x

    def func(a, b=default_b, c=default_c, x_data=None):
        return a, b, c, x_data

    wrapper = _algorithm_setup._class_wrapper(Dummy)
    func2 = wrapper(func)

    assert func(0) == (0, default_b, default_c, None)
    assert func(0) == Dummy().func(0)
    assert func(0) == func2(0)

    a = 5
    b = 9
    c = 10
    x = 10

    assert func(a, b, c, x) == (a, b, c, x)
    assert func(a, b, c, x) == Dummy(x).func(a, b, c)
    assert func(a, b, c, x) == func2(a, b, c, x)


def test_class_wrapper_kwargs():
    """Ensures the class wrapper function correctly processes kwargs for _Algorithm classes."""
    default_b = 2
    default_c = 3

    class Dummy:

        def __init__(self, x_data=None):
            self.x = x_data

        def func(self, a, b=default_b, c=default_c, **kwargs):
            return a, b, c, self.x, kwargs

    def func(a, b=default_b, c=default_c, x_data=None, **kwargs):
        return a, b, c, x_data, kwargs

    wrapper = _algorithm_setup._class_wrapper(Dummy)
    func2 = wrapper(func)

    d = 10

    assert func(0, d=d) == (0, default_b, default_c, None, {'d': d})
    assert func(0, d=d) == Dummy().func(0, d=d)
    assert func(0, d=d) == func2(0, d=d)

    a = 5
    b = 9
    c = 10
    x = 10

    assert func(a, b, c, x, d=d) == (a, b, c, x, {'d': d})
    assert func(a, b, c, x, d=d) == Dummy(x).func(a, b, c, d=d)
    assert func(a, b, c, x, d=d) == func2(a, b, c, x, d=d)


def test_override_x(algorithm):
    """Ensures the `override_x` method correctly initializes with the new x values."""
    new_len = 20
    new_x = np.arange(new_len)
    with algorithm._override_x(new_x) as new_algorithm:
        assert len(new_algorithm.x) == new_len
        assert new_algorithm._len == new_len
        assert new_algorithm.poly_order == -1
        assert new_algorithm.vandermonde is None
        assert new_algorithm.whittaker_system is None
        assert new_algorithm.pspline is None


def test_override_x_polynomial(algorithm):
    """Ensures the polynomial attributes are correctly reset and then returned by override_x."""
    old_len = len(algorithm.x)
    poly_order = 2
    new_poly_order = 3

    algorithm._setup_polynomial(np.arange(old_len), poly_order=poly_order, calc_vander=True)
    # sanity check
    assert algorithm.vandermonde.shape == (old_len, poly_order + 1)
    assert algorithm.poly_order == poly_order
    old_vandermonde = algorithm.vandermonde.copy()

    new_len = 20
    new_x = np.arange(new_len)
    with algorithm._override_x(new_x) as new_algorithm:
        assert new_algorithm.vandermonde is None
        new_algorithm._setup_polynomial(
            np.arange(new_len), poly_order=new_poly_order, calc_vander=True
        )
        assert new_algorithm.vandermonde.shape == (new_len, new_poly_order + 1)
        assert new_algorithm.poly_order == new_poly_order

    # ensure vandermonde is reset
    assert algorithm.vandermonde.shape == (old_len, poly_order + 1)
    assert_allclose(old_vandermonde, algorithm.vandermonde, rtol=1e-14, atol=1e-14)
    assert algorithm.poly_order == poly_order


def test_override_x_whittaker(algorithm):
    """Ensures the whittaker attributes are correctly reset and then returned by override_x."""
    old_len = len(algorithm.x)
    diff_order = 2
    new_diff_order = 3

    algorithm._setup_whittaker(np.arange(old_len), diff_order=diff_order)
    # sanity check
    assert algorithm.whittaker_system.diff_order == diff_order

    new_len = 20
    new_x = np.arange(new_len)
    with algorithm._override_x(new_x) as new_algorithm:
        assert new_algorithm.whittaker_system is None
        new_algorithm._setup_whittaker(np.arange(new_len), diff_order=new_diff_order)
        assert new_algorithm.whittaker_system.diff_order == new_diff_order

    # ensure Whittaker system is reset
    assert algorithm.whittaker_system.diff_order == diff_order


def test_override_x_spline(algorithm):
    """Ensures the spline attributes are correctly reset and then returned by override_x."""
    old_len = len(algorithm.x)
    spline_degree = 2
    new_spline_degree = 3

    algorithm._setup_spline(np.arange(old_len), spline_degree=spline_degree)
    # sanity check
    assert algorithm.pspline.spline_degree == spline_degree
    old_basis = algorithm.pspline.basis.toarray().copy()

    new_len = 20
    new_x = np.arange(new_len)
    with algorithm._override_x(new_x) as new_algorithm:
        assert new_algorithm.pspline is None
        new_algorithm._setup_spline(np.arange(new_len), spline_degree=new_spline_degree)
        assert new_algorithm.pspline.spline_degree == new_spline_degree
        assert old_basis.shape != algorithm.pspline.basis.shape

    # ensure P-spline system is reset
    assert algorithm.pspline.spline_degree == spline_degree
    assert old_basis.shape == algorithm.pspline.basis.shape
    assert_allclose(algorithm.pspline.basis.toarray(), old_basis, 1e-14, 1e-14)
