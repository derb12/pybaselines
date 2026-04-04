# -*- coding: utf-8 -*-
"""Tests for pybaselines._nd._algorithm_setup.

@author: Donald Erb
Created on March 25, 2026

"""

from inspect import signature

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from pybaselines import _algorithm_setup as _algorithm_setup1d
from pybaselines.utils import SortingWarning
from pybaselines.two_d import _algorithm_setup as _algorithm_setup2d
from pybaselines._nd import _algorithm_setup

from ..base_tests import get_data, get_data2d


def test_handle_io_signature():
    """Ensures _handle_io has the same signature and defaults as _Algorithm and _Algorithm2D."""
    wrapper_parameters = signature(_algorithm_setup._handle_io).parameters

    algorithm_parameters = signature(_algorithm_setup1d._Algorithm._handle_io).parameters
    algorithm2d_parameters = signature(_algorithm_setup2d._Algorithm2D._handle_io).parameters

    for alg_parameters in (algorithm_parameters, algorithm2d_parameters):
        assert len(wrapper_parameters) == len(alg_parameters)
        # ensure key and values for all parameters match for both signatures
        for key in wrapper_parameters:
            assert key in alg_parameters
            wrapper_value = alg_parameters[key].default
            algorithm_value = alg_parameters[key].default
            # all the defaults should just be booleans and empty tuples
            assert wrapper_value == algorithm_value, f'Parameter mismatch for key "{key}"'


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
@pytest.mark.parametrize('skip_sorting', (True, False))
def test_handle_io_1d(assume_sorted, output_dtype, change_order, list_input, skip_sorting):
    """Ensures the _handle_io wrapper passes all tests expected by _Algorithm2D._handle_io."""
    x = np.arange(20)
    y = 5 * x
    sort_indices = slice(0, 10)

    class SubClass(_algorithm_setup1d._Algorithm):
        # 'a' values will be sorted and 'b' values will be kept the same
        @_algorithm_setup._handle_io(sort_keys=('a',))
        def func(self, data, *args, **kwargs):
            """For checking sorting of output parameters."""
            expected_x = np.arange(20)
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

        @_algorithm_setup._handle_io(sort_keys=('a',), skip_sorting=skip_sorting)
        def func2(self, data, *args, **kwargs):
            """For checking skip_sorting."""
            expected_x = np.arange(20)
            expected_input = 5 * expected_x
            if change_order and skip_sorting:
                expected_input[sort_indices] = expected_input[sort_indices][::-1]

            assert_allclose(data, expected_input, 1e-14, 1e-14)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)

            params = {
                'a': np.arange(len(x)),
                'b': np.arange(len(x))
            }
            return 1 * data, params

        @_algorithm_setup._handle_io(require_unique=False)
        def func3(self, data, *args, **kwargs):
            """For ensuring require_unique works as intedended."""
            return 1 * data, {}

        @_algorithm_setup._handle_io(require_unique=True)
        def func4(self, data, *args, **kwargs):
            """For ensuring require_unique works as intedended."""
            return 1 * data, {}

    if change_order:
        x[sort_indices] = x[sort_indices][::-1]
        y[sort_indices] = y[sort_indices][::-1]
    expected_baseline = (1 * y).astype(output_dtype)
    if output_dtype is None:
        expected_dtype = y.dtype
    else:
        expected_dtype = expected_baseline.dtype
    if list_input:
        x = x.tolist()
        y = y.tolist()

    expected_params = {
        'a': np.arange(len(x)),
        'b': np.arange(len(x))
    }
    if change_order:
        expected_params['a'][sort_indices] = expected_params['a'][sort_indices][::-1]

    if change_order and assume_sorted:
        with pytest.warns(SortingWarning):
            algorithm = SubClass(
                x, assume_sorted=assume_sorted, output_dtype=output_dtype, check_finite=False
            )
    else:
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

    assert not algorithm._validated_x  # has not had a need to validate x yet
    output = algorithm.func4(y)
    assert algorithm._validated_x

    new_x = np.arange(20)
    new_x[0] = new_x[1]
    new_algorithm = SubClass(new_x)
    # ensure calling a method that does not require unique x does not validate or raise an error
    out = new_algorithm.func3(y)
    assert not new_algorithm._validated_x
    with pytest.raises(ValueError):
        out = new_algorithm.func4(y)


@pytest.mark.parametrize('input_x', (True, False))
def test_algorithm_handle_io_1d_2d(data_fixture, input_x):
    """Ensures 2D data is allowed for 1D algorithms only when specified.

    Also checks _Algorithm setup when given 2D data as the first call.

    """
    _, expected_y = get_data()

    class SubClass(_algorithm_setup1d._Algorithm):

        @_algorithm_setup._handle_io
        def func(self, data, *args, **kwargs):
            """Errors if input is not 1D."""
            assert data.ndim == 1
            assert data.shape == expected_y.shape
            return data, {}

        @_algorithm_setup._handle_io(ensure_dims=False)
        def func2(self, data, *args, **kwargs):
            """Allows 2D data."""
            assert data.ndim == 2
            assert data.shape[1:] == expected_y.shape
            return data, {}

    x_, y_1d = data_fixture
    x = None
    if input_x:
        x = x_
        initial_size = len(x)
        initial_shape = (len(x),)
    else:
        initial_size = None
        initial_shape = (None,)

    input_y = np.stack((y_1d, y_1d), axis=0)
    assert input_y.shape == (2, *y_1d.shape)  # sanity check for correct setup

    algorithm = SubClass(x)
    assert algorithm._shape == initial_shape
    assert algorithm._size == initial_size

    with pytest.raises(ValueError, match='input data must be a one dimensional'):
        algorithm.func(input_y)
    assert algorithm._shape == initial_shape

    # should run without issues and set stored shape correctly
    output, _ = algorithm.func2(input_y)
    assert algorithm._shape == y_1d.shape
    assert algorithm._size == y_1d.size
    assert output.shape == input_y.shape


@pytest.mark.parametrize('assume_sorted', (True, False))
@pytest.mark.parametrize('output_dtype', (None, int, float, np.float64))
@pytest.mark.parametrize('change_order', (True, False))
@pytest.mark.parametrize('skip_sorting', (True, False))
@pytest.mark.parametrize('list_input', (True, False))
def test_handle_io_2d(assume_sorted, output_dtype, change_order, skip_sorting, list_input):
    """Ensures the _handle_io wrapper passes all tests expected by _Algorithm2D._handle_io."""
    x, z, y = get_data2d()

    class SubClass(_algorithm_setup2d._Algorithm2D):
        # 'a' values will be sorted and 'b' values will be kept the same
        @_algorithm_setup._handle_io(sort_keys=('a', 'd'), reshape_keys=('c', 'd'))
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

        @_algorithm_setup._handle_io
        def func2(self, data, *args, **kwargs):
            """For checking reshaping output baseline."""
            expected_x, expected_z, expected_y = get_data2d()

            assert isinstance(data, np.ndarray)
            assert_allclose(data, expected_y, 1e-14, 1e-14)
            assert isinstance(self.x, np.ndarray)
            assert_allclose(self.x, expected_x, 1e-14, 1e-14)
            assert isinstance(self.z, np.ndarray)
            assert_allclose(self.z, expected_z, 1e-14, 1e-14)

            return 1 * data.flatten(), {}

        @_algorithm_setup._handle_io
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

        @_algorithm_setup._handle_io(
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

        @_algorithm_setup._handle_io(require_unique=False)
        def func5(self, data, *args, **kwargs):
            """For ensuring require_unique works as intended."""
            return 1 * data, {}

        @_algorithm_setup._handle_io(require_unique=True)
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
    expected_baseline = (1 * y).astype(output_dtype)
    if output_dtype is None:
        expected_dtype = y.dtype
    else:
        expected_dtype = expected_baseline.dtype
    if list_input:
        x = x.tolist()
        z = z.tolist()
        y = y.tolist()

    if change_order:
        expected_params['a'] = expected_params['a'][::-1, ::-1]
        expected_params['d'] = expected_params['d'][::-1, ::-1]

    if assume_sorted and change_order:
        with pytest.warns(SortingWarning):
            algorithm = SubClass(
                x, z, check_finite=False, assume_sorted=assume_sorted,
                output_dtype=output_dtype
            )
    else:
        algorithm = SubClass(
            x, z, check_finite=False, assume_sorted=assume_sorted, output_dtype=output_dtype
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


def test_algorithm_handle_io_2d_no_data_fails():
    """Ensures an error is raised if the input data is None."""

    class SubClass(_algorithm_setup2d._Algorithm2D):

        @_algorithm_setup._handle_io
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._handle_io
        def func2(self, data, *args, **kwargs):
            """For checking closed decorator."""
            return data, {}

    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func()
    with pytest.raises(TypeError, match='"data" cannot be None'):
        SubClass().func2()


def test_algorithm_handle_io_2d_1d_fails(data_fixture):
    """Ensures an error is raised if 1D data is used for 2D algorithms."""

    class SubClass(_algorithm_setup2d._Algorithm2D):

        @_algorithm_setup._handle_io
        def func(self, data, *args, **kwargs):
            """For checking empty decorator."""
            return data, {}

        @_algorithm_setup._handle_io
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
def test_algorithm_handle_io_2d_3d(data_fixture2d, input_x, input_z):
    """Ensures 3D data is allowed for 2D algorithms only when specified.

    Also checks _Algorithm2D setup when given 3D data as the first call.

    """
    _, _, expected_y = get_data2d()

    class SubClass(_algorithm_setup2d._Algorithm2D):

        @_algorithm_setup._handle_io
        def func(self, data, *args, **kwargs):
            """Errors if input is not 2D."""
            assert data.ndim == 2
            assert data.shape == expected_y.shape
            return data, {}

        @_algorithm_setup._handle_io(ensure_dims=False)
        def func2(self, data, *args, **kwargs):
            """Allows 3D data."""
            assert data.ndim == 3
            assert data.shape[1:] == expected_y.shape
            return data, {}

        @_algorithm_setup._handle_io(ensure_dims=False)
        def func3(self, data, *args, **kwargs):
            """For checking reshaping output baseline for 3D input raveled on last axis."""
            assert data.ndim == 3
            assert data.shape[1:] == expected_y.shape

            return 1 * data.reshape(data.shape[0], -1), {}

    x_, z_, y_2d = data_fixture2d
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

    input_y = np.stack((y_2d, y_2d), axis=0)
    assert input_y.shape == (2, *y_2d.shape)  # sanity check for correct setup

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

    output2, _ = algorithm.func3(input_y)
    assert output2.shape == input_y.shape
