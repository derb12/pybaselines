# -*- coding: utf-8 -*-
"""Setup code for testing pybaselines.

@author: Donald Erb
Created on March 20, 2021

"""

from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
import inspect

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest


try:
    import pentapy  # noqa
except ImportError:
    _HAS_PENTAPY = False
else:
    _HAS_PENTAPY = True

has_pentapy = pytest.mark.skipif(not _HAS_PENTAPY, reason='pentapy is not installed')


def gaussian(x, height=1.0, center=0.0, sigma=1.0):
    """
    Generates a gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center : float, optional
        The center of the distribution. Default is 0.0.
    sigma : float, optional
        The standard deviation of the distribution. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The gaussian distribution evaluated with x.

    Notes
    -----
    This is the same code as in pybaselines.utils.gaussian, but
    this removes the dependence on pybaselines so that if an error
    with pybaselines occurs, this will be unaffected.

    """
    return height * np.exp(-0.5 * ((x - center)**2) / sigma**2)


def gaussian2d(x, z, height=1.0, center_x=0.0, center_z=0.0, sigma_x=1.0, sigma_z=1.0):
    """
    Generates a Gaussian distribution based on height, center, and sigma.

    Parameters
    ----------
    x : numpy.ndarray, shape (M, N)
        The x-values at which to evaluate the distribution.
    z : numpy.ndarray, shape (M, N)
        The z-values at which to evaluate the distribution.
    height : float, optional
        The maximum height of the distribution. Default is 1.0.
    center_x : float, optional
        The center of the distribution in the x-axis. Default is 0.0.
    sigma_x : float, optional
        The standard deviation of the distribution in the x-axis. Default is 1.0.
    center_z : float, optional
        The center of the distribution in the z-axis. Default is 0.0.
    sigma_z : float, optional
        The standard deviation of the distribution in the z-axis. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The Gaussian distribution evaluated with x.

    Notes
    -----
    This is the same code as in pybaselines.utils.gaussian2d, but
    this removes the dependence on pybaselines so that if an error
    with pybaselines occurs, this will be unaffected.

    """
    return height * gaussian(x, 1, center_x, sigma_x) * gaussian(z, 1, center_z, sigma_z)


def get_data(include_noise=True, num_points=1000):
    """Creates x- and y-data for testing.

    Parameters
    ----------
    include_noise : bool, optional
        If True (default), will include noise with the y-data.
    num_points : int, optional
        The number of data points to use. Default is 1000.

    Returns
    -------
    x_data : numpy.ndarray
        The x-values.
    y_data : numpy.ndarray
        The y-values.

    """
    x_data = np.linspace(1, 100, num_points)
    y_data = (
        500  # constant baseline
        + 0.1 * x_data  # linear baseline
        + gaussian(x_data, 10, 25)
        + gaussian(x_data, 20, 50)
        + gaussian(x_data, 10, 75)
    )
    if include_noise:
        y_data += np.random.default_rng(0).normal(0, 0.5, x_data.size)

    return x_data, y_data


def get_data2d(include_noise=True, num_points=(30, 41)):
    """Creates x-, z-, and y-data for testing.

    Parameters
    ----------
    include_noise : bool, optional
        If True (default), will include noise with the y-data.
    num_points : Container(int, int), optional
        The number of data points to use for x, and z, respectively. Default
        is (30, 41), which uses different numbers so that any issues caused
        by not having a square matrix will be seen.

    Returns
    -------
    x_data : numpy.ndarray
        The x-values.
    z_data : numpy.ndarray
        The z-values
    y_data : numpy.ndarray
        The y-values.

    """
    x_num_points, z_num_points = num_points
    x_data = np.linspace(1, 100, x_num_points)
    z_data = np.linspace(1, 120, z_num_points)
    X, Z = np.meshgrid(x_data, z_data, indexing='ij')
    y_data = (
        500  # constant baseline
        + gaussian2d(X, Z, 10, 25, 25)
        + gaussian2d(X, Z, 20, 50, 50)
        + gaussian2d(X, Z, 10, 75, 75)
    )
    if include_noise:
        y_data += np.random.default_rng(0).normal(0, 0.5, y_data.shape)

    return x_data, z_data, y_data


def get_2dspline_inputs(num_knots=5, spline_degree=3, lam=1, diff_order=2):
    """Helper function to handle array-like values for simple cases in testing."""
    if isinstance(num_knots, int):
        num_knots_x = num_knots
        num_knots_z = num_knots
    else:
        num_knots_x, num_knots_z = num_knots
    if isinstance(spline_degree, int):
        spline_degree_x = spline_degree
        spline_degree_z = spline_degree
    else:
        spline_degree_x, spline_degree_z = spline_degree
    if isinstance(lam, (int, float)):
        lam_x = lam
        lam_z = lam
    else:
        lam_x, lam_z = lam
    if isinstance(diff_order, int):
        diff_order_x = diff_order
        diff_order_z = diff_order
    else:
        diff_order_x, diff_order_z = diff_order

    return (
        num_knots_x, num_knots_z, spline_degree_x, spline_degree_z,
        lam_x, lam_z, diff_order_x, diff_order_z
    )


@pytest.fixture
def small_data():
    """A small array of data for testing."""
    return np.arange(10, dtype=float)


@pytest.fixture
def small_data2d():
    """A small array of data for testing."""
    return np.arange(60, dtype=float).reshape(6, 10)


@pytest.fixture()
def data_fixture():
    """Test fixture for creating x- and y-data for testing."""
    return get_data()


@pytest.fixture()
def data_fixture2d():
    """Test fixture for creating x-, z-, and y-data for testing."""
    return get_data2d()


@pytest.fixture()
def no_noise_data_fixture():
    """Test fixture that creates x- and y-data without noise for testing."""
    return get_data(include_noise=False)


@pytest.fixture()
def no_noise_data_fixture2d():
    """
    Test fixture that creates x-, z-, and y-data without noise for testing.

    Reduces the number of data points since this is used for testing that numerical
    issues are avoided for large iterations in spline and Whittaker functions, which
    can otherwise be time consuming.
    """
    return get_data2d(include_noise=False, num_points=(20, 31))


def changing_dataset(data_size=1000, dataset_size=100):
    """
    Creates a dataset containing data with different baselines.

    Useful for comparing results where issues repeatedly fitting a baseline are possible.

    Parameters
    ----------
    data_size : int, optional
        The number of points for the data. Default is 1000.
    dataset_size : int, optional
        The number of data within the datasset. Default is 100.

    Returns
    -------
    x : numpy.ndarray, shape (`data_size`,)
        The x-values for the data.
    dataset : numpy.ndarray, shape (`dataset_size`, `data_size`)
        The dataset for testing.

    """
    x = np.linspace(0, 1000, data_size)
    signal = gaussian(x, 6, 550, 6)
    baseline = 5 + 15 * np.exp(-x / 1000) + gaussian(x, 5, 700, 300)
    # change baseline shapes so that differences are more pronounced between each fit
    baselines = np.linspace(-1, 1, dataset_size).reshape(-1, 1) * baseline
    noise = np.random.default_rng(0).normal(
        0, 0.2, (dataset_size * x.size)
    ).reshape(dataset_size, -1)

    dataset = signal + noise + baselines

    return x, dataset


def changing_dataset2d(data_size=(40, 33), dataset_size=20):
    """
    Creates a dataset containing data with different baselines.

    Useful for comparing results where issues repeatedly fitting a baseline are possible.

    Parameters
    ----------
    data_size : Container(int, int), optional
        The number of data points to use for x, and z, respectively. Default
        is (50, 40), which uses different numbers so that any issues caused
        by not having a square matrix will be seen.
    dataset_size : int, optional
        The number of data within the datasset. Default is 20.

    Returns
    -------
    x : numpy.ndarray, shape (`data_size[0]`,)
        The x-values for the data.
    z : numpy.ndarray, shape (`data_size[1]`,)
        The z-values for the data.
    dataset : numpy.ndarray, shape (`dataset_size`, `data_size`)
        The dataset for testing.

    """
    x_num_points, z_num_points = data_size
    x = np.linspace(1, 100, x_num_points)
    z = np.linspace(1, 120, z_num_points)
    X, Z = np.meshgrid(x, z, indexing='ij')
    signal = gaussian2d(X, Z, 50, 50, 60)

    baseline = 5 + 0.5 * X + 0.005 * Z**2
    # change baseline shapes so that differences are more pronounced between each fit
    baselines = np.array([baseline * val for val in np.linspace(-1, 1, dataset_size)])
    noise = np.random.default_rng(0).normal(
        0, 1, (dataset_size * X.size)
    ).reshape(dataset_size, *data_size)

    dataset = signal + noise + baselines

    return x, z, dataset


def dummy_wrapper(func):
    """A dummy wrapper to simulate using the _Algorithm._register wrapper function."""
    @wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)
    return inner


class DummyModule:
    """A dummy object to serve as a fake module."""

    @staticmethod
    def func(*args, data=None, x_data=None, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')


class DummyAlgorithm:
    """A dummy object to serve as a fake Algorithm subclass."""

    def __init__(self, *args, **kwargs):
        pass

    @dummy_wrapper
    def func(self, data=None, *args, **kwargs):
        """Dummy function."""
        raise NotImplementedError('need to set func')


def check_param_keys(expected_keys, output_keys):
    """
    Ensures the output keys within the parameter dictionary matched the expected keys.

    Parameters
    ----------
    expected_keys : Iterable[str, ...]
        An iterable of the expected keys within the parameter dictionary.
    output_keys : Iterable[str, ...]
        An iterable of the actual keys within the parameter dictionary.

    Raises
    ------
    AssertionError
        Raised if `expected_keys` and `output_keys` are not the same.

    """
    expected = set(expected_keys)
    output = set(output_keys)

    missed_keys = expected.difference(output)
    if missed_keys:
        raise AssertionError(f'key(s) missing from param dictionary: {missed_keys}')
    unchecked_keys = output.difference(expected)
    if unchecked_keys:
        raise AssertionError(f'unchecked key(s) in param dictionary output: {unchecked_keys}')


class BaseTester:
    """
    A base class for testing all algorithms.

    Ensure the functional and class-based algorithms are the same and that both do not
    modify the inputs. After that, only the class-based call is used to potentially save
    time from the setup.

    Attributes
    ----------
    kwargs : dict
        The keyword arguments that will be used as inputs for all default test cases.

    """

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'func'
    checked_keys = None
    required_kwargs = None
    required_repeated_kwargs = None
    two_d = False

    @classmethod
    def setup_class(cls):
        """Sets up the class for testing."""
        cls.x, cls.y = get_data()
        if cls.two_d:
            cls.y = np.vstack((cls.y, cls.y))
        func = getattr(cls.module, cls.func_name)
        cls.func = lambda self, *args, **kws: func(*args, **kws)
        cls.algorithm = cls.algorithm_base(cls.x, check_finite=False, assume_sorted=True)
        cls.class_func = getattr(cls.algorithm, cls.func_name)
        # kwargs are for fitting the data generated by get_data
        cls.kwargs = cls.required_kwargs if cls.required_kwargs is not None else {}
        # repeated_kwargs are for fitting the data generated by changing_dataset
        cls.repeated_kwargs = (
            cls.required_repeated_kwargs if cls.required_repeated_kwargs is not None else {}
        )
        cls.param_keys = cls.checked_keys if cls.checked_keys is not None else []

    @classmethod
    def teardown_class(cls):
        """
        Resets class attributes after testing.

        Probably not needed, but done anyway to catch changes in how pytest works.

        """
        cls.x = None
        cls.y = None
        cls.func = None
        cls.algorithm = None
        cls.class_func = None
        cls.kwargs = None
        cls.repeated_kwargs = None
        cls.param_keys = None

    def test_ensure_wrapped(self):
        """Ensures the class method was wrapped using _Algorithm._register to control inputs."""
        assert hasattr(self.class_func, '__wrapped__')

    @pytest.mark.parametrize('use_class', (True, False))
    def test_unchanged_data(self, use_class, **kwargs):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        x2, y2 = get_data()
        if self.two_d:
            y = np.vstack((y, y))
            y2 = np.vstack((y2, y2))
        x.setflags(write=False)
        y.setflags(write=False)

        try:
            if use_class:
                getattr(self.algorithm_base(x_data=x), self.func_name)(
                    data=y, **self.kwargs, **kwargs
                )
            else:
                self.func(data=y, x_data=x, **self.kwargs, **kwargs)
        except ValueError as e:  # from trying to assign value to read-only array
            raise AssertionError('method modified the input x- or y-data.') from e

        assert_array_equal(y2, y, err_msg='the y-data was changed by the algorithm')
        assert_array_equal(x2, x, err_msg='the x-data was changed by the algorithm')

    def test_repeated_fits(self):
        """Ensures the setup is properly reset when using class api."""
        first_output = self.class_func(data=self.y, **self.kwargs)
        second_output = self.class_func(data=self.y, **self.kwargs)

        assert_allclose(first_output[0], second_output[0], 1e-14)

    def test_functional_vs_class_output(self, **assertion_kwargs):
        """
        Ensures the functional and class-based functions perform the same.

        The functional api is tested by reinitializing the fitting object each call since that
        is what internally is done when using the functional interface.
        """
        x, dataset = changing_dataset(dataset_size=10)
        if self.two_d:
            dataset = np.array([np.vstack((data, data)) for data in dataset])

        class_method = getattr(
            self.algorithm_base(x, check_finite=False, assume_sorted=True), self.func_name
        )
        for data in dataset:
            class_output, class_params = class_method(data=data, **self.repeated_kwargs)
            func_output, func_params = getattr(
                self.algorithm_base(x, check_finite=False, assume_sorted=True), self.func_name
            )(data=data, **self.repeated_kwargs)

            assert_allclose(class_output, func_output, **assertion_kwargs)
            # TODO should later add back in the check for equivalence for the values within the
            # two parameter dictionaries
            check_param_keys(class_params.keys(), func_params.keys())

    def test_functional_vs_class_parameters(self):
        """
        Ensures the args and kwargs for functional and class-based functions are the same.

        Also ensures that both api have a `data` argument. The only difference between
        the two signatures should be that the functional api has an `x_data` keyword.

        """
        class_parameters = inspect.signature(self.class_func).parameters
        functional_parameters = inspect.signature(
            getattr(self.module, self.func_name)
        ).parameters

        # should be the same except that functional signature has x_data
        assert len(class_parameters) == len(functional_parameters) - 1
        assert 'data' in class_parameters
        assert 'x_data' in functional_parameters
        # also ensure 'data' is first argument for class api
        assert list(class_parameters.keys())[0] == 'data'
        # ensure key and values for all parameters match for both signatures
        for key in class_parameters:
            assert key in functional_parameters
            class_value = class_parameters[key].default
            functional_value = functional_parameters[key].default
            if isinstance(class_value, float) or isinstance(functional_value, float):
                assert_allclose(
                    class_value, functional_value, rtol=1e-14, atol=1e-14,
                    err_msg=f'Parameter mismatch for key "{key}"'
                )
            else:
                assert class_value == functional_value, f'Parameter mismatch for key "{key}"'

    def test_list_input(self, **assertion_kwargs):
        """Ensures that function works the same for both array and list inputs."""
        output_array = self.class_func(data=self.y, **self.kwargs)
        output_list = self.class_func(data=self.y.tolist(), **self.kwargs)

        assert_allclose(
            output_array[0], output_list[0],
            err_msg='algorithm output is different for arrays vs lists', **assertion_kwargs
        )
        for key in output_array[1]:
            assert key in output_list[1]

    def test_no_x(self, **assertion_kwargs):
        """
        Ensures that function output is the same when no x is input.

        Usually only valid for evenly spaced data, such as used for testing.

        """
        output_with = self.class_func(data=self.y, **self.kwargs)
        output_without = getattr(self.algorithm_base(), self.func_name)(
            data=self.y, **self.kwargs
        )

        assert_allclose(
            output_with[0], output_without[0],
            err_msg='algorithm output is different with no x-values',
            **assertion_kwargs
        )

    def test_output(self, additional_keys=None, optimizer_keys=None, **kwargs):
        """
        Ensures that the output has the desired format.

        Ensures that output has two elements, a numpy array and a param dictionary,
        and that the output baseline is the same shape as the input y-data.

        Parameters
        ----------
        additional_keys : Iterable(str, ...), optional
            Additional keys to check for in the output parameter dictionary. Default is None.
        optimizer_keys : Iterable(str, ...), optional
            Keys to check within the 'method_params' key for optimizer algorithms.
        **kwargs
            Additional keyword arguments to pass to the function.

        """
        output = self.class_func(data=self.y, **self.kwargs, **kwargs)

        assert len(output) == 2, 'algorithm output should have two items'
        assert isinstance(output[0], np.ndarray), 'output[0] should be a numpy ndarray'
        assert isinstance(output[1], dict), 'output[1] should be a dictionary'
        assert self.y.shape == output[0].shape, 'output[0] must have same shape as y-data'

        if additional_keys is not None:
            total_keys = list(self.param_keys) + list(additional_keys)
        else:
            total_keys = self.param_keys

        # ensure input keys are correct before checking output
        total_key_set = set(total_keys)
        assert len(total_key_set) == len(total_keys), 'repeated keys within param_keys'

        check_param_keys(total_key_set, output[1].keys())

        if optimizer_keys is not None:
            check_param_keys(optimizer_keys, output[1]['method_params'].keys())

    def test_x_ordering(self, assertion_kwargs=None, **kwargs):
        """Ensures arrays are correctly sorted within the function."""
        reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=False)

        regular_inputs_result = self.class_func(data=self.y, **self.kwargs, **kwargs)[0]
        reverse_inputs_result = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), **self.kwargs, **kwargs
        )[0]

        if assertion_kwargs is None:
            assertion_kwargs = {}
        if 'rtol' not in assertion_kwargs:
            assertion_kwargs['rtol'] = 1e-10

        assert_allclose(
            regular_inputs_result, self.reverse_array(reverse_inputs_result), **assertion_kwargs
        )

    def reverse_array(self, array):
        """Reverses the input along the last dimension."""
        return np.asarray(array)[..., ::-1]

    def test_threading(self, **kwargs):
        """
        Ensures the method produces the same output when using the same object within threading.

        While thread-safety cannot be guaranteed if the non-data arguments change between each
        call, ensuring thread-safety with all non-data arguments the same is about as good as can
        be done, and is a much more likely use-case.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the function.

        """
        x, dataset = changing_dataset()
        if self.two_d:
            dataset = np.array([np.vstack((data, data)) for data in dataset])

        fitter = self.algorithm_base(x, check_finite=False, assume_sorted=True)
        fitter_func = getattr(fitter, self.func_name)
        output_serial = np.empty_like(dataset)
        serial_params = []
        for i, data in enumerate(dataset):
            output_serial[i], method_params = fitter_func(data, **self.repeated_kwargs, **kwargs)
            serial_params.append(method_params)

        # create a new fitter to reinitialize for the threaded calculations
        fitter = self.algorithm_base(x, check_finite=False, assume_sorted=True)
        fitter_func = partial(getattr(fitter, self.func_name), **self.repeated_kwargs, **kwargs)
        output_threaded = np.empty_like(dataset)
        threaded_params = []
        try:
            with ThreadPoolExecutor() as pool:
                for i, (baseline, param) in enumerate(pool.map(fitter_func, dataset)):
                    output_threaded[i] = baseline
                    threaded_params.append(param)
        except np.linalg.LinAlgError as exc:
            # LinAlgError was raised by Whittaker methods when penalty matrix was overridden
            # between threads
            raise AssertionError('threading produced numerical errors') from exc
        else:
            assert_allclose(
                output_threaded, output_serial, rtol=1e-12, atol=1e-12,
                err_msg='Threaded results not equal to serial results'
            )
            for i, param_dict in enumerate(serial_params):
                # TODO should later compare parameter values as well
                check_param_keys(param_dict.keys(), threaded_params[i].keys())


class BasePolyTester(BaseTester):
    """
    A base class for testing polynomial algorithms.

    Checks that the polynomial coefficients are correctly returned and that they correspond
    to the polynomial used to create the baseline.

    """

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures the polynomial coefficients are output if `return_coef` is True."""
        if return_coef:
            additional_keys = ['coef']
        else:
            additional_keys = None
        super().test_output(additional_keys=additional_keys, return_coef=return_coef)

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self.class_func(data=self.y, **self.kwargs, return_coef=True)

        assert 'coef' in params

        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)
        assert_allclose(baseline, recreated_poly)


class InputWeightsMixin:
    """A mixin for BaseTester and BaseTester2D for ensuring input weights are correctly sorted."""

    weight_keys = ('weights',)

    def test_input_weights(self, assertion_kwargs=None, **kwargs):
        """Ensures input weights are correctly sorted within the function."""
        weights = np.random.default_rng(0).normal(0.8, 0.05, self.y.size)
        weights = np.clip(weights, 0, 1).astype(float, copy=False)

        if hasattr(self, 'two_d'):  # BaseTester
            reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=False)
        else:  # BaseTester2D
            reverse_fitter = self.algorithm_base(self.x[::-1], self.z[::-1], assume_sorted=False)
            weights = weights.reshape(self.y.shape)

        regular_output, regular_output_params = self.class_func(
            data=self.y, weights=weights, **self.kwargs, **kwargs
        )
        reverse_output, reverse_output_params = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), weights=self.reverse_array(weights),
            **self.kwargs, **kwargs
        )

        if assertion_kwargs is None:
            assertion_kwargs = {}
        if 'rtol' not in assertion_kwargs:
            assertion_kwargs['rtol'] = 1e-10
        if 'atol' not in assertion_kwargs:
            assertion_kwargs['atol'] = 1e-14

        for key in self.weight_keys:
            assert key in regular_output_params
            assert key in reverse_output_params

            assert_allclose(
                regular_output_params[key], self.reverse_array(reverse_output_params[key]),
                **assertion_kwargs
            )
        assert_allclose(
            regular_output, self.reverse_array(reverse_output), **assertion_kwargs
        )


class BaseTester2D:
    """
    A base class for testing all 2D algorithms.

    Attributes
    ----------
    kwargs : dict
        The keyword arguments that will be used as inputs for all default test cases.

    """

    module = DummyModule
    algorithm_base = DummyAlgorithm
    func_name = 'func'
    checked_keys = None
    required_kwargs = None
    required_repeated_kwargs = None
    three_d = False

    @classmethod
    def setup_class(cls):
        """Sets up the class for testing."""
        cls.x, cls.z, cls.y = get_data2d()
        if cls.three_d:
            cls.y = np.array((cls.y, cls.y))
        cls.algorithm = cls.algorithm_base(cls.x, cls.z, check_finite=False, assume_sorted=True)
        cls.class_func = getattr(cls.algorithm, cls.func_name)
        # kwargs are for fitting the data generated by get_data
        cls.kwargs = cls.required_kwargs if cls.required_kwargs is not None else {}
        # repeated_kwargs are for fitting the data generated by changing_dataset
        cls.repeated_kwargs = (
            cls.required_repeated_kwargs if cls.required_repeated_kwargs is not None else {}
        )
        cls.param_keys = cls.checked_keys if cls.checked_keys is not None else []

    @classmethod
    def teardown_class(cls):
        """
        Resets class attributes after testing.

        Probably not needed, but done anyway to catch changes in how pytest works.

        """
        cls.x = None
        cls.z = None
        cls.y = None
        cls.algorithm = None
        cls.class_func = None
        cls.kwargs = None
        cls.repeated_kwargs = None
        cls.param_keys = None

    def test_ensure_wrapped(self):
        """Ensures the class method was wrapped using _Algorithm._register to control inputs."""
        assert hasattr(self.class_func, '__wrapped__')

    @pytest.mark.parametrize('new_instance', (True, False))
    def test_unchanged_data(self, new_instance, **kwargs):
        """Ensures that input data is unchanged by the function."""
        x, z, y = get_data2d()
        x2, z2, y2 = get_data2d()
        if self.three_d:
            y = np.array((y, y))
            y2 = np.array((y2, y2))

        y.setflags(write=False)
        try:
            if new_instance:
                x.setflags(write=False)
                z.setflags(write=False)
                getattr(self.algorithm_base(x_data=x, z_data=z), self.func_name)(
                    data=y, **self.kwargs, **kwargs
                )
                compared_x = x
                compared_z = z
            else:
                self.x.setflags(write=False)
                self.z.setflags(write=False)
                self.class_func(data=y, **self.kwargs, **kwargs)
                compared_x = self.x
                compared_z = self.z
        except ValueError as e:  # from trying to assign value to read-only array
            raise AssertionError('method modified the input x- or y-data.') from e
        finally:
            if not new_instance:
                self.x.setflags(write=True)
                self.z.setflags(write=True)

        assert_array_equal(y2, y, err_msg='the y-data was changed by the algorithm')
        assert_array_equal(x2, compared_x, err_msg='the x-data was changed by the algorithm')
        assert_array_equal(z2, compared_z, err_msg='the z-data was changed by the algorithm')

    def test_repeated_fits(self):
        """Ensures the setup is properly reset when using class api."""
        first_output = self.class_func(data=self.y, **self.kwargs)
        second_output = self.class_func(data=self.y, **self.kwargs)

        assert_allclose(first_output[0], second_output[0], 1e-14)

    def test_list_input(self, **assertion_kwargs):
        """Ensures that function works the same for both array and list inputs."""
        output_array = self.class_func(data=self.y, **self.kwargs)
        output_list = self.class_func(data=self.y.tolist(), **self.kwargs)

        assert_allclose(
            output_array[0], output_list[0],
            err_msg='algorithm output is different for arrays vs lists', **assertion_kwargs
        )
        for key in output_array[1]:
            assert key in output_list[1]

    @pytest.mark.parametrize('has_x', (True, False))
    @pytest.mark.parametrize('has_z', (True, False))
    def test_no_xz(self, has_x, has_z, **assertion_kwargs):
        """
        Ensures that function output is the same when no x and/or z is input.

        Usually only valid for evenly spaced data, such as used for testing.

        """
        if has_x and has_z:
            return  # the one test case that would not produce any difference so skip to save time
        output_with = self.class_func(data=self.y, **self.kwargs)

        input_x = self.x if has_x else None
        input_z = self.z if has_z else None
        output_without = getattr(
            self.algorithm_base(x_data=input_x, z_data=input_z), self.func_name
        )(data=self.y, **self.kwargs)

        assert_allclose(
            output_with[0], output_without[0],
            err_msg='algorithm output is different with no x-values and/or z-values',
            **assertion_kwargs
        )

    def test_output(self, additional_keys=None, optimizer_keys=None, **kwargs):
        """
        Ensures that the output has the desired format.

        Ensures that output has two elements, a numpy array and a param dictionary,
        and that the output baseline is the same shape as the input y-data.

        Parameters
        ----------
        additional_keys : Iterable(str, ...), optional
            Additional keys to check for in the output parameter dictionary. Default is None.
        optimizer_keys : Iterable(str, ...), optional
            Keys to check within the 'method_params' key for optimizer algorithms.
        **kwargs
            Additional keyword arguments to pass to the function.

        """
        output = self.class_func(data=self.y, **self.kwargs, **kwargs)

        assert len(output) == 2, 'algorithm output should have two items'
        assert isinstance(output[0], np.ndarray), 'output[0] should be a numpy ndarray'
        assert isinstance(output[1], dict), 'output[1] should be a dictionary'
        assert self.y.shape == output[0].shape, 'output[0] must have same shape as y-data'

        if additional_keys is not None:
            total_keys = list(self.param_keys) + list(additional_keys)
        else:
            total_keys = self.param_keys

        # ensure input keys are correct before checking output
        total_key_set = set(total_keys)
        assert len(total_key_set) == len(total_keys), 'repeated keys within param_keys'

        check_param_keys(total_key_set, output[1].keys())

        if optimizer_keys is not None:
            check_param_keys(optimizer_keys, output[1]['method_params'].keys())

    def test_xz_ordering(self, assertion_kwargs=None, **kwargs):
        """Ensures arrays are correctly sorted within the function."""
        reverse_fitter = self.algorithm_base(self.x[::-1], self.z[::-1], assume_sorted=False)

        regular_inputs_result = self.class_func(data=self.y, **self.kwargs, **kwargs)[0]
        reverse_inputs_result = getattr(reverse_fitter, self.func_name)(
            data=self.reverse_array(self.y), **self.kwargs, **kwargs
        )[0]

        if assertion_kwargs is None:
            assertion_kwargs = {}
        if 'rtol' not in assertion_kwargs:
            assertion_kwargs['rtol'] = 1e-10

        assert_allclose(
            regular_inputs_result, self.reverse_array(reverse_inputs_result), **assertion_kwargs
        )

    def reverse_array(self, array):
        """Reverses the input along the last two dimensions."""
        return np.asarray(array)[..., ::-1, ::-1]

    def test_threading(self, **kwargs):
        """
        Ensures the method produces the same output when using the same object within threading.

        While thread-safety cannot be guaranteed if the non-data arguments change between each
        call, ensuring thread-safety with all non-data arguments the same is about as good as can
        be done, and is a much more likely use-case.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the function.

        """
        # if dataset size is too small, won't catch some threading issues; however, using
        # dataset_size=50 catches the slowest performers of a family of algorithms (eg. those
        # that use Whittaker smoothing), which allows fixing the underlying problem without
        # causing the entire test suite to take a very long time to run
        x, z, dataset = changing_dataset2d((40, 33), dataset_size=50)
        if self.three_d:
            dataset = np.array([np.array((data, data)) for data in dataset])

        fitter = self.algorithm_base(x, z, check_finite=False, assume_sorted=True)
        fitter_func = getattr(fitter, self.func_name)
        output_serial = np.empty_like(dataset)
        serial_params = []
        for i, data in enumerate(dataset):
            output_serial[i], method_params = fitter_func(data, **self.repeated_kwargs, **kwargs)
            serial_params.append(method_params)

        # create a new fitter to reinitialize for the threaded calculations
        fitter = self.algorithm_base(x, z, check_finite=False, assume_sorted=True)
        fitter_func = partial(getattr(fitter, self.func_name), **self.repeated_kwargs, **kwargs)
        output_threaded = np.empty_like(dataset)
        threaded_params = []
        try:
            with ThreadPoolExecutor() as pool:
                for i, (baseline, param) in enumerate(pool.map(fitter_func, dataset)):
                    output_threaded[i] = baseline
                    threaded_params.append(param)
        except np.linalg.LinAlgError as exc:
            # LinAlgError was raised by Whittaker methods when penalty matrix was overridden
            # between threads
            raise AssertionError('threading produced numerical errors') from exc
        else:
            assert_allclose(
                output_threaded, output_serial, rtol=1e-12, atol=1e-12,
                err_msg='Threaded results not equal to serial results'
            )
            for i, param_dict in enumerate(serial_params):
                # TODO should later compare parameter values as well
                check_param_keys(param_dict.keys(), threaded_params[i].keys())


class BasePolyTester2D(BaseTester2D):
    """
    A base class for testing 2D polynomial algorithms.

    Checks that the polynomial coefficients are correctly returned and that they correspond
    to the polynomial used to create the baseline.

    """

    @pytest.mark.parametrize('return_coef', (True, False))
    def test_output(self, return_coef):
        """Ensures the polynomial coefficients are output if `return_coef` is True."""
        if return_coef:
            additional_keys = ['coef']
        else:
            additional_keys = None
        super().test_output(additional_keys=additional_keys, return_coef=return_coef)

    @pytest.mark.parametrize('poly_order', (1, 2, [2, 3]))
    def test_output_coefs(self, poly_order):
        """
        Ensures the output coefficients can correctly reproduce the baseline.

        Checks both the manual way using the Vandermonde and directly using numpy's polyval2d.
        """
        baseline, params = self.class_func(
            data=self.y, poly_order=poly_order, **self.kwargs, return_coef=True
        )

        assert 'coef' in params

        if isinstance(poly_order, int):
            x_order = poly_order
            z_order = poly_order
        else:
            x_order, z_order = poly_order

        X, Z = np.meshgrid(self.x, self.z, indexing='ij')
        vander = np.polynomial.polynomial.polyvander2d(
            X, Z, (x_order, z_order)
        ).reshape((-1, (x_order + 1) * (z_order + 1)))

        recreated_poly = (vander @ params['coef'].flatten()).reshape(self.y.shape)
        assert_allclose(recreated_poly, baseline, rtol=1e-10, atol=1e-12)

        numpy_poly = np.polynomial.polynomial.polyval2d(X, Z, params['coef'])
        assert_allclose(numpy_poly, baseline, rtol=1e-10, atol=1e-12)
