# -*- coding: utf-8 -*-
"""High level functions for making better use of baseline algorithms.

Functions in this module make use of other baseline algorithms in
pybaselines to provide better results or optimize parameters.

Created on March 3, 2021
@author: Donald Erb

"""

from math import ceil
import warnings

import numpy as np

from . import classification, morphological, polynomial, spline, whittaker
from ._algorithm_setup import _setup_polynomial, _whittaker_smooth, _yx_arrays
from .utils import _check_scalar, _get_edges, _inverted_sort, gaussian


def _get_function(method, modules):
    """
    Tries to retrieve the indicated function from a list of modules.

    Parameters
    ----------
    method : str
        The string name of the desired function. Case does not matter.
    modules : Sequence
        A sequence of modules in which to look for the method.

    Returns
    -------
    func : Callable
        The corresponding function.
    func_module : str
        The module that `func` belongs to.

    Raises
    ------
    AttributeError
        Raised if no matching function is found within the modules.

    """
    function_string = method.lower()
    for module in modules:
        if hasattr(module, function_string):
            func = getattr(module, function_string)
            func_module = module.__name__.split('.')[-1]
            break
    else:  # in case no break
        raise AttributeError(f'unknown method {method}')

    return func, func_module


def collab_pls(data, average_dataset=True, method='asls', **method_kwargs):
    """
    Collaborative Penalized Least Squares (collab-PLS).

    Averages the data or the fit weights for an entire dataset to get more
    optimal results. Uses any Whittaker-smoothing-based or weighted spline algorithm.

    Parameters
    ----------
    data : array-like, shape (M, N)
        An array with shape (M, N) where M is the number of entries in
        the dataset and N is the number of data points in each entry.
    average_dataset : bool, optional
        If True (default) will average the dataset before fitting to get the
        weighting. If False, will fit each individual entry in the dataset and
        then average the weights to get the weighting for the dataset.
    method : str, optional
        A string indicating the Whittaker-smoothing-based or weighted spline method to
        use for fitting the baseline. Default is 'asls'.
    **method_kwargs
        Keyword arguments to pass to the selected `method` function.

    Returns
    -------
    baselines : np.ndarray, shape (M, N)
        An array of all of the baselines.
    params : dict
        A dictionary with the following items:

        * 'average_weights': numpy.ndarray, shape (N,)
            The weight array used to fit all of the baselines.

        Additional items depend on the output of the selected method. Every
        other key will have a list of values, with each item corresponding to a
        fit.

    References
    ----------
    Chen, L., et al. Collaborative Penalized Least Squares for Background
    Correction of Multiple Raman Spectra. Journal of Analytical Methods
    in Chemistry, 2018, 2018.

    """
    fit_func = _get_function(method, (whittaker, morphological, classification, spline))[0]
    dataset = np.asarray(data)
    if dataset.ndim < 2:
        raise ValueError((
            'the input data must have a shape of (number of measurements, number of points), '
            f'but instead has a shape of {dataset.shape}'
        ))
    if average_dataset:
        _, fit_params = fit_func(np.mean(dataset.T, 1), **method_kwargs)
        method_kwargs['weights'] = fit_params['weights']
    else:
        weights = np.empty_like(dataset)
        for i, entry in enumerate(dataset):
            _, fit_params = fit_func(entry, **method_kwargs)
            weights[i] = fit_params['weights']
        method_kwargs['weights'] = np.mean(weights.T, 1)

    method_kwargs['tol'] = np.inf
    baselines = np.empty(dataset.shape)
    params = {'average_weights': method_kwargs['weights']}
    method = method.lower()
    if method == 'fabc':
        # have to handle differently since weights for fabc is the mask for
        # classification rather than weights for fitting
        fit_func = _whittaker_smooth
        for key in list(method_kwargs.keys()):
            if key not in {'weights', 'lam', 'diff_order'}:
                method_kwargs.pop(key)

    for i, entry in enumerate(dataset):
        baselines[i], param = fit_func(entry, **method_kwargs)
        if method == 'fabc':
            param = {'weights': param}
        for key, value in param.items():
            if key in params:
                params[key].append(value)
            else:
                params[key] = [value]

    return baselines, params


def optimize_extended_range(data, x_data=None, method='asls', side='both', width_scale=0.1,
                            height_scale=1., sigma_scale=1. / 12., min_value=2, max_value=8,
                            step=1, pad_kwargs=None, method_kwargs=None, **kwargs):
    """
    Extends data and finds the best parameter value for the given baseline method.

    Adds additional data to the left and/or right of the input data, and then iterates
    through parameter values to find the best fit. Useful for calculating the optimum
    `lam` or `poly_order` value required to optimize other algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    method : str, optional
        A string indicating the Whittaker-smoothing-based, polynomial, or spline method
        to use for fitting the baseline. Default is 'asls'.
    side : {'both', 'left', 'right'}, optional
        The side of the measured data to extend. Default is 'both'.
    width_scale : float, optional
        The number of data points added to each side is `width_scale` * N. Default
        is 0.1.
    height_scale : float, optional
        The height of the added Gaussian peak(s) is calculated as
        `height_scale` * max(`data`). Default is 1.
    sigma_scale : float, optional
        The sigma value for the added Gaussian peak(s) is calculated as
        `sigma_scale` * `width_scale` * N. Default is 1/12, which will make
        the Gaussian span +- 6 sigma, making its total width about half of the
        added length.
    min_value : int or float, optional
        The minimum value for the `lam` or `poly_order` value to use with the
        indicated method. If using a polynomial method, `min_value` must be an
        integer. If using a Whittaker-smoothing-based method, `min_value` should
        be the exponent to raise to the power of 10 (eg. a `min_value` value of 2
        designates a `lam` value of 10**2).
        Default is 2.
    max_value : int or float, optional
        The maximum value for the `lam` or `poly_order` value to use with the
        indicated method. If using a polynomial method, `max_value` must be an
        integer. If using a Whittaker-smoothing-based method, `max_value` should
        be the exponent to raise to the power of 10 (eg. a `max_value` value of 3
        designates a `lam` value of 10**3).
        Default is 8.
    step : int or float, optional
        The step size for iterating the parameter value from `min_value` to `max_value`.
        If using a polynomial method, `step` must be an integer.
    pad_kwargs : dict, optional
        A dictionary of options to pass to :func:`.pad_edges` for padding
        the edges of the data when adding the extended left and/or right sections.
        Default is None, which will use an empty dictionary.
    method_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the selected `method` function.
        Default is None, which will use an empty dictionary.
    **kwargs
        Deprecated in version 0.7.0 and will be removed in version 0.9.0. Pass any
        keyword arguments for the fitting function in the `method_kwargs` dictionary.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline calculated with the optimum parameter.
    method_params : dict
        A dictionary with the following items:

        * 'optimal_parameter': int or float
            The `lam` or `poly_order` value that produced the lowest
            root-mean-squared-error.
        * 'min_rmse': float
            The minimum root-mean-squared-error obtained when using
            the optimal parameter.

        Additional items depend on the output of the selected method.

    Raises
    ------
    ValueError
        Raised if `side` is not 'left', 'right', or 'both'.
    TypeError
        Raised if using a polynomial method and `min_value`, `max_value`, or
        `step` is not an integer.
    ValueError
        Raised if using a Whittaker-smoothing-based method and `min_value`,
        `max_value`, or `step` is greater than 100.

    Notes
    -----
    Based on the extended range penalized least squares (erPLS) method from [1]_.
    The method proposed by [1]_ was for optimizing lambda only for the aspls
    method by extending only the right side of the spectrum. The method was
    modified by allowing extending either side following [2]_, and for optimizing
    lambda or the polynomial degree for all of the affected algorithms in
    pybaselines.

    References
    ----------
    .. [1] Zhang, F., et al. An Automatic Baseline Correction Method Based on
           the Penalized Least Squares Method. Sensors, 2020, 20(7), 2015.
    .. [2] Krishna, H., et al. Range-independent background subtraction algorithm
           for recovery of Raman spectra of biological tissue. Journal of Raman
           Spectroscopy. 2012, 43(12), 1884-1894.

    """
    method = method.lower()
    side = side.lower()
    if side not in ('left', 'right', 'both'):
        raise ValueError('side must be "left", "right", or "both"')

    fit_func, func_module = _get_function(
        method, (whittaker, polynomial, morphological, spline, classification)
    )
    if func_module == 'polynomial' or method in ('dietrich', 'cwt_br'):
        if any(not isinstance(val, int) for val in (min_value, max_value, step)):
            raise TypeError((
                'min_value, max_value, and step must all be integers when'
                ' using a polynomial method'
            ))
        param_name = 'poly_order'
    else:
        if any(val > 100 for val in (min_value, max_value, step)):
            raise ValueError((
                'min_value, max_value, and step should be the power of 10 to use '
                '(eg. min_value=2 denotes 10**2), not the actual "lam" value, and '
                'thus should not be greater than 100'
            ))
        param_name = 'lam'

    y, x = _yx_arrays(data, x_data)
    added_window = int(x.shape[0] * width_scale)
    method_kwargs = method_kwargs.copy() if method_kwargs is not None else {}
    if kwargs:  # TODO remove in version 0.9
        warnings.warn(
            ('Passing additional keyword arguments directly to optimize_extended_range is '
             'deprecated and will be removed in version 0.9.0. Place all keyword arguments '
             'into the method_kwargs dictionary instead.'),
            DeprecationWarning, stacklevel=2
        )
        method_kwargs.update(kwargs)
    sort_x = x_data is not None
    if sort_x:
        sort_order = np.argsort(x, kind='mergesort')  # to ensure x is increasing
        x = x[sort_order]
        y = y[sort_order]
        if 'weights' in method_kwargs:
            # have to adjust weight length to accomodate the added sections; set weights
            # to 1 to ensure the added sections are fit
            method_kwargs['weights'] = np.pad(
                method_kwargs['weights'][sort_order],
                [0 if side == 'right' else added_window, 0 if side == 'left' else added_window],
                'constant', constant_values=1
            )
    max_x = x.max()
    min_x = x.min()
    x_range = max_x - min_x
    known_background = np.array([])
    fit_x_data = x
    fit_data = y
    lower_bound = upper_bound = 0

    if pad_kwargs is None:
        pad_kwargs = {}
    added_left, added_right = _get_edges(y, added_window, **pad_kwargs)
    added_gaussian = gaussian(
        np.linspace(-added_window / 2, added_window / 2, added_window),
        height_scale * abs(y.max()), 0, added_window * sigma_scale
    )
    if side in ('right', 'both'):
        added_x = np.linspace(
            max_x, max_x + x_range * (width_scale / 2), added_window + 1
        )[1:]
        fit_x_data = np.concatenate((fit_x_data, added_x))
        fit_data = np.concatenate((fit_data, added_gaussian + added_right))
        known_background = added_right
        upper_bound += added_window
    if side in ('left', 'both'):
        added_x = np.linspace(
            min_x - x_range * (width_scale / 2), min_x, added_window + 1
        )[:-1]
        fit_x_data = np.concatenate((added_x, fit_x_data))
        fit_data = np.concatenate((added_gaussian + added_left, fit_data))
        known_background = np.concatenate((known_background, added_left))
        lower_bound += added_window

    if func_module == 'polynomial' or method in ('iasls', 'dietrich', 'cwt_br'):
        method_kwargs['x_data'] = fit_x_data

    added_len = 2 * added_window if side == 'both' else added_window
    upper_idx = fit_data.shape[0] - upper_bound
    min_sum_squares = np.inf
    best_val = None
    # TODO maybe switch to linspace since arange is inconsistent when using floats
    for var in np.arange(min_value, max_value + step, step):
        if param_name == 'lam':
            method_kwargs[param_name] = 10**var
        else:
            method_kwargs[param_name] = var
        fit_baseline, fit_params = fit_func(fit_data, **method_kwargs)
        # TODO change the known baseline so that np.roll does not have to be
        # calculated each time, since it requires additional time
        residual = (
            known_background - np.roll(fit_baseline, upper_bound)[:added_len]
        )
        # just calculate the sum of squares to reduce time from using sqrt for rmse
        sum_squares = residual.dot(residual)
        if sum_squares < min_sum_squares:
            baseline = fit_baseline[lower_bound:upper_idx]
            params = fit_params
            best_val = var
            min_sum_squares = sum_squares

    params.update(
        {'optimal_parameter': best_val, 'min_rmse': np.sqrt(min_sum_squares / added_len)}
    )

    if sort_x:
        baseline = baseline[_inverted_sort(sort_order)]

    return baseline, params


def _determine_polyorders(y, x, poly_order, weights, fit_function, **fit_kwargs):
    """
    Selects the appropriate polynomial orders based on the baseline-to-signal ratio.

    Parameters
    ----------
    y : numpy.ndarray
        The array of y-values.
    x : numpy.ndarray
         The array of x-values.
    poly_order : int
        The polynomial order for fitting.
    weights : numpy.ndarray
        The weight array for fitting.
    fit_function : Callable
        The function to use for the polynomial fit.
    **fit_kwargs
        Additional keyword arguments to pass to `fit_function`.

    Returns
    -------
    orders : tuple(int, int)
        The two polynomial orders to use based on the baseline to signal
        ratio according to the reference.

    References
    ----------
    Cao, A., et al. A robust method for automated background subtraction
    of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38, 1199-1205.

    """
    baseline = fit_function(y, x, poly_order, weights=weights, **fit_kwargs)[0]
    signal = y - baseline
    baseline_to_signal = (baseline.max() - baseline.min()) / (signal.max() - signal.min())
    # Table 2 in reference
    if baseline_to_signal < 0.2:
        orders = (1, 2)
    elif baseline_to_signal < 0.75:
        orders = (2, 3)
    elif baseline_to_signal < 8.5:
        orders = (3, 4)
    elif baseline_to_signal < 55:
        orders = (4, 5)
    elif baseline_to_signal < 240:
        orders = (5, 6)
    elif baseline_to_signal < 517:
        orders = (6, 7)
    else:
        orders = (6, 8)  # not a typo, use 6 and 8 rather than 7 and 8

    return orders


def adaptive_minmax(data, x_data=None, poly_order=None, method='modpoly',
                    weights=None, constrained_fraction=0.01, constrained_weight=1e5,
                    estimation_poly_order=2, **method_kwargs):
    """
    Fits polynomials of different orders and uses the maximum values as the baseline.

    Each polynomial order fit is done both unconstrained and constrained at the
    endpoints.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, Sequence(int, int) or None, optional
        The two polynomial orders to use for fitting. If a single integer is given,
        then will use the input value and one plus the input value. Default is None,
        which will do a preliminary fit using a polynomial of order `estimation_poly_order`
        and then select the appropriate polynomial orders according to [3]_.
    method : {'modpoly', 'imodpoly'}, optional
        The method to use for fitting each polynomial. Default is 'modpoly'.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    constrained_fraction : float or Sequence(float, float), optional
        The fraction of points at the left and right edges to use for the
        constrained fit. Default is 0.01. If `constrained_fraction` is a sequence,
        the first item is the fraction for the left edge and the second is the
        fraction for the right edge.
    constrained_weight : float or Sequence(float, float), optional
        The weighting to give to the endpoints. Higher values ensure that the
        end points are fit, but can cause large fluctuations in the other sections
        of the polynomial. Default is 1e5. If `constrained_weight` is a sequence,
        the first item is the weight for the left edge and the second is the
        weight for the right edge.
    estimation_poly_order : int, optional
        The polynomial order used for estimating the baseline-to-signal ratio
        to select the appropriate polynomial orders if `poly_order` is None.
        Default is 2.
    **method_kwargs
        Additional keyword arguments to pass to :func:`.modpoly` or
        :func:`.imodpoly`. These include `tol`, `max_iter`, `use_original`,
        `mask_initial_peaks`, and `num_std`.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'constrained_weights': numpy.ndarray, shape (N,)
            The weight array used for the endpoint-constrained fits.
        * 'poly_order': numpy.ndarray, shape (2,)
            An array of the two polynomial orders used for the fitting.

    References
    ----------
    .. [3] Cao, A., et al. A robust method for automated background subtraction
           of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
           1199-1205.

    """
    fit_func = {'modpoly': polynomial.modpoly, 'imodpoly': polynomial.imodpoly}[method.lower()]
    y, x, weight_array, _ = _setup_polynomial(data, x_data, weights)
    if poly_order is None:
        poly_orders = _determine_polyorders(
            y, x, estimation_poly_order, weight_array, fit_func, **method_kwargs
        )
    else:
        poly_orders, scalar_poly_order = _check_scalar(poly_order, 2, True, dtype=int)
        if scalar_poly_order:
            poly_orders[1] += 1  # add 1 since they are initially equal if scalar input

    # use high weighting rather than Lagrange multipliers to constrain the points
    # to better work with noisy data
    weightings = _check_scalar(constrained_weight, 2, True)[0]
    constrained_fractions = _check_scalar(constrained_fraction, 2, True)[0]
    if np.any(constrained_fractions < 0) or np.any(constrained_fractions > 1):
        raise ValueError('constrained_fraction must be between 0 and 1')
    len_y = len(y)
    constrained_weights = weight_array.copy()
    constrained_weights[:ceil(len_y * constrained_fractions[0])] = weightings[0]
    constrained_weights[len_y - ceil(len_y * constrained_fractions[1]):] = weightings[1]

    # TODO should make parameters available; a list with an item for each fit like collab_pls
    baselines = np.empty((4, y.shape[0]))
    baselines[0] = fit_func(y, x, poly_orders[0], weights=weight_array, **method_kwargs)[0]
    baselines[1] = fit_func(y, x, poly_orders[0], weights=constrained_weights, **method_kwargs)[0]
    baselines[2] = fit_func(y, x, poly_orders[1], weights=weight_array, **method_kwargs)[0]
    baselines[3] = fit_func(y, x, poly_orders[1], weights=constrained_weights, **method_kwargs)[0]

    # TODO should the coefficients also be made available? Would need to get them from
    # each of the fits
    params = {
        'weights': weight_array, 'constrained_weights': constrained_weights,
        'poly_order': poly_orders
    }

    return np.maximum.reduce(baselines), params
