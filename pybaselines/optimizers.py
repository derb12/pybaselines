# -*- coding: utf-8 -*-
"""High level functions for making better use of baseline algorithms.

Functions in this module make use of other baseline algorithms in
pybaselines to provide better results or optimize parameters.

Optimizers
    1) collab_pls (Collaborative Penalized Least Squares)
    2) optimize_extended_range
    3) adaptive_minmax (Adaptive MinMax)

Created on March 3, 2021
@author: Donald Erb

"""

from math import ceil

import numpy as np

from . import morphological, polynomial, whittaker
from .utils import gaussian
from ._algorithm_setup import _setup_polynomial, _yx_arrays


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

    Raises
    ------
    AttributeError
        Raised if no matching function is found within the modules.

    """
    function_string = method.lower()
    func = None
    for module in modules:
        try:
            func = getattr(module, function_string)
        except AttributeError:
            pass

    if func is None:
        raise AttributeError('unknown method')

    return func


def collab_pls(data, average_dataset=True, method='asls', **method_kwargs):
    """
    Collaborative Penalized Least Squares (collab-PLS).

    Averages the data or the fit weights for an entire dataset to get more
    optimal results. Uses any Whittaker-smoothing-based algorithm.

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
        A string indicating the Whittaker-smoothing-based method to use for
        fitting the baseline. Default is 'asls'.
    **method_kwargs
        Keyword arguments to pass to the selected `method` function.

    Returns
    -------
    np.ndarray, shape (M, N)
        An array of all of the baselines.

    References
    ----------
    Chen, L., et al. Collaborative Penalized Least Squares for Background
    Correction of Multiple Raman Spectra. Journal of Analytical Methods
    in Chemistry, 2018, 2018.

    """
    fit_func = _get_function(method, (whittaker, morphological))
    dataset = np.asarray(data)
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
    baselines = []
    for entry in dataset:
        baselines.append(fit_func(entry, **method_kwargs)[0])

    return np.vstack(baselines), {'weights': method_kwargs['weights']}


def _iter_solve(func, fit_data, known_background, lower_bound, upper_bound, variable,
                min_value, max_value, step=1, allowed_misses=1, **func_kwargs):
    """Iterates through possible values to find the one with lowest root-mean-square-error."""
    min_rmse = np.inf
    misses = 0
    for var in np.arange(min_value, max_value, step):
        if variable == 'lam':
            func_kwargs[variable] = 10**var
        else:
            func_kwargs[variable] = var
        baseline, other_params = func(fit_data, **func_kwargs)
        #TODO change the known baseline so that np.roll does not have to be
        # calculated each time, since it requires additional time
        rmse = np.sqrt(np.mean(
            (known_background - np.roll(baseline, upper_bound)[:upper_bound + lower_bound])**2
        ))
        if rmse < min_rmse:
            best_baseline = baseline[lower_bound:baseline.shape[0] - upper_bound]
            min_var = var
            misses = 0
            min_rmse = rmse
        else:
            misses += 1
            if misses > allowed_misses:
                break

    return best_baseline, min_var, other_params


def optimize_extended_range(data, x_data=None, method='asls', side='both', **method_kwargs):
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
        A string indicating the Whittaker-smoothing-based or polynomial method
        to use for fitting the baseline. Default is 'aspls'.
    side : {'both', 'left', 'right'}, optional
        The side of the measured data to extend. Default is 'both'.
    **method_kwargs
        Keyword arguments to pass to the selected `method` function.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline calculated with the optimum parameter.
    method_params : dict
        A dictionary with the following items:

        * 'optimal_parameter': int or float
            The `lam` or `poly_order` value that produced the lowest
            root-mean-squared-error.

        Additional items depend on the output of the selected method.

    Raises
    ------
    ValueError
        Raised if `side` is not 'left', 'right', or 'both'.

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
    if side.lower() not in ('left', 'right', 'both'):
        raise ValueError('side must be "left", "right", or "both"')

    fit_func = _get_function(method, (whittaker, polynomial, morphological))
    y, x = _yx_arrays(data, x_data)
    sort_order = np.argsort(x)  # to ensure x is increasing
    x = x[sort_order]
    y = y[sort_order]
    max_x = np.nanmax(x)
    min_x = np.nanmin(x)
    x_range = max_x - min_x
    known_background = np.array([])
    fit_x_data = x
    fit_data = y
    lower_bound = upper_bound = 0

    W = x.shape[0] // 10
    #TODO use utils._get_edges

    if side.lower() in ('right', 'both'):
        added_x = np.linspace(max_x, max_x + x_range / 5, W)
        line = np.polynomial.Polynomial.fit(
            x[x > max_x - x_range / 20], y[x > max_x - x_range / 20], 1
        )(added_x)
        gaus = gaussian(added_x, np.nanmax(y), np.median(added_x), x_range / 50)
        fit_x_data = np.hstack((fit_x_data, added_x))
        fit_data = np.hstack((fit_data, gaus + line))
        known_background = line
        upper_bound += W
    if side.lower() in ('left', 'both'):
        added_x = np.linspace(min_x - x_range / 5, min_x, W)
        line = np.polynomial.Polynomial.fit(
            x[x < min_x + x_range / 20], y[x < min_x + x_range / 20], 1
        )(added_x)
        gaus = gaussian(added_x, np.nanmax(y), np.median(added_x), x_range / 50)
        fit_x_data = np.hstack((added_x, fit_x_data))
        fit_data = np.hstack((gaus + line, fit_data))
        known_background = np.hstack((known_background, line))
        lower_bound += W

    if method.lower() in ('iasls', 'modpoly', 'imodpoly', 'poly', 'penalized_poly'):
        method_kwargs['x_data'] = fit_x_data

    if 'poly' in method.lower():
        baseline, best_val, method_params = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'poly_order',
            0, 20, 1, 4, **method_kwargs
        )
    else:
        _, best_val, _ = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'lam',
            1, 50, 1, 2, **method_kwargs
        )
        baseline, best_val, method_params = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'lam',
            best_val - 0.9, best_val + 1.1, 0.1, 2, **method_kwargs
        )
    method_params['optimal_parameter'] = best_val

    return (
        baseline[[val[0] for val in sorted(enumerate(sort_order), key=lambda v: v[1])]],
        method_params
    )


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
    basline_to_signal = (max(baseline) - min(baseline)) / (max(signal) - min(signal))
    # Table 2 in reference
    if basline_to_signal < 0.2:
        orders = (1, 2)
    elif basline_to_signal < 0.75:
        orders = (2, 3)
    elif basline_to_signal < 8.5:
        orders = (3, 4)
    elif basline_to_signal < 55:
        orders = (4, 5)
    elif basline_to_signal < 240:
        orders = (5, 6)
    elif basline_to_signal < 517:
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
    constrained_fraction : float, optional
        The fraction of points at the left and right edges to use for the
        constrained fit. Default is 0.01.
    constrained_weight : float, optional
        The weighting to give to the endpoints. Higher values ensure that the
        end points are fit, but can cause large fluctuations in the other sections
        of the polynomial. Default is 1e5.
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
        * 'poly_order': tuple(int, int)
            A tuple of the two polynomial orders used for the fitting.

    References
    ----------
    .. [3] Cao, A., et al. A robust method for automated background subtraction
           of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
           1199-1205.

    """
    fit_func = {'modpoly': polynomial.modpoly, 'imodpoly': polynomial.imodpoly}[method.lower()]
    y, x, weight_array, _ = _setup_polynomial(data, x_data, weights)
    constrained_range = max(1, ceil(y.shape[0] * constrained_fraction))

    if isinstance(poly_order, int):
        poly_orders = (poly_order, poly_order + 1)
    elif poly_order is not None:
        if len(poly_order) == 1:
            poly_orders = (poly_order[0], poly_order[0] + 1)
        else:
            poly_orders = (poly_order[0], poly_order[1])
    else:
        poly_orders = _determine_polyorders(
            y, x, estimation_poly_order, weight_array, fit_func, **method_kwargs
        )

    # use high weighting rather than Lagrange multipliers to constrain the points
    # to better work with noisy data
    constrained_weights = weight_array.copy()
    constrained_weights[:constrained_range] = constrained_weight
    constrained_weights[-constrained_range:] = constrained_weight

    baselines = np.empty((4, y.shape[0]))
    baselines[0] = fit_func(y, x, poly_orders[0], weights=weight_array, **method_kwargs)[0]
    baselines[1] = fit_func(y, x, poly_orders[0], weights=constrained_weights, **method_kwargs)[0]
    baselines[2] = fit_func(y, x, poly_orders[1], weights=weight_array, **method_kwargs)[0]
    baselines[3] = fit_func(y, x, poly_orders[1], weights=constrained_weights, **method_kwargs)[0]

    #TODO should the coefficients also be made available? Would need to get them from
    # each of the fits
    params = {
        'weights': weight_array, 'constrained_weights': constrained_weights,
        'poly_order': poly_orders
    }

    return np.maximum.reduce(baselines), params
