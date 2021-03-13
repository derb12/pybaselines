# -*- coding: utf-8 -*-
"""High level functions for .


Created on March 3, 2021

@author: Donald Erb

"""

import numpy as np

from .morphological import mpls
from .penalized_least_squares import iarpls, airpls, arpls, asls, aspls, drpls, iasls
from .polynomial import imodpoly, modpoly

from .utils import gaussian


def collab_pls(data, average_dataset=True, method='asls', full=False, **method_kwargs):
    """
    Collaborative Penalized Least Squares (collab-PLS).

    Parameters
    ----------
    data : np.ndarray
        A numpy array with shape (M, N) where M is the number of entries in
        the dataset and N is the number of data points in each entry.
    average_dataset : bool
        If True (default) will average the dataset before fitting to get the
        weighting. If False, will fit each individual entry in the dataset and
        then average the weights to get the weighting for the dataset.
    method : {}
        [description], by default 'asls'
    full : bool
        If True, will return the weights along with the baselines.
    **method_kwargs

    Returns
    -------
    np.ndarray
        An array of all of the baselines with shape (M, N).

    Adapted from:
        Chen, L. et. al. Collaborative Penalized Least Squares for Background
        Correction of Multiple Raman Spectra. Journal of Analytical Methods in
        Chemistry, 2018 (2018). DOI:https://doi.org/10.1155/2018/9031356

    """
    fit_func = {
        'arpls': arpls,
        'aspls': aspls,
        'iarpls': iarpls,
        'airpls': airpls,
        'mpls': mpls,
        'asls': asls,
        'iasls': iasls,
        'drpls': drpls
    }[method.lower()]
    method_kwargs['full'] = True
    if average_dataset:
        _, fit_params = fit_func(np.mean(data.transpose(), 1), **method_kwargs)
        method_kwargs['weights'] = fit_params['weights']
    else:
        weights = []
        for entry in data:
            _, fit_params = fit_func(entry, **method_kwargs)
            weights.append(fit_params['weights'])
        method_kwargs['weights'] = np.mean(np.array(weights), 1)

    method_kwargs.update({'full': False, 'tol': np.inf})
    baselines = []
    for entry in data:
        baselines.append(fit_func(entry, **method_kwargs))

    if not full:
        return np.vstack(baselines)
    else:
        return np.vstack(baselines), {'weights': method_kwargs['weights']}


def _iter_solve(func, fit_data, known_background, lower_bound, upper_bound, variable,
                min_value, max_value, step=1, allowed_misses=1, **func_kwargs):
    min_rmse = np.inf
    misses = 0
    for var in np.arange(min_value, max_value, step):
        if variable == 'lam':
            func_kwargs[variable] = 10**var
        else:
            func_kwargs[variable] = var
        baseline = func(fit_data, **func_kwargs)
        rmse = np.sqrt(np.mean(
            (known_background - np.roll(baseline, upper_bound)[:upper_bound + lower_bound])**2
        ))
        if rmse < min_rmse:
            z = baseline[lower_bound:baseline.shape[0] - upper_bound]
            min_var = var
            misses = 0
            min_rmse = rmse
        else:
            misses += 1
            if misses > allowed_misses:
                break

    return z, min_var


def erpls(data, x_data, method='aspls', side='left', **method_kwargs):
    """
    Extended range penalized least squares (erPLS) baseline method.

    Useful for calculating the lambda value required to optimize other
    assymetric least squares algorithms.

    Adapted from:
        Zhang, F. et. al. An Automatic Baseline Correction Method Based on
        the Penalized Least Squares Method. Sensors, 20(7) (2020), 2015.
    Note: the 2015 is the article number, not the page number due to how
    MDPI does its referencing.
    """
    if side.lower() not in ('left', 'right', 'both'):
        raise ValueError('side must be "left", "right", or "both"')
    fit_func = {
        'arpls': arpls,
        'aspls': aspls,
        'iarpls': iarpls,
        'airpls': airpls,
        'mpls': mpls,
        'asls': asls,
        'iasls': iasls,
        'drpls': drpls,
        'modpoly': modpoly,
        'imodpoly': imodpoly
    }[method.lower()]

    x = np.asarray(x_data)
    sort_order = tuple(enumerate(np.argsort(x)))  # to ensure x is increasing
    x = x[[val[1] for val in sort_order]]
    y = np.asarray(data)[[val[1] for val in sort_order]]
    max_x = np.nanmax(x)
    min_x = np.nanmin(x)
    x_range = max_x - min_x
    known_background = np.array([])
    fit_x_data = x
    fit_data = y
    lower_bound = upper_bound = 0

    W = x.shape[0] // 10

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

    if method.lower() in ('iasls', 'modpoly', 'imodpoly'):
        method_kwargs['x_data'] = fit_x_data

    if method.lower() in ('modpoly', 'imodpoly'):
        z, best_val = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'poly_order',
            0, 20, 1, 4, **method_kwargs
        )
    else:
        _, best_val = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'lam',
            1, 50, 1, 2, **method_kwargs
        )
        z, best_val = _iter_solve(
            fit_func, fit_data, known_background, lower_bound, upper_bound, 'lam',
            best_val - 0.9, best_val + 1.1, 0.1, 2, **method_kwargs
        )

    print(best_val)

    return z[[val[0] for val in sorted(sort_order, key=lambda v: v[1])]]


def _optional_kwargs(keys, kwargs):
    """
    [summary]

    Parameters
    ----------
    keys : Iterable(str)
        [description]
    kwargs : dict


    Returns
    -------
    dict
        A dictionary of keyword arguments containing any values in kwargs.

    """
    if isinstance(keys, str):
        keys = (keys,)

    return {key: kwargs[key] for key in keys if key in kwargs}


def get_baseline(method, data=None, x_data=None, **kwargs):
    """
    A convenience function to select the appropriate baseline function.

    Parameters
    ----------
    method : {}
        A string indicating the baseline method to use.
    data : [type], optional
        [description], by default None
    x_data : [type], optional
        [description], by default None
    **kwargs
        Additional keyword arguments for the various baseline functions.

    Returns
    -------
    baseline : np.ndarray
        The baseline array for the input data and options.

    """
    method = method.lower()
    available_methods = {'manual', 'asls', 'iasls', 'airpls', 'drpls', 'arpls', 'iarpls'}
    if method not in available_methods:
        raise ValueError(f'baseline method must be in {available_methods}')

    if method == 'manual':
        baseline = manual_baseline(x_data, **_optional_kwargs(('background_points',), kwargs))
    elif method == 'asls':
        baseline = asls_baseline(
            data, **_optional_kwargs(('lam', 'p', 'max_iter', 'tol'), kwargs)
        )
    elif method == 'iasls':
        baseline = iasls_baseline(
            data, x_data, **_optional_kwargs(('lam', 'lam_1', 'p', 'max_iter', 'tol'), kwargs)
        )
    elif method == 'airpls':
        baseline = airpls_baseline(
            data, **_optional_kwargs(('lam', 'order', 'max_iter', 'tol'), kwargs)
        )
    elif method == 'drpls':
        baseline = drpls_baseline(
            data, **_optional_kwargs(('lam', 'eta', 'max_iter', 'tol'), kwargs)
        )
    elif method == 'arpls':
        baseline = arpls_baseline(
            data, **_optional_kwargs(('lam', 'max_iter', 'tol'), kwargs)
        )
    elif method == 'iarpls':
        baseline = iarpls_baseline(
            data, **_optional_kwargs(('lam', 'max_iter', 'tol'), kwargs)
        )

    return baseline
