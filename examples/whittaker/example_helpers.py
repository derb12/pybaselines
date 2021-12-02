# -*- coding: utf-8 -*-
"""Helper functions for use in some examples.

The functions are put within this file so that all of the setup code does
not need to clutter the output of the example programs in the documentation.

"""

import warnings

import numpy as np

from pybaselines.utils import gaussian


def _minimize(data, known_baseline, func, lams, **kwargs):
    """Finds the `lam` value that minimizes the L2 norm."""
    min_error = np.inf
    best_lam = lams[0]
    with warnings.catch_warnings():
        # ignore warnings that occur when using very small lam values
        warnings.filterwarnings('ignore')
        for lam in lams:
            baseline = func(data, lam=10**lam, **kwargs)[0]
            error = np.linalg.norm(known_baseline - baseline, 2)
            if error < min_error:
                min_error = error
                best_lam = lam

    return best_lam


def optimize_lam(data, known_baseline, func, previous_min=None, **kwargs):
    """
    Finds the optimum `lam` value.

    The optimal `lam` value should be ``10**best_lam``. Could alternatively
    use scipy.optimize.fmin, but this simple version is faster for this
    particular example.

    """
    if previous_min is None:
        min_lam = 0
    else:
        min_lam = previous_min - 0.5
    # coarse optimization
    lams = np.arange(min_lam, 13.5, 0.5)
    best_lam = _minimize(data, known_baseline, func, lams, **kwargs)
    # fine optimization
    lams = np.arange(best_lam - 0.5, best_lam + 0.7, 0.2)
    best_lam = _minimize(data, known_baseline, func, lams, **kwargs)

    return best_lam


def make_data(num_x, bkg_type='exponential'):
    """Creates the data for the example."""
    x = np.linspace(0, 1000, num_x)
    signal = (
        gaussian(x, 9, 100, 12)
        + gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 11)
        + gaussian(x, 15, 400, 18)
        + gaussian(x, 6, 550, 6)
        + gaussian(x, 13, 700, 8)
        + gaussian(x, 9, 800, 9)
        + gaussian(x, 9, 880, 7)
    )
    if bkg_type == 'exponential':
        baseline = 5 + 15 * np.exp(-x / 200)
    elif bkg_type == 'gaussian':
        baseline = 30 + gaussian(x, 20, 500, 150)
    elif bkg_type == 'linear':
        baseline = 1 + x * 0.005
    elif bkg_type == 'sine':
        baseline = 70 + 5 * np.sin(x / 50)
    else:
        raise ValueError(f'unknown bkg_type {bkg_type}')

    noise = np.random.default_rng(0).normal(0, 0.1, num_x)
    y = signal + baseline + noise

    return y, baseline
