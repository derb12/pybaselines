# -*- coding: utf-8 -*-
"""
Whittaker solver timings
------------------------

The Whittaker-smoothing-based algorithms in pybaselines make use of
the banded structure of the linear system to reduce the computation time.

This example shows the difference in computation times of the asymmetic least squares
(:meth:`~.Baseline.asls`) algorithm when using the banded solver from Scipy (solveh_banded)
and the banded solver from the optional dependency
`pentapy <https://github.com/GeoStat-Framework/pentapy>`_. In addition, the time
it takes when solving the system using sparse matrices rather than the banded matrices
is compared, since most other libraries use the sparse solution.

Compared to the time required to solve using sparse matrices, Scipy's banded solver
is ~50-70% faster and pentapy's banded solver is ~70-90% faster, ultimately reducing
the computation time by about an order of magnitude.

Note that the performance of solving the sparse system can be improved by using
`CHOLMOD from SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_, which has
Python bindings provided by `scikit-sparse <https://github.com/scikit-sparse/scikit-sparse>`_.

"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from pybaselines import whittaker, _banded_utils
from pybaselines.utils import difference_matrix, gaussian, relative_difference


def sparse_asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    A sparse version of the asymmetric least squares (AsLS) algorithm.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    y = np.asarray_chkfinite(data)
    num_y = len(y)
    if weights is None:
        weight_array = np.ones(num_y)
    else:
        weight_array = np.asarray(weights)
        if len(weight_array) != num_y:
            raise ValueError('data and weights must have the same length')

    diff_matrix = difference_matrix(num_y, diff_order, 'csc')
    penalty_matrix = lam * (diff_matrix.T @ diff_matrix)
    original_diag = penalty_matrix.diagonal()
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        penalty_matrix.setdiag(weight_array + original_diag)
        baseline = spsolve(penalty_matrix, weight_array * y, 'NATURAL')
        mask = y > baseline
        new_weights = p * mask + (1 - p) * (~mask)
        calc_difference = relative_difference(weight_array, new_weights)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

    return baseline, params


def scipy_asls(*args, **kwargs):
    """Temporarily turns off pentapy support to force scipy usage."""
    if _banded_utils._HAS_PENTAPY:
        _banded_utils._HAS_PENTAPY = False
        reset_pentapy = True
    else:
        reset_pentapy = False

    try:
        output = whittaker.asls(*args, **kwargs)
    finally:
        if reset_pentapy:
            _banded_utils._HAS_PENTAPY = True

    return output


def make_data(num_x):
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
    baseline = 5 + 15 * np.exp(-x / 200)
    noise = np.random.default_rng(0).normal(0, 0.1, num_x)
    y = signal + baseline + noise

    return y


if __name__ == '__main__':

    if not _banded_utils._HAS_PENTAPY:
        warnings.warn(
            'pentapy is not installed so pentapy and scipy-banded timings will be identical',
            stacklevel=2
        )

    # equation obtained following similar procedure as `lam` vs data size example
    lam_equation = lambda n: 10**(-6.35 + np.log10(n) * 4.17)
    repeats = 25
    functions = (
        (whittaker.asls, 'pentapy'),
        (scipy_asls, 'scipy-banded'),
        (sparse_asls, 'scipy-sparse'),
    )

    for i, (func, func_name) in enumerate(functions):
        timings = []
        for num_x in np.logspace(np.log10(500), np.log10(40000), 8, dtype=int):
            y = make_data(num_x)
            lam = lam_equation(num_x)
            times = []
            for j in range(repeats + 1):
                t0 = time.perf_counter()
                # force same number of iterations for all functions so that
                # timings are comparable
                baseline, params = func(y, lam=lam, tol=-1, max_iter=8)
                t1 = time.perf_counter() - t0
                if j > 0:  # ignore first function call for more accurate timings
                    times.append(t1)
            # use median instead of mean so timing outliers have less effect
            timings.append((num_x, np.median(times), np.std(times, ddof=1)))
        plt.errorbar(*np.array(timings).T, label=func_name)

    plt.loglog()
    plt.xlabel('Input Array Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.show()
