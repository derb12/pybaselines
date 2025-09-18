# -*- coding: utf-8 -*-
"""
Whittaker solver timings
------------------------

The Whittaker-smoothing-based algorithms in pybaselines make use of
the banded structure of the linear system to reduce the computation time.

This example shows the difference in computation times of the asymmetic least squares
(:meth:`~pybaselines.Baseline.asls`) algorithm when using the banded solver from SciPy,
:func:`scipy.linalg.solve_banded`, and the banded solver from the optional dependency
`pentapy <https://github.com/GeoStat-Framework/pentapy>`_. In addition, the time
it takes when solving the system using sparse matrices rather than the banded matrices
is compared, since direct adaptation from literature usually uses the sparse solution.
All three of these solvers are based on LU decomposition. Since the asls algorithm results
in a symmetric, positive-definite left-hand side of the normal equation, it can additionally
be solved using Cholesky decomposition through the dedicated SciPy solver
:func:`scipy.linalg.solveh_banded`.

Compared to the time required to solve using sparse matrices, SciPy's banded solvers
are ~50-80% faster and pentapy's banded solver is ~70-90% faster, ultimately reducing
the computation time by about an order of magnitude.

Note that the performance of solving this particular sparse system can be improved by using
the sparse Cholesky decomposition solver
`CHOLMOD from SuiteSparse <https://github.com/DrTimothyAldenDavis/SuiteSparse>`_, which has
Python bindings provided by `scikit-sparse <https://github.com/scikit-sparse/scikit-sparse>`_.

"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve

from pybaselines import Baseline, utils
from pybaselines.utils import difference_matrix, relative_difference


def sparse_asls(data, lam=1e6, p=1e-2, diff_order=2, max_iter=50, tol=1e-3, weights=None):
    """
    A sparse version of the asymmetric least squares (AsLS) algorithm.

    References
    ----------
    Eilers, P. A Perfect Smoother. Analytical Chemistry, 2003, 75(14), 3631-3636.

    """
    y = np.asarray(data)
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
        new_weights = np.where(y > baseline, p, 1 - p)
        calc_difference = relative_difference(weight_array, new_weights)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        weight_array = new_weights

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}

    return baseline, params


if __name__ == '__main__':

    try:
        import pentapy  # noqa
    except ImportError:
        warnings.warn(
            'pentapy is not installed so pentapy and solveh_banded timings will be identical',
            stacklevel=2
        )

    # equation obtained following similar procedure as `lam` vs data size example
    lam_equation = lambda n: 10**(-6.35 + np.log10(n) * 4.17)
    repeats = 25
    functions = (
        'sparse',
        'solve_banded',
        'solveh_banded',
        'pentapy',
    )
    # solver_numbers corresponds to the settings for the `banded_solver` attribute of
    # Baseline objects for each of the solvers; pentapy could be 1 or 2
    solver_numbers = {'solve_banded': 4, 'solveh_banded': 3, 'pentapy': 2}
    func_timings = {}
    data_sizes = np.logspace(np.log10(500), np.log10(40000), 8, dtype=int)
    for func_name in functions:
        timings = []
        for num_x in data_sizes:
            y = utils.make_data(num_x, bkg_type='exponential', noise_std=0.1, signal_type=2)[1]
            lam = lam_equation(num_x)
            if func_name == 'sparse':
                func = sparse_asls
            else:
                fitter = Baseline(check_finite=False, assume_sorted=True)
                fitter.banded_solver = solver_numbers[func_name]
                func = fitter.asls
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
            timings.append((np.median(times), np.std(times, ddof=1)))
        total_timings = np.array(timings).T
        plt.errorbar(data_sizes, *total_timings, label=func_name)
        func_timings[func_name] = total_timings

    plt.loglog()
    plt.xlabel('Input Array Size')
    plt.ylabel('Median Time (seconds)')
    plt.legend()

    # The relative time reduced by using pentapy can be compared for each of the other methods
    plt.figure()
    reference_key = 'pentapy'
    reference_times = func_timings[reference_key]
    for key, values in func_timings.items():
        if key == reference_key:
            continue
        relative_speedup = 100 * (values[0] - reference_times[0]) / values[0]
        # use propogation of errors to estimate relative speedup error
        speedup_err = (
            (100 / values[0])
            * np.sqrt(reference_times[1]**2 + reference_times[0]**2 * values[1]**2 / values[0]**2)
        )
        plt.errorbar(data_sizes, relative_speedup, speedup_err, label=key)

    plt.semilogx()
    plt.xlabel('Input Array Size')
    plt.ylabel('Relative Time Reduction (%)')
    plt.legend()

    plt.show()
