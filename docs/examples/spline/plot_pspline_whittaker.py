# -*- coding: utf-8 -*-
"""
P-spline versions of Whittaker functions
----------------------------------------

pybaselines contains penalized spline (P-spline) versions of all of the
Whittaker-smoothing-based algorithms implemented in pybaselines. The reason
for doing so was that P-splines offer additional user flexibility when choosing
parameters for fitting and more easily work for unequally spaced data. This example
will examine the relationship of `lam` versus the number of data points when fitting
a baseline with the :meth:`~.Baseline.arpls` function and its P-spline version,
:meth:`~.Baseline.pspline_arpls`.

Note that the exact optimal `lam` values reported in this example are not of significant
use since they depend on many other factors such as the baseline curvature, noise, peaks,
etc.; however, the examined trends can be used to simplify the process of selecting `lam`
values for fitting new datasets.

"""
# sphinx_gallery_thumbnail_number = 2

from itertools import cycle
import warnings

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline, utils


def _minimize(data, known_baseline, func, lams, **kwargs):
    """Finds the `lam` value that minimizes the L2 norm."""
    min_error = np.inf
    best_lam = lams[0]
    with warnings.catch_warnings():
        # ignore warnings that occur when using very small lam values
        warnings.filterwarnings('ignore')
        for lam in lams:
            try:
                baseline = func(data, lam=10**lam, **kwargs)[0]
            except np.linalg.LinAlgError:
                continue  # numerical instability can occur for lam >~1e12, just ignore
            fit_error = np.linalg.norm(known_baseline - baseline, 2)
            if fit_error < min_error:
                min_error = fit_error
                best_lam = lam

    return best_lam


def optimize_lam(data, known_baseline, func, previous_min=None, **kwargs):
    """
    Finds the optimum `lam` value.

    The optimal `lam` value should be ``10**best_lam``. Could alternatively
    use scipy.optimize.fmin, but this simple version is enough for examples.

    """
    if previous_min is None:
        min_lam = -1
    else:
        min_lam = previous_min - 0.5
    # coarse optimization
    lams = np.arange(min_lam, 13.5, 0.5)
    best_lam = _minimize(data, known_baseline, func, lams, **kwargs)
    # fine optimization
    lams = np.arange(best_lam - 0.5, best_lam + 0.7, 0.2)
    best_lam = _minimize(data, known_baseline, func, lams, **kwargs)

    return best_lam


# %%
# The baseline for this example is an exponentially decaying baseline, shown below.
# Other baseline types could be examined, similar to the
# :ref:`Whittaker lam vs data size example <sphx_glr_generated_examples_whittaker_plot_lam_vs_data_size.py>`,
# which should give similar results.
plt.plot(utils.make_data(1000, bkg_type='exponential', signal_type=2)[1])

# %%
# For each function, the optimal `lam` value will be calculated for data sizes
# ranging from 500 to 20000 points. Further, the intercept and slope of the linear fit
# of the log of the data size, N, and the log of the `lam` value will be reported.
# The number of knots for the P-spline version is fixed at the default, 100 (the effect
# of the number of knots versus optimal `lam` is shown in another
# :ref:`example <sphx_glr_generated_examples_spline_plot_lam_vs_num_knots.py>`).
print('Function, intercept & slope of log(N) vs log(lam) fit')
print('-' * 60)

show_plots = False  # for debugging
num_points = np.logspace(np.log10(500), np.log10(20000), 6, dtype=int)
symbols = cycle(['o', 's'])
_, ax = plt.subplots()
legend = [[], []]
for i, func_name in enumerate(('arpls', 'pspline_arpls')):
    best_lams = np.empty_like(num_points, float)
    min_lam = None
    for j, num_x in enumerate(num_points):
        func = getattr(Baseline(), func_name)
        x, y, baseline = utils.make_data(
            num_x, bkg_type='exponential', signal_type=2, return_baseline=True
        )
        # use a slightly lower tolerance to speed up the calculation
        min_lam = optimize_lam(y, baseline, func, min_lam, tol=1e-2, max_iter=50)
        best_lams[j] = min_lam

        if show_plots:
            plt.figure(num=num_x)
            if i == 0:
                plt.plot(y)
            plt.plot(baseline)
            plt.plot(func(y, lam=10**min_lam)[0], '--')

    fit = np.polynomial.polynomial.Polynomial.fit(np.log10(num_points), best_lams, 1)
    coeffs = fit.convert().coef
    print(f'{func_name:<16} {coeffs}')
    line = 10**fit(np.log10(num_points))

    handle_1 = ax.plot(num_points, line, label=func_name)[0]
    handle_2 = ax.plot(num_points, 10**best_lams, next(symbols))[0]
    legend[0].append((handle_1, handle_2))
    legend[1].append(func_name)

ax.loglog()
ax.legend(*legend)
ax.set_xlabel('Input Array Size, N')
ax.set_ylabel('Optimal lam Value')

plt.show()

# %%
# The results shown above demonstrate that the slope of the `lam` vs data
# size best fit line is much smaller for the P-spline based version of arpls.
# This means that once the number of knots is fixed for a particular baseline,
# the required `lam` value should be much less affected by a change in the
# number of data points (assuming the curvature of the data does not change).
#
# The above results are particularly useful when processing very large datasets.
# A `lam` value greater than ~1e14 typically causes numerical issues that can cause
# the solver to fail. Most Whittaker-smoothing-based algorithms reach that `lam`
# cutoff when the number of points is around ~20,000-500,000 (depends on the exact
# algorithm). Since the P-spline versions do not experience such a large increase in
# the required `lam`, they are more suited to fit those larger datasets. Additionally,
# the required `lam` value for the P-spline versions can be lowered simply by reducing
# the number of knots.
#
# It should be addressed that a similar result could be obtained using the regular
# Whittaker-smoothing-based version by truncating the number of points to a fixed
# value. That, however, would require additional processing steps to smooth out the
# resulting baseline after interpolating back to the original data size. Thus, the
# P-spline versions require less user-intervention to achieve the same result.
