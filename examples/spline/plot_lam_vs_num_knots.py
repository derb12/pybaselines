# -*- coding: utf-8 -*-
"""
`lam` vs number of knots
------------------------

This example will examine the effects of `lam` for fitting a penalized spline baseline
while varying both the number of knots for the spline, `num_knots`, and the number of
data points. The function :meth:`.mixture_model` is used for all calculations.

Note that the exact optimal `lam` values reported in this example are not of significant
use since they depend on many other factors such as the baseline curvature, noise, peaks,
etc.; however, the examined trends can be used to simplify the process of selecting `lam`
values for fitting new datasets.

"""
# sphinx_gallery_thumbnail_number = 2

from functools import partial
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from pybaselines.spline import mixture_model

# local import with setup code
from example_helpers import make_data, optimize_lam


# %%
# The baseline for this example is an exponentially decaying baseline, shown below.
# Other baseline types could be examined, similar to the
# :ref:`Whittaker lam vs data size example <sphx_glr_examples_whittaker_plot_lam_vs_data_size.py>`,
# which should give similar results.
plt.plot(make_data(1000, bkg_type='exponential')[0])

# %%
# The number of knots will vary from 20 to 1000 on a logarithmic scale. For each number
# of knots, the optimal `lam` value will be calculated for data sizes ranging from
# 500 to 20000 points, also on a logarithmic scale.
show_plots = False  # for debugging
num_knots = np.logspace(np.log10(20), np.log10(1000.1), 5, dtype=int)
num_points = np.logspace(np.log10(500.1), np.log10(20000), 6, dtype=int)
symbols = cycle(['o', 's', 'd', 'h', '^', 'x'])
best_lams = np.empty((len(num_knots), len(num_points)))
for i, num_knot in enumerate(num_knots):
    func = partial(mixture_model, num_knots=num_knot, diff_order=2)
    min_lam = 0
    for j, num_x in enumerate(num_points):
        y, baseline = make_data(num_x, bkg_type='exponential')
        # use a slightly lower tolerance to speed up the calculation
        min_lam = optimize_lam(y, baseline, func, min_lam, tol=1e-2, max_iter=50)
        best_lams[i, j] = min_lam
        if show_plots:
            plt.figure(num=num_x)
            if i == 0:
                plt.plot(y)
            plt.plot(baseline)
            plt.plot(func(y, lam=10**min_lam)[0], '--')

# %%
# First, examine the relationship between the number of data points, `N`,
# and the optimal `lam` for fixed numbers of knots. The plots below show that
# the slopes of the best-fit lines are relatively small, indicating that the
# optimal `lam` increases only slightly as the number of points increases. The
# intercepts of the best-fit lines increase significantly as the number of
# knots increases, which will be investigated below.
print('Number of knots, intercept & slope of log(N) vs log(lam) fit')
print('-' * 60)
_, ax = plt.subplots()
legend = [[], []]
for i, num_knot in enumerate(num_knots):
    fit = np.polynomial.polynomial.Polynomial.fit(np.log10(num_points), best_lams[i], 1)
    coeffs = fit.convert().coef
    print(f'{num_knot:<6} {coeffs}')
    line = 10**fit(np.log10(num_points))

    handle_1 = ax.plot(num_points, line)[0]
    handle_2 = ax.plot(num_points, 10**best_lams[i], next(symbols))[0]
    legend[0].append((handle_1, handle_2))
    legend[1].append(f'num_knots={num_knot}')

ax.loglog()
ax.legend(*legend)
ax.set_xlabel('Input Array Size, N')
ax.set_ylabel('Optimal lam Value')

# %%
# Now, examine the relationship between the number of knots, and the optimal `lam`
# for fixed numbers of points. The plots below show that the slopes of the best-fit
# lines are much greater than the slopes from the `lam` versus `N` plots. The `lam`
# versus number of knots plots closely resemble the `lam` versus `N` plots from the
# :ref:`Whittaker lam vs data size example <sphx_glr_examples_whittaker_plot_lam_vs_data_size.py>`,
# which makes sense since the number of data points for Whittaker smoothing is more
# analogous to the number of knots for penalized splines when considering their minimized
# linear equations.
print('Number of points, intercept & slope of log(number of knots) vs log(lam) fit')
print('-' * 80)
_, ax = plt.subplots()
legend = [[], []]
for i, num_x in enumerate(num_points):
    fit = np.polynomial.polynomial.Polynomial.fit(np.log10(num_knots), best_lams[:, i], 1)
    coeffs = fit.convert().coef
    print(f'{num_x:<6} {coeffs}')
    line = 10**fit(np.log10(num_knots))

    handle_1 = ax.plot(num_knots, line)[0]
    handle_2 = ax.plot(num_knots, 10**best_lams[:, i], next(symbols))[0]
    legend[0].append((handle_1, handle_2))
    legend[1].append(f'data size={num_x}')

ax.loglog()
ax.legend(*legend)
ax.set_xlabel('Number of Knots')
ax.set_ylabel('Optimal lam Value')

plt.show()
