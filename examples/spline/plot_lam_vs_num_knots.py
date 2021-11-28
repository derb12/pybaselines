# -*- coding: utf-8 -*-
"""
`lam` vs number of knots
------------------------
This example will examine the effects of `lam` for fitting a penalized spline baseline
while varying both the number of knots for the spline, `num_knots`, and the number of
data points. The function :func:`.mixture_model` is used for all calculations.

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
# Other baseline types could be examined, similar to
# :ref:`Whittaker lam vs data size example <sphx_glr_examples_whittaker_plot_lam_vs_data_size.py>`,
# which should give similar results.
plt.plot(make_data(1000, bkg_type='exponential')[0])

# %%
# For each number of knots, the optimal `lam` value will be calculated for data sizes
# ranging from 500 to 20000 points. Further, the intercept and slope of the linear fit
# of the log of the data size, N, and the log of the `lam` value will be reported.
print('Number of knots, intercept & slope of log(N) vs log(lam) fit')
print('-' * 60)

show_plots = False  # for debugging
num_knots = (10, 50, 200, 500)
_, ax = plt.subplots()
legend = [[], []]
num_points = np.logspace(np.log10(500), np.log10(20000), 6, dtype=int)
symbols = cycle(['o', 's', 'd', 'h'])
for i, num_knot in enumerate(num_knots):
    func = partial(mixture_model, num_knots=num_knot)
    best_lams = np.empty_like(num_points, float)
    min_lam = -1
    for j, num_x in enumerate(num_points):
        y, baseline = make_data(num_x, bkg_type='exponential')
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
    print(f'{num_knot:<6} {coeffs}')
    line = 10**fit(np.log10(num_points))

    handle_1 = ax.plot(num_points, line)[0]
    handle_2 = ax.plot(num_points, 10**best_lams, next(symbols))[0]
    legend[0].append((handle_1, handle_2))
    legend[1].append(f'num_knots={num_knot}')

ax.loglog()
ax.legend(*legend)
ax.set_xlabel('Input Array Size, N')
ax.set_ylabel('Optimal lam Value')

# %%
# The results shown above show two findings: the slope of the `lam` vs data
# size best fit line only slightly decreases as the number of knots increases,
# and the intercept of the best fit lines increase as the number of knots increases.
