# -*- coding: utf-8 -*-
"""
Fitting Multiple Datasets
-------------------------

When fitting multiple datasets that all share the same independant variable, pybaselines
allows saving time by reusing the same :class:`~.Baseline` object to allow only
performing some of the computationally heavy setup only once. For example,
:doc:`polynomial methods <../../../algorithms/polynomial>` will only compute the Vandermonde
matrix, and potentially its pseudoinverse, once. Likewise,
:doc:`spline methods <../../../algorithms/spline>` will only have to compute the spline
basis matrix once. Note that this only applies if the same non-data parameters
(eg. ``poly_order``, ``num_knots``, etc.) are used for each fit.

This example will explore the efficiency of reusing the same ``Baseline`` object when fitting
multiple datasets for different types of algorithms.

"""
# sphinx_gallery_thumbnail_number = 2

from collections import defaultdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline
from pybaselines.utils import gaussian


num_points = 1000  # number of data points in one set of data
num_fits = 1000  # equivalent to the number of data in a dataset

x = np.linspace(0, 1000, num_points)
signal = (
    + gaussian(x, 6, 150, 5)
    + gaussian(x, 8, 350, 11)
    + gaussian(x, 6, 550, 6)
    + gaussian(x, 13, 700, 8)
    + gaussian(x, 9, 880, 7)
)
baseline = 5 + 10 * np.exp(-x / 600) + gaussian(x, 15, 1000, 400)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

# %%
# Six different methods will be timed. The polynomial method :meth:`~.Baseline.penalized_poly`,
# the spline method :meth:`~.Baseline.mixture_model`, the Whittaker smoothing method
# :meth:`~.Baseline.iarpls`, the morphological method :meth:`~.Baseline.mor`, the smoothing
# method :meth:`~.Baseline.ria`, and the classification method :meth:`~.Baseline.std_distribution`.

methods = (
    ('penalized_poly', {'poly_order': 4}),
    ('mixture_model', {'lam': 1e5}),
    ('iarpls', {'lam': 1e5}),
    ('mor', {'half_window': 30}),
    ('ria', {'half_window': 20}),
    ('std_distribution', {'half_window': 25})
)
plt.plot(x, y)
timings = defaultdict(list)
for method, kwargs in methods:
    baseline_fitter = Baseline(x_data=x, check_finite=False, assume_sorted=True)
    for reuse_object in (True, False):
        for i in range(num_fits):
            if reuse_object:
                func = getattr(baseline_fitter, method)
            else:
                func = getattr(Baseline(x, check_finite=False, assume_sorted=True), method)
            t0 = perf_counter()
            calc_baseline = func(y, **kwargs)[0]
            t1 = perf_counter()
            if i == 0 and reuse_object:  # only plot once per algorithm
                plt.plot(x, calc_baseline, label=method)
            elif i > 0:
                # only add timings after first call to allow for any necessary compilation
                timings[f'{method}_{reuse_object}'].append(t1 - t0)

plt.legend()
plt.tight_layout()

# %%
# The total times for each method when using a new ``Baseline`` object each call
# and when reusing the same ``Baseline`` object are plotted below, as well as
# the relative time reduction by reusing the same ``Baseline`` object. Note
# that time reductions less than +/-5% can be considered as irrelevant.

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
for i, (method, _) in enumerate(methods):
    reuse = sum(timings[f'{method}_True'])
    new = sum(timings[f'{method}_False'])
    if i == 0:
        new_label = 'New'
        reuse_label = 'Reuse'
    else:
        new_label = ''
        reuse_label = ''
    plt.plot()
    ax1.bar(i - 0.2, new, width=0.4, label=new_label, color='c')
    ax1.bar(i + 0.2, reuse, width=0.4, label=reuse_label, color='m')
    speedup = 100 * (new - reuse) / new
    bar = ax2.bar(i, speedup)
    ax2.bar_label(bar, fmt='{:.1f}%')

ax1.legend()
ax2.set_xticks(np.arange(len(methods)), [method for method, _ in methods], rotation=15)
ax2.set_ylim(ax2.get_ylim() + np.array([-5, 5]))  # add space for the bar labels
ax1.set_ylabel('Total time (s)')
ax2.set_ylabel('Relative Time Reduction (%)')
fig.tight_layout()

plt.show()

# %%
# As expected, the polynomial and spline methods see a significant time reduction by reusing
# the same ``Baseline`` object to fit the entire dataset while the other methods see no difference.
# Note that these results are a generalization; algorithms that are more computationally intensive
# will see less of a benefit from reuse since less time is from the setup and more from the actual
# algorithm.
