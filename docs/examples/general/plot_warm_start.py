# -*- coding: utf-8 -*-
"""
Warm-Starting Iteratively Reweighted Methods
--------------------------------------------

For methods that perform iterative reweighting, which includes
:doc:`Whittaker smoothing methods <../../../algorithms/algorithms_1d/whittaker>`, most
:doc:`spline methods <../../../algorithms/algorithms_1d/spline>`,
:meth:`~pybaselines.Baseline.loess` and
:meth:`~pybaselines.Baseline.quant_reg`, if the peaks within a dataset have the same
general position, then the weights from a previous fit can be used to provide a warm-start for
the next calculation, which generally improves the convergence rate. This can be especially
useful for baseline correction algorithms in which each iteration is fairly computationally
expensive, such as :meth:`~pybaselines.Baseline.loess`.

This example will show a demonstration of how to use the output weights of a method
to then warm-start each subsequent fit for a large dataset containing data with
similar peaks. The methods selected for this example are chosen for their convergence
behavior (classified as convergence typically occurring in <5 iterations (fast),
5-20 iterations (medium), and >20 iterations (slow)) and computational speed:

1) :meth:`~pybaselines.Baseline.asls`: fast convergence, fast computation
2) :meth:`~pybaselines.Baseline.arpls`: medium convergence, fast computation
3) :meth:`~pybaselines.Baseline.aspls`: slow convergence, fast computation
4) :meth:`~pybaselines.Baseline.loess`: medium convergence, slow computation
5) :meth:`~pybaselines.Baseline.quant_reg`: slow convergence, medium computation

"""
# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_multi_image = "single"

from collections import defaultdict
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline, utils


def generate_data(x, seed):
    """Generates a dataset where peaks and baseline are slightly different each time."""
    rng = np.random.default_rng(seed)
    heights = (9, 6, 8, 15, 6, 13, 9, 9)
    centers = (100, 180, 350, 400, 550, 700, 800, 880)
    sigmas = (12, 5, 11, 18, 6, 8, 9, 7)

    signal = np.zeros_like(x)
    for (height, center, sigma) in zip(heights, centers, sigmas):
        peak = utils.gaussian(
            x, rng.normal(height, 2), rng.normal(center, 5), rng.normal(sigma, 0.5)
        )
        signal += peak
    baseline = rng.normal(5, 0.5) + rng.normal(10, 3) * np.exp(-x / rng.normal(600, 100))
    noise = rng.normal(0, 0.1, len(x))
    return signal + baseline + noise


num_fits = 500  # equivalent to the number of data in a dataset
x = np.linspace(0, 1000, 1000)
baseline_fitter = Baseline(x_data=x, check_finite=False, assume_sorted=True)

# %%
# This example will use a dataset that contains peaks with similar positions
# and somewhat similar baselines.
for i in range(num_fits):
    plt.plot(x, generate_data(x, seed=i))

# %%
# The code below loops through each method and records the time and number of iterations
# for fitting the entire dataset using the full calculation and a warm-start.
# A comparative plot of the warm-started and non-warm-started fits for one of the datasets
# is shown below, showing good agreement. In addition, the convergence of each method is
# shown for one dataset for the full calculation and warm-start.
methods = (
    ('asls', {'lam': 1e7}),
    ('arpls', {'lam': 1e7}),
    ('aspls', {'lam': 1e8}),
    ('loess', {}),
    ('quant_reg', {'poly_order': 5, 'tol': 1e-4}),  # default quant_reg tol is 1e-6
)
_, (ax2, ax3) = plt.subplots(nrows=2, sharex=True, tight_layout=True)
_, (ax4, ax5) = plt.subplots(nrows=2, sharex=True, tight_layout=True)
ax2.plot(x, generate_data(x, seed=0))
ax3.plot(x, generate_data(x, seed=0))
timings = defaultdict(list)
for i, (method, kwargs) in enumerate(methods):
    func = getattr(baseline_fitter, method)
    for warm_start in (True, False):
        weights = None
        if method == 'aspls':
            kwargs['alpha'] = None
        for j in range(num_fits + 1):
            y = generate_data(x, j)
            t0 = perf_counter()
            calc_baseline, params = func(y, weights=weights, **kwargs)
            t1 = perf_counter()
            if j == 0:  # only plot once per algorithm
                if warm_start:
                    axis = ax3
                    label = None
                else:
                    axis = ax2
                    label = method
                axis.plot(x, calc_baseline, label=method)
            elif j == 2:
                if warm_start:
                    axis = ax5
                    label = method
                else:
                    axis = ax4
                    label = None
                # plot one representative tol_history for each method
                axis.plot(params['tol_history'], 'o-', label=label)
            if j > 0:
                # only add timings after first call to allow for any necessary compilation
                timings[f'{method}_{warm_start}_time'].append(t1 - t0)
                if warm_start:
                    # also only sets weights after first call to include the timing of the
                    # first non-warm-started call
                    weights = params['weights']
                    if method == 'aspls':
                        # aspls also uses an 'alpha' term that controls local stiffness
                        # around peaks, which can also be warm-started
                        kwargs['alpha'] = params['alpha']
            if j > 1:
                # only count iterations after first call that set the weights for warm-start
                timings[f'{method}_{warm_start}_iterations'].append(len(params['tol_history']))

ax2.set_title('Full Calculation')
ax3.set_title('Warm-start')

ax4.xaxis.set_tick_params(labelbottom=True)  # gets removed since its shares x with ax5
ax4.set_xlabel('Iterations')
ax5.set_xlabel('Iterations')
ax4.set_ylabel('Calculated Tolerance')
ax5.set_ylabel('Calculated Tolerance')
ax4.set_title('Full Calculation')
ax5.set_title('Warm-start')

ax2.legend()
ax5.legend()
ax4.semilogy()
ax5.semilogy()

# %%
# As seen in the above figures, the use of prior information from a previous fit
# reduces the overall number of iterations to reach the specified exit tolerance
# for all methods. We can also examine how many iterations on average were saved
# by using a warm-start for each method.
for (method, _) in methods:
    warmstart_sum = np.sum(timings[f'{method}_True_iterations'])
    new_sum = np.sum(timings[f'{method}_False_iterations'])
    percent_saved = 100 * (new_sum - warmstart_sum) / new_sum
    new_avg = np.mean(timings[f'{method}_False_iterations'])
    warmstart_avg = np.mean(timings[f'{method}_True_iterations'])

    print((
        f'{method} saved {new_avg - warmstart_avg:.1f} iterations on average, '
        f'or about {percent_saved:.1f}% of its iterations'
    ))

# %%
# In addition to the number of iterations saved, the time saved by using
# warm-starting for each method is shown below. While all methods save time
# overall, ones that had a high computational cost per iteration
# especially benefited from the faster convergence.
_, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, tight_layout=True)
for i, (method, _) in enumerate(methods):
    warmstart = sum(timings[f'{method}_True_time'])
    new = sum(timings[f'{method}_False_time'])
    if i == 0:
        new_label = 'Full Calculation'
        warmstart_label = 'Warm-start'
    else:
        new_label = ''
        warmstart_label = ''
    plt.plot()
    ax1.bar(i - 0.2, new, width=0.4, label=new_label, color='c')
    ax1.bar(i + 0.2, warmstart, width=0.4, label=warmstart_label, color='m')
    speedup = 100 * (new - warmstart) / new
    bar = ax2.bar(i, speedup)
    ax2.bar_label(bar, fmt='{:.1f}%')

ax1.legend()
ax2.set_xticks(np.arange(len(methods)), [method for method, _ in methods], rotation=15)
ax2.set_ylim(ax2.get_ylim() + np.array([-5, 5]))  # add space for the bar labels
ax1.set_ylabel('Total time (s)')
ax2.set_ylabel('Relative Time Reduction (%)')

plt.show()
