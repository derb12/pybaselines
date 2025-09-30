# -*- coding: utf-8 -*-
"""
`lam` vs data size
------------------

When publishing new Whittaker-smoothing-based algorithms in literature, the `lam` value
used by the researchers is usually reported as a single value or a range of values.
However, these values are deceptive since the `lam` value required for a particular
Whittaker-smoothing-based algorithm is dependent on the number of data points. Thus,
this can cause issues when adapting an algorithm to a new set of data since the published
optimal `lam` value is not universal. This example shows an analysis of this dependence
for all available :ref:`Whittaker smoothing methods <api/Baseline:Whittaker Smoothing Algorithms>`.

Note that the exact optimal `lam` values reported in this example are not of significant
use since they depend on many other factors such as the baseline curvature, noise, peaks,
etc.; however, the examined trends can be used to simplify the process of selecting `lam`
values for fitting new datasets.

"""
# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_multi_image = "single"

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


def iasls(*args, lam=1, p=0.05, **kwargs):
    """Ensures the `lam_1` value for whittaker.iasls scales with `lam`."""
    # not sure if lam_1 should be fixed or proportional to lam;
    # both give similar results
    return Baseline().iasls(*args, lam=lam, lam_1=1e-8 * lam, p=p, **kwargs)


# %%
# Three baselines will be tested: an exponentially decaying baseline, a gaussian
# baseline, and a sinusoidal baseline. The exponential baseline is the most smooth
# and the sine baseline is the least smooth, so it would be expected that a lower
# `lam` value would be required to fit the sine baseline and a higher value for the
# exponential baseline. The three different plots are shown below.

bkg_dict = {}
bkg_types = ('exponential', 'gaussian', 'sine')
for bkg_type in bkg_types:
    bkg_dict[bkg_type] = {}
    plt.plot(utils.make_data(1000, bkg_type, signal_type=2, noise_std=0.1)[1], label=bkg_type)
plt.legend()

# %%
# For each function, the optimal `lam` value will be calculated for data sizes
# ranging from 500 to 20000 points. Further, the intercept and slope of the linear fit
# of the log of the data size, N, and the log of the `lam` value will be reported.
print('Function, baseline type, intercept & slope of log(N) vs log(lam) fit')
print('-' * 60)

show_plots = False  # for debugging
num_points = np.logspace(np.log10(500), np.log10(20000), 6, dtype=int)
symbols = cycle(['o', 's', 'd'])
for i, func_name in enumerate((
    'asls', 'iasls', 'airpls', 'arpls', 'iarpls', 'drpls', 'aspls', 'psalsa', 'derpsalsa',
    'brpls', 'lsrpls'
)):
    legend = [[], []]
    _, ax = plt.subplots(num=func_name)
    for bkg_type in bkg_types:
        best_lams = np.empty_like(num_points, float)
        min_lam = None
        for j, num_x in enumerate(num_points):
            if func_name == 'iasls':
                func = iasls
            else:
                func = getattr(Baseline(), func_name)
            x, y, baseline = utils.make_data(
                num_x, bkg_type, signal_type=2, noise_std=0.1, return_baseline=True
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
        print(f'{func_name:<11} {bkg_type:<13} {coeffs}')
        line = 10**fit(np.log10(num_points))
        bkg_dict[bkg_type][func_name] = line

        handle_1 = ax.plot(num_points, line, label=bkg_type)[0]
        handle_2 = ax.plot(num_points, 10**best_lams, next(symbols))[0]
        legend[0].append((handle_1, handle_2))
        legend[1].append(bkg_type)

    ax.loglog()
    ax.legend(*legend)
    ax.set_xlabel('Input Array Size, N')
    ax.set_ylabel('Optimal lam Value')
    ax.set_title(func_name)

# %%
# To further analyze the relationship of `lam` and data size, the best
# fit lines for all algorithms for each baseline type are shown below.
# Interestingly, for each baseline type, the slopes of the `lam` vs data size
# lines are approximately the same for all algorithms; only the intercept
# is different. This makes sense since all functions use very similar linear
# equations for solving for the baseline; thus, while the optimal `lam` values
# may differ between the algorithms, the relationship between `lam` and the data
# size should be similar for all of them.

for bkg_type in bkg_types:
    plt.figure()
    for func_name, line in bkg_dict[bkg_type].items():
        plt.plot(num_points, line, label=func_name)
    plt.legend()
    plt.loglog()
    plt.xlabel('Input Array Size, N')
    plt.ylabel('Optimal lam Value')
    plt.title(f'{bkg_type} baseline')

plt.show()
