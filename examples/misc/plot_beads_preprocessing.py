# -*- coding: utf-8 -*-
"""
Preprocessing for beads
-----------------------

The Baseline Estimation And Denoising with Sparsity (:func:`.beads`) algorithm is a
robust method for both performing baseline subtraction and removing noise. One of the
main drawbacks of the original algorithm is that it requires that both ends of
the data to be at zero. This example will explore the consequences of this as
well as a preprocessing step proposed by `Navarro-Huerta, J.A., et al. Assisted baseline
subtraction in complex chromatograms using the BEADS algorithm. Journal of Chromatography
A, 2017, 1507, 1-10` that helps to address this issue.

"""
# sphinx_gallery_thumbnail_number = 4

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from pybaselines.misc import beads, _parabola
from pybaselines.utils import gaussian


def make_data(x, baseline_type=0):
    """Creates the data and baseline where the endpoints are either zero or nonzero."""
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
    if baseline_type == 0:  # simple parabola that ends at 0 on both ends
        baseline = 2e-5 * (x - 500)**2 - 5
    elif baseline_type == 1:  # exponentially decaying baseline
        baseline = 10 - 10 * np.exp(-x / 600)
    else:  # very complicated baseline
        # approximate a logistic baseline by integrating a Gaussian
        baseline = -np.cumsum(gaussian(x, 0.05, 400, 100)) + gaussian(x, 3, 800, 100) - 5
    noise = np.random.default_rng(0).normal(0, 0.2, len(x))
    y = signal + baseline + noise

    return y, baseline


linestyles = cycle(('-', '--'))

# %%
# Three different baselines will be compared: (1) a simple polynomial that ends at zero on
# both ends; (2) an exponential baseline that is only zero on one end; (3) a complex baseline
# that is not near 0 on either end. Each plot shows the raw data, the parabola used to preprocess
# the data as proposed by Navarro-Huerta, J.A., et al. to get both ends of the data to zero, and
# the raw data after subtracting the parabola.
x = np.linspace(0, 1000, 1000)
figure, axes = plt.subplots(nrows=3)
for i in range(3):
    y, baseline = make_data(x, baseline_type=i)
    parabola = _parabola(y)
    labels = ('raw data', 'fit parabola', 'processed data') if i == 0 else (None, None, None)
    axes[i].plot(y, label=labels[0])
    axes[i].plot(parabola, '--', label=labels[1])
    axes[i].plot(y - parabola, label=labels[2])
    axes[i].set_xticks([])
    axes[i].set_title(f'Baseline {i + 1}', y=0.95)

figure.legend(ncol=3, loc='upper center')
figure.tight_layout()
figure.subplots_adjust(top=0.85)

# %%
# For the polynomial baseline that is zero at both ends, the beads algorithm gives similar
# results for both the raw data and the parabola-subtracted data. This indicates that any
# preprocessing is not needed for such a baseline, but also shows that the preprocessing
# does not negatively influence the result.
y, baseline = make_data(x, baseline_type=0)
plt.figure()
plt.plot(y)
for subtract_parabola in (False, True):
    fit_baseline = beads(y, lam_0=0.005, lam_1=0.01, lam_2=1, fit_parabola=subtract_parabola)[0]
    plt.plot(fit_baseline, ls=next(linestyles), label=f'parabola subtracted: {subtract_parabola}')
plt.plot(baseline, ':', label='true baseline')
plt.legend()

# %%
# For the baseline that is zero only on the left side, the non-processed data now incorrectly
# fits the data on the right side due to the non-zero value. The parabola-subtracted data,
# however, fits the data well on both ends and gives the expected baseline.
y, baseline = make_data(x, baseline_type=1)
plt.figure()
plt.plot(y)
for subtract_parabola in (False, True):
    fit_baseline = beads(y, lam_0=0.015, lam_1=0.1, lam_2=1, fit_parabola=subtract_parabola)[0]
    plt.plot(fit_baseline, ls=next(linestyles), label=f'parabola subtracted: {subtract_parabola}')
plt.plot(baseline, ':', label='true baseline')
plt.legend()

# %%
# The parabola subtraction method works well even for more complicated baselines, as seen below.
y, baseline = make_data(x, baseline_type=2)
plt.figure()
plt.plot(y)
for subtract_parabola in (False, True):
    fit_baseline = beads(
        y, lam_0=0.00006, lam_1=0.00008, lam_2=0.05, fit_parabola=subtract_parabola, tol=1e-3,
        freq_cutoff=0.04, asymmetry=3
    )[0]
    plt.plot(fit_baseline, ls=next(linestyles), label=f'parabola subtracted: {subtract_parabola}')
plt.plot(baseline, ':', label='true baseline')
plt.legend()

plt.show()
