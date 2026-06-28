# -*- coding: utf-8 -*-
"""
Parameter Selection for `beads`
-------------------------------

This example examines the parameter selection criteria for :meth:`~pybaselines.Baseline.beads`
proposed by `Navarro-Huerta, J.A., et al. "Assisted baseline subtraction in complex chromatograms
using the BEADS algorithm" <https://doi.org/10.1016/j.chroma.2017.05.057>`_. Note that the
preprocessing of data using a parabola before performing `beads` from the same paper is examined
in :ref:`another example <sphx_glr_generated_examples_misc_plot_beads_preprocessing.py>`.

The proposed parameter selection process works by varying the parameter of interest (e.g.
`freq_cutoff`, `lam_0`, `lam_1`, etc.) and calculating the autocorrelation coefficient,
:math:`r^2`, from the residual, ``data - baseline``, for each fit. The autocorrelation plots
will show plateau regions in which the calculated baseline is fairly consistent with regards
to the varying parameter. In theory, the optimal parameter value will then lie near the center of
one of these plateaus (often the final one) before :math:`r^2` approaches 0 (or some constant,
minimum value).

The example given here is fairly simple and illustrates how this parameter selection can be
done visually; for a more in-depth analysis of automating this procedure for real chromatograms,
see the `Weaselytics package <https://github.com/physicien/Weaselytics>`_.

"""
# sphinx_gallery_thumbnail_number = 3
# sphinx_gallery_multi_image = "single"

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np

from pybaselines import Baseline
from pybaselines.utils import make_data


def autocorrelation(residual):
    """Computes the autocorrelation of the residual following Navarro-Huerta, J.A., et al."""
    diff_residual = np.diff(residual)
    durbin_watson = (diff_residual @ diff_residual) / (residual @ residual)
    return 0.25 * (2 - durbin_watson)**2


x, y, baseline = make_data(return_baseline=True)
baseline_fitter = Baseline(x)

# %%
# Before varying any parameters, the input data will first be transformed by taking
# the log of the input. The benefit of this transformation for `beads` is twofold:
#
# * As noted by Navarro-Huerta, J.A., et al., taking the log transform of the data
#   helps to reduce ripples in the baseline when peaks have a significant magnitude
#   difference compared to the baseline.
#
# * The regularization parameters for `beads` penalize the L1 norms, so they are not
#   scale-invariant, which means that fitting ``y`` and ``100 * y`` would require changing
#   the various `lam_d` parameters. Taking the log transform helps to put all data on the same
#   relative scale such that the necessary regularization parameters between different
#   datasets will not vary as much as long as other factors remain constant (e.g. the number
#   of points, baseline curvature, etc.).
min_y = y.min()
y_transf = np.log(y - min_y + 1)  # offset to ensure all values > 0 to prevent issues with log

# %%
# The first parameter to vary will be the cutoff frequency, `freq_cutoff`, since it typically
# has the most influence on the baseline fit. A general rule of thumb is a higher cutoff
# frequency is required for more flexible baselines; in other words, a sinusoidal baseline
# would required a higher `freq_cutoff` than a linear baseline.
#
# The various regularization parameters `lam_d` will be initially fixed, with their values set
# according to the L1 norm of each d'th derivative, as suggested by the `original BEADS paper
# <https://doi.org/10.1016/j.chemolab.2014.09.014>`_.
lam_0 = 1 / abs(y_transf).sum()  # np.diff(y_transf, 0) == y_transf
lam_1 = 1 / abs(np.diff(y_transf, 1)).sum()
lam_2 = 1 / abs(np.diff(y_transf, 2)).sum()

plt.figure()
plt.plot(y)
r_squares = []
min_mse = np.inf
cutoff_freqs = np.geomspace(1e-3, 0.4, 30)
for i, cutoff_freq in enumerate(cutoff_freqs):
    fit_transf, params = baseline_fitter.beads(
        y_transf, freq_cutoff=cutoff_freq, lam_0=lam_0, lam_1=lam_1, lam_2=lam_2
    )
    fit = np.exp(fit_transf) + min_y - 1
    r_squares.append(autocorrelation(y - fit))
    if i % 6 == 0:
        plt.plot(fit, '--', label=f'cutoff_freq={cutoff_freq:.2e}')
    mse = ((fit - baseline)**2).mean()
    if mse < min_mse:
        best_index = i
        min_mse = mse
        best_fit = fit

plt.legend()

plt.figure()
plt.plot(y)
plt.plot(best_fit, '--', label='best fit')
plt.plot(baseline, ':', label='true baseline')
plt.legend()

fig, ax = plt.subplots()
ax.plot(cutoff_freqs, r_squares, 'o-')
ax.plot(cutoff_freqs[best_index], r_squares[best_index], 'ro', label='best fit')
plt.legend()

inset_ax = ax.inset_axes(
    [0.15, 0.3, 0.5, 0.5], xlim=[8e-4, 0.07], ylim=[0.97, 1.02]
)
inset_ax.plot(cutoff_freqs, r_squares, 'o-')
inset_ax.plot(cutoff_freqs[best_index], r_squares[best_index], 'ro')
mark_inset(ax, inset_ax, loc1=1, loc2=2, alpha=0.5)
ax.semilogx()
inset_ax.semilogx()
ax.set_ylabel(r'Autocorrelation, $r^2$')
ax.set_xlabel('Frequency cutoff')

print(f'Best cutoff frequency: {cutoff_freqs[best_index]:.2e}')

plt.show()
# %%
# Visual inspection of the autocorrelation plot above suggests that the best cutoff frequency,
# selected as the middle of the last plateau, occurs at ~0.01, which matches with the cutoff
# frequency that produced the lowest mean squared error with the known baseline, labeled as
# "best fit" in the plots above.
#
# As noted at the start of this example, this same procedure of finding plateaus on the
# autocorrelation plot can be repeated to select appropriate values for other parameters
# for `beads`, including `lam_0`, `lam_1`, `lam_2`, and `asymmetry`.
