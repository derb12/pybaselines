# -*- coding: utf-8 -*-
"""
Customized Baseline Correction
------------------------------

This example looks at the ingenious baseline correction method created
by Liland et al., :meth:`~pybaselines.Baseline.custom_bc`.

The :meth:`.custom_bc` method works exceedingly well for morphological
and smoothing baselines, since those methods typically depend directly
on the number of data points, and for Whittaker-smoothing-based methods,
since the `lam` value is :ref:`heavily dependant on the number of data
points <sphx_glr_generated_examples_whittaker_plot_lam_vs_data_size.py>`.

This example will examine the use of the optimizer method
:meth:`~pybaselines.Baseline.custom_bc` paired with the Whittaker-smoothing-based
method :meth:`~pybaselines.Baseline.arpls`

"""
# sphinx_gallery_thumbnail_number = 1

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline
from pybaselines.utils import gaussian


x = np.linspace(20, 1000, 1000)
signal = (
    + gaussian(x, 6, 240, 5)
    + gaussian(x, 8, 350, 11)
    + gaussian(x, 15, 400, 18)
    + gaussian(x, 6, 550, 6)
    + gaussian(x, 13, 700, 8)
    + gaussian(x, 9, 800, 9)
    + gaussian(x, 9, 880, 7)
)
baseline = 5 + 6 * np.exp(-(x - 40) / 30) + gaussian(x, 5, 1000, 300)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

baseline_fitter = Baseline(x_data=x)
# %%
# For certain types of data, there can often be a sharp change in the
# baseline within a small region, such as in Raman spectroscopy
# near a Raman shift of 0 or in XRD at low two-theta. This presents a
# significant challenge to baseline algorithms that fit a single "global"
# baseline such as Whittaker-smoothing-based methods.
#
# The majority of the data can be fit using a "stiff" baseline, but the anomolous region
# requires a more flexible baseline.  Plotting each of these two cases
# separately, it is apparent each fits its target region well, but combining the
# two into a single baseline is difficult.
lam_flexible = 1e2
lam_stiff = 5e5

flexible_baseline = baseline_fitter.arpls(y, lam=lam_flexible)[0]
stiff_baseline = baseline_fitter.arpls(y, lam=lam_stiff)[0]

plt.figure()
plt.plot(x, y)
plt.plot(x, flexible_baseline, label='Flexible baseline')
plt.plot(x, stiff_baseline, label='Stiff baseline')
plt.legend()

# %%
# The beauty of Liland's customized baseline correction method is that
# it allows fitting baselines that are stiff in some regions and flexible
# in others by simply truncating the data to increase the stiffness. The input
# ``lam`` value within `method_kwargs` should correspond to the most flexible region, and the
# truncation should begin close to where the stiff and flexible baselines
# overlap, which is at approximately x=160 from the above figure. A small `lam` value
# of 1e1 is used to then smooth the calculated baseline using Whittaker
# smoothing so that the two regions connect without any significant discontinuity.

crossover_index = np.argmin(abs(x - 160))
fit_baseline, params = baseline_fitter.custom_bc(
    y, 'arpls',
    regions=([crossover_index, None],),
    sampling=15,
    method_kwargs={'lam': lam_flexible},
    lam=1e1
)

plt.figure()
plt.plot(x, y)
plt.plot(x, fit_baseline, label='Fit baseline')
plt.plot(x, baseline, '--', label='True baseline')
plt.legend()

# %%
# Looking at the results, this method is able to accurately recreate the
# true data even though the two baselines have significantly different
# requirements for stiffness.

plt.figure()
plt.plot(x, y - baseline, label='True data')
plt.plot(x, y - fit_baseline, label='Baseline corrected')
plt.legend()


plt.show()
