# -*- coding: utf-8 -*-
"""
Classification masks
--------------------

The baseline algorithms in the :mod:`.classification` module estimate the baseline
by classifying each point as belonging to either the baseline or the peaks. When
first using a function, the correct parameters may not be known. To make the effects
of input parameters on the classification process more easily understood, all functions
in the classification module provide a 'mask' item in the output parameter dictionary.
The mask parameter is a boolean numpy array that is True for any point classified as
belonging to the baseline and False otherwise.

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline
from pybaselines.utils import gaussian


half_window = 50
x = np.linspace(0, 1000, 1000)
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
baseline = gaussian(x, -6, 700, 500)
noise = np.random.default_rng(0).normal(0, 0.1, x.size)
y = signal + baseline + noise

baseline_fitter = Baseline(x_data=x)

# %%
# When first fitting a new dataset, it may be difficult to estimate the correct
# parameters. For this example, the main parameter for the baseline function is
# the `half_window` used for the rolling standard deviation calculation. Try a low
# and high value to see the difference.

half_window_1 = 15
half_window_2 = 45
fit_1, params_1 = baseline_fitter.std_distribution(y, half_window_1, smooth_half_window=10)
fit_2, params_2 = baseline_fitter.std_distribution(y, half_window_2, smooth_half_window=10)

plt.plot(x, y)
plt.plot(x, fit_1, label=f'half_window={half_window_1}')
plt.plot(x, fit_2, '--', label=f'half_window={half_window_2}')
plt.legend()

# %%
# The two baselines are similar in most regions except for the two small peaks.
# To investigate why such different results were obtained, the `mask` item in
# the output parameter dictionary can be used.
mask_1 = params_1['mask']
mask_2 = params_2['mask']

_, (ax, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
ax.plot(x, y)
patch_1 = ax.plot(x[mask_1], y[mask_1], 'o')[0]
ax2.plot(x, y)
patch_2 = ax2.plot(x[mask_2], y[mask_2], 'ms')[0]
ax.legend((patch_1, patch_2), (f'half_window={half_window_1}', f'half_window={half_window_2}'))

plt.show()

# %%
# After comparing the two masks, it is clear that the higher half_window value
# mis-identified the small peaks as belonging to the baseline. Thus, the smaller
# half_window is the better parameter.
