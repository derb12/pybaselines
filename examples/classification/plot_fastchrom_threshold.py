# -*- coding: utf-8 -*-
"""
fastchrom threshold
-------------------

:func:`.fastchrom` classifies baseline points based on their rolling standard
deviation value.
The default threshold for fastchrom is set to the fifteenth percentile of the rolling
standard deviation distribution. This default is rather conservative in assigning
baseline points, but was selected since it provides good results for many different
inputs.

Starting in version 0.7.0, the fastchrom function allows a callable object to be input
as the threshold, so that the user may determine the threshold based on the rolling
standard deviation distribution. This example shows how to specify custom thresholds
and investigate their effects.

Note: the author has found that the triangle threshold method, available from scikit-image,
typically performs quite well as the thresholding method for fastchrom due to the shape
of the rolling standard deviation distribution.

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from pybaselines.classification import fastchrom, _padded_rolling_std
from pybaselines.utils import gaussian
try:
    from skimage.filters import threshold_triangle as custom_threshold
except ImportError:
    # median works okay for this simple example if scikit-image is not installed
    custom_threshold = lambda values: np.median(values)


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
baseline = gaussian(x, 6, 400, 500)
noise = np.random.default_rng(0).normal(0, 0.2, len(x))
y = signal + baseline + noise

# %%
# Three thresholds will be compared: the default, a fixed threshold of 1.5, and
# a user-specified thresholding function.
fixed_threshold = 1.5
half_window = 15

fit_1, params_1 = fastchrom(y, x, half_window)
fit_2, params_2 = fastchrom(y, x, half_window, threshold=fixed_threshold)
fit_3, params_3 = fastchrom(y, x, half_window, threshold=custom_threshold)

plt.plot(y)
plt.plot(fit_1, label='default')
plt.plot(fit_2, '--', label='fixed')
plt.plot(fit_3, ':', label='custom')

plt.legend()

# %%
# From the plot, it is clear that the fixed threshold is too high since it classifies
# some of the peaks as baseline. The default and custom thresholds produce similar results.
#
# To further investigate, the `mask` item in the output parameter dictionaries can be
# used. The `mask` items are boolean numpy arrays, with values of True at baseline
# points and values of False at peak points. After plotting the masks, it is clear
# that the custom threshold was best able to identify the baseline regions. The default
# method did okay, but identified many regions of the baseline as peaks.
mask_1 = params_1['mask']
mask_2 = params_2['mask']
mask_3 = params_3['mask']

_, (ax, ax2, ax3) = plt.subplots(3, sharex=True, gridspec_kw={'hspace': 0})
ax.plot(x, y)
patch_1 = ax.plot(x[mask_1], y[mask_1], 'o')[0]
ax2.plot(x, y)
patch_2 = ax2.plot(x[mask_2], y[mask_2], 'ms')[0]
ax3.plot(x, y)
patch_3 = ax3.plot(x[mask_3], y[mask_3], 'c.')[0]
ax.legend((patch_1, patch_2, patch_3), ('default', 'fixed', 'custom'))

# %%
# For an even better understanding of how the threshold is applied, look
# at the rolling standard deviation of the data, and at the resulting standard
# deviation distribution. From the two plots, it is clear that the custom
# threshold works the best since it sets the threshold for baseline points
# high enough to account for slight variation in the baseline standard deviation,
# while remaining low enough to not misidentify any peak regions.
rolling_std = _padded_rolling_std(y, half_window, ddof=1)

plt.figure()
plt.plot(rolling_std)
plt.axhline(np.percentile(rolling_std, 15), color='orange', label='default')
plt.axhline(fixed_threshold, color='r', ls='--', label='fixed')
plt.axhline(custom_threshold(rolling_std), color='g', ls=':', label='custom')
plt.ylabel('Standard Deviation')
plt.legend()

plt.figure()
# use 256 bins to match the default number of bins used by threshold_triangle
plt.hist(rolling_std, 256)
plt.axvline(np.percentile(rolling_std, 15), color='orange', label='default')
plt.axvline(fixed_threshold, color='r', ls='--', label='fixed')
plt.axvline(custom_threshold(rolling_std), color='g', ls=':', label='custom')
plt.xlabel('Standard Deviation')
plt.ylabel('Counts')
plt.legend()

plt.show()
