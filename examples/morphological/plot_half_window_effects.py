# -*- coding: utf-8 -*-
"""
`half_window` effects
---------------------

This example shows the influence of the `half_window` parameter that is used when
fitting any morphological algorithm.

For this example, the :func:`.mor` algorithm will be used, which is a relatively
robust baseline algorithm.

"""
# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np

from pybaselines.morphological import mor
from pybaselines.utils import gaussian


x = np.linspace(0, 1000, 2000)
signal = (
    gaussian(x, 9, 100, 12)
    + gaussian(x, 6, 180, 5)
    + gaussian(x, 8, 300, 11)
    + gaussian(x, 15, 400, 12)
    + gaussian(x, 6, 550, 6)
    + gaussian(x, 13, 700, 8)
    + gaussian(x, 9, 800, 9)
    + gaussian(x, 9, 880, 7)
)
baseline = 5 + gaussian(x, 10, 650, 150)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

# %%
# For many morphology-based baseline algorithms, the optimal `half_window` value
# is approximately equal to the full-width-at-half-maximum (FWHM) of the widest
# peak. We can calculate the FWHM of our synthetic data from the largest :math:`\sigma`
# value. The FWHM of a Gaussian distribution is related to its :math:`\sigma` value by
# the relationship: :math:`FWHM = 2 \sigma \sqrt{2 ln{2}} \approx 2.3548 \sigma`.
#
# The spacing of the x-values, :math:`dx` also has to be taken into account since the
# `half_window` value is index-based, while :math:`\sigma` is based on the x-values.
# Thus, the approximate FWHM is: :math:`FWHM \approx 2.3548 \sigma / dx`. Also note
# that `half_window` has to be an integer since it is index-based.
dx = np.diff(x).mean()  # set dx as the average of all the x-spacings
print(int(2.3548 * 12 / dx))

# %%
# Thus, a good `half_window` value is ~60. To investigate the effect of the
# `half_window` value, we will use 30, 60, and 120.
#
# Note that the actual :math:`\sigma` value for a peak is often unknown, so
# the FWHM value is usually estimated simply by looking at a plot of the data
# (using plt.plot(y)).
#

# %%
# Using a small `half_window` value makes the baseline cut into the larger peaks.
plt.figure()
plt.plot(y, label='data')
half_window = 30
plt.plot(mor(y, half_window=half_window)[0], label=f'half_window={half_window}')
plt.legend()


# %%
# Setting the `half_window` value as the approximate FWHM now fits the
# expected baseline without reducing the peak area.
plt.figure()
plt.plot(y, label='data')
half_window = 60
plt.plot(mor(y, half_window=half_window)[0], label=f'half_window={half_window}')
plt.legend()


# %%
# Further increasing the `half_window` value then makes the baseline unable
# to follow the curve of the data.
plt.figure()
plt.plot(y, label='data')
half_window = 120
plt.plot(mor(y, half_window=half_window)[0], label=f'half_window={half_window}')
plt.legend()

# %%
# Now, put together all the results to show the change in the baseline
# as `half_window` increases. Note how the `half_window` value can be considered
# as the approximate "stiffness" of the baseline. For small `half_window` values,
# the baseline has more flexibility to fit inbetween the peaks, while the largest
# `half_window` value is too stiff to account for the localized curvature of the
# baseline.
plt.figure()
plt.plot(y, label='data')
for half_window in (30, 60, 120):
    plt.plot(mor(y, half_window=half_window)[0], label=f'half_window={half_window}')
plt.legend()

# %%
# Finally, note that the effect of the larger `half_window` value is decreased
# if the baseline is relatively flat, so there is less of a penalty for using a
# larger `half_window` value on flat baselines.
flat_baseline = 5 + gaussian(x, 10, 650, 400)
y = y - baseline + flat_baseline  # replace the old baseline with the flat baseline
plt.figure()
plt.plot(y, label='data')
for half_window in (30, 60, 120):
    plt.plot(mor(y, half_window=half_window)[0], label=f'half_window={half_window}')
plt.legend()

plt.show()
