# -*- coding: utf-8 -*-
"""
`lam` effects
-------------

This example shows the influence of the `lam` parameter that is used when
fitting any algorithm that is based on Whittaker-smoothing. Note that the
exact `lam` values used in this example are unimportant, just the changes
in their scale.

For this example, the :func:`.arpls` algorithm will be used, which performs
well in the presence of noise.

"""
# sphinx_gallery_thumbnail_number = 5

import matplotlib.pyplot as plt
import numpy as np
from pybaselines.utils import gaussian
from pybaselines.whittaker import arpls


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
baseline = 5 + 10 * np.exp(-x / 800)
noise = np.random.default_rng(0).normal(0, 0.2, len(x))
y = signal + baseline + noise

# %%
# For extremely small `lam` values, the algorithm's output is more comparable
# with smoothing than baseline correction (which would be the desired result if
# using pure Whittaker smoothing).
plt.plot(y, label='data')
lam = 1
plt.plot(arpls(y, lam=lam)[0], label=f'lam={lam:.0f}')
plt.legend()

# %%
# Increasing the `lam` value produces results that more resemble the baseline. However,
# the baseline is still too flexible and cuts into the larger peaks.
plt.figure()
plt.plot(y, label='data')
lam = 1e3
plt.plot(arpls(y, lam=lam)[0], label=f'lam={lam:.0f}')
plt.legend()


# %%
# Increasing the `lam` value further produces the desired result and follows the
# expected baseline without reducing the peak area.
plt.figure()
plt.plot(y, label='data')
lam = 1e6
plt.plot(arpls(y, lam=lam)[0], label=f'lam={lam:.0f}')
plt.legend()


# %%
# Further increasing the `lam` value then makes the baseline too stiff and unable
# to follow the curve of the data.
plt.figure()
plt.plot(y, label='data')
lam = 1e10
plt.plot(arpls(y, lam=lam)[0], label=f'lam={lam:.0f}')
plt.legend()

# %%
# Finally, put together all the results to show the change in the baseline
# as `lam` increases.
plt.figure()
plt.plot(y, label='data')
for lam in (1, 1e3, 1e6, 1e10):
    plt.plot(arpls(y, lam=lam)[0], label=f'lam={lam:.0f}')
plt.legend()

plt.show()
