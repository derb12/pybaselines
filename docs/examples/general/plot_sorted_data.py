# -*- coding: utf-8 -*-
"""
Sorting Data
------------

Some types of characterization data, such as FTIR and XPS, are typically expressed
with their x-values in decending order. Rather than having to ensure
users rearrange their data into ascending order before using pybaselines,
the :class:`~.Baseline` object handles this internally and returns values in the
same order as the input x-values. This is especially important if other parameters
from the selected baseline method are desired such as weights, since those are likewise
returned in the same order as the input x-values.

"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline
from pybaselines.utils import gaussian


x = np.linspace(500, 4000, 1000)
signal = (
    + gaussian(x, 8, 650, 16)
    + gaussian(x, 9, 1100, 50)
    + gaussian(x, 8, 1350, 20)
    + gaussian(x, 11, 2800, 20)
    + gaussian(x, 8, 2900, 20)
    + gaussian(x, 5, 3400, 120)
)
baseline = 0.08 + 0.00004 * x + gaussian(x, 3, 3500, 1000)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

# reverse both x and y
x_reversed = x[::-1]
y_reversed = y[::-1]
lam = 1e6

fit, params = Baseline(x).iarpls(y, lam=lam)
fit_reversed, params_reversed = Baseline(x_reversed).iarpls(y_reversed, lam=lam)

_, (ax_1, ax_2) = plt.subplots(nrows=2, sharex=True)
ax_1.plot(x, y)
ax_1.plot(x, fit)
ax_2.plot(x_reversed, y_reversed)
ax_2.plot(x_reversed, fit_reversed)
ax_2.invert_xaxis()
ax_2.set_xlabel(r'Wavenumber (cm$^{-1}$)')
ax_1.set_ylabel('Absorbance (%)')
ax_2.set_ylabel('Absorbance (%)')

ax_1.set_title('Sorted Data')
ax_2.set_title('Reversed Data')

print(
    'Outputs same for sorted and unsorted baselines: ',
    np.allclose(fit, fit_reversed[::-1], rtol=1e-14, atol=1e-14)
)


# %%
# Likewise, relevant output parameters such as weights are also sorted to match
# the input x-values.

_, (ax_1, ax_2) = plt.subplots(nrows=2, sharex=True)
ax_1.plot(x, y)
ax_1.plot(x, fit)
ax_2.plot(x, params['weights'], 'ro', label='sorted')
ax_2.plot(x_reversed, params_reversed['weights'], 'b.', label='reversed')
ax_2.invert_xaxis()
ax_2.legend()

ax_2.set_xlabel(r'Wavenumber (cm$^{-1}$)')
ax_1.set_ylabel('Absorbance (%)')
ax_2.set_ylabel('Weights')

print(
    'Weights same for sorted and unsorted baselines: ',
    np.allclose(params['weights'], params_reversed['weights'][::-1], rtol=1e-14, atol=1e-14)
)

plt.show()
