# -*- coding: utf-8 -*-
"""
Classification masks
--------------------

The baseline algorithms in the :mod:`~pybaselines.classification` module estimate the baseline
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
from scipy.interpolate import BSpline

from pybaselines import Baseline
from pybaselines.utils import gaussian, pspline_smooth


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
baseline = np.sin(x / 100) * 2 + 10
noise = np.random.default_rng(0).normal(0, 0.1, x.size)
y = signal + baseline + noise

baseline_fitter = Baseline(x_data=x)

# %%

half_window_1 = 15
fit, params = baseline_fitter.std_distribution(y, half_window_1, smooth_half_window=10)

mask = params['mask']
x_interp = x[mask]
y_interp = y[mask]

plt.plot(x, y)
plt.plot(x, fit)
plt.plot(x_interp, y_interp, 'o')


# tck denotes the spline knots, coefficients, and degree
_, spline_tck = pspline_smooth(y_interp, x_interp, lam=10, num_knots=50)
spline_baseline = BSpline(*spline_tck)(x)

_, ax = plt.subplots(1)
ax.plot(x, y)
ax.plot(x, fit, label='Linear Interpolation')
ax.plot(x, spline_baseline, label='P-Spline Interpolation')
ax.legend()

plt.show()
