# -*- coding: utf-8 -*-
"""
Using `individual_axes` for 1D Baseline Correction
--------------------------------------------------

This example will show how to apply one dimensional baseline correction to two
dimensional data using :meth:`~pybaselines.Baseline2D.individual_axes`. Note that this is valid
only if each baseline along the axis uses the same inputs; otherwise, the more appropriate
approach is to use a for-loop with the corresponding :class:`.Baseline` method.

"""
# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np

from pybaselines import Baseline2D
from pybaselines.utils import gaussian


def plot_contour_with_projection(X, Z, data):
    """Plots the countour plot and 3d projection."""
    fig = plt.figure(layout='constrained', figsize=plt.figaspect(0.5))
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.contourf(X, Z, data, cmap='coolwarm')
    ax_2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax_2.plot_surface(X, Z, data, cmap='coolwarm')

    ax_1.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax_1.set_ylabel('Temperature ($^o$C)')
    ax_2.set_xlabel('Raman Shift (cm$^{-1}$)')
    ax_2.set_ylabel('Temperature ($^o$C)')
    ax_2.set_zticks([])


def plot_1d(x, data):
    """Plots the data in only one dimension."""
    plt.figure()
    # reverse so that data for lowest temperatures is plotted first
    plt.plot(x, data[::-1].T)
    plt.xlabel('Raman Shift (cm$^{-1}$)')
    plt.ylabel('Intensity (Counts)')


# %%
# The data for this example will simulate Raman spectroscopy measurements that
# were taken while heating a sample. Within the sample, peaks for one specimen
# disappear as the temperature is raised, which could occur due to a chemical
# reaction, phase change, decomposition, etc. Further, as the temperature increases,
# the measured baseline slightly increases.
len_temperature = 25
wavenumber = np.linspace(50, 300, 1000)
temperature = np.linspace(25, 100, len_temperature)
X, T = np.meshgrid(wavenumber, temperature, indexing='ij')
noise_generator = np.random.default_rng(0)
data = []
for i, t_value in enumerate(temperature):
    signal = (
        gaussian(wavenumber, 11 * (1 - i / len_temperature), 90, 3)
        + gaussian(wavenumber, 12 * (1 - i / len_temperature), 110, 6)
        + gaussian(wavenumber, 13, 210, 8)
    )
    real_baseline = 100 + 0.005 * wavenumber + 0.0001 * (wavenumber - 120)**2 + 0.08 * t_value
    data.append(signal + real_baseline + noise_generator.normal(scale=0.1, size=wavenumber.size))
y = np.array(data)

plot_contour_with_projection(X, T, y.T)

# %%
# When considering the baseline of this data, it is more helpful to plot all measurements
# only considering the wavenumber dependence.
plot_1d(wavenumber, y)

# %%
# While the measured data is two dimensional, each baseline can be considered as
# only dependent on the wavenumbers and independent of every other measurement along the
# temperature axis. Thus, individual_axes can be called on just the axis corresponding
# to the wavenumbers (ie. axis 1, the columns), and the one dimensional
# :meth:`~pybaselines.Baseline.pspline_arpls` method will be applied to each spectra.
baseline_fitter = Baseline2D(temperature, wavenumber)
baseline, params = baseline_fitter.individual_axes(
    y, axes=1, method='pspline_arpls', method_kwargs={'lam': 1e4}
)

# %%
# Looking at the one dimensional representation, each spectrum was correctly baseline
# corrected.
plot_1d(wavenumber, y - baseline)

# %%
# Finally, looking at the two dimensional representation of the data again, the dependance
# of the intensity for each peak with temperature is more easily seen.
plot_contour_with_projection(X, T, (y - baseline).T)

plt.show()
