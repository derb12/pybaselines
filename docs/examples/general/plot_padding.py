# -*- coding: utf-8 -*-
"""
Padding data
------------

Several baseline algorithms, notably :doc:`smoothing <../../../user_guide/algorithms/algorithms_1d/smooth>`
and :doc:`morphological <../../../user_guide/algorithms/algorithms_1d/morphological>` algorithms, pad the
input data in order to reduce edge effects from calculations. Padding is performed by
:func:`~.pad_edges`, which is a thin wrapper around :func:`numpy.pad` that also allows linear
extrapolation.

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

from pybaselines.utils import gaussian, pad_edges


half_window = 80  # for 161 point moving average window
num_points = 1000
x = np.linspace(0, 1000, num_points)
# use an exponentially decaying line to get a large edge effect one
# one side and a small edge effect on the other
line = 10 * np.exp(-x / 150)
noise = np.random.default_rng(0).normal(0, 0.5, num_points)
y = line + noise
pad_len = 2 * half_window + 1

plt.plot(pad_edges(y, pad_len, mode='constant', constant_values=np.nan), label='original')
for i, pad_mode in enumerate(('reflect', 'edge', 'extrapolate'), 1):
    padded_y = pad_edges(y, pad_len, pad_mode, extrapolate_window=int(0.1 * num_points))
    # offset each plot to show differences
    plt.plot(padded_y + 5 * i, label=pad_mode)

plt.legend()

# %%
# To show the effects that padding can have, the data is smoothed with a moving
# average. Poor padding produces more severe edge effects from the smoothing.
plt.figure()
plt.plot(y, label='original')
for i, pad_mode in enumerate(('reflect', 'edge', 'extrapolate'), 1):
    padded_y = pad_edges(y, pad_len, pad_mode, extrapolate_window=int(0.1 * num_points))
    plt.plot(
        uniform_filter1d(padded_y, 2 * half_window + 1)[pad_len:num_points + pad_len],
        label=pad_mode
    )

plt.legend()

# %%
# For the ``'extrapolate'`` padding mode, a list/tuple of two separate
# values can be specified for `extrapolate_window`, giving more control
# over the padding. This is useful when edge artifacts would affect the
# linear extrapolation, as demonstrated below.

# Add a peak near the right edge to make extrapolation harder
line_2 = 5 * np.exp(-x / 200) + gaussian(x, 5, 900, 20) + gaussian(x, 5, 200, 20)
noise_2 = np.random.default_rng(0).normal(0, 0.1, num_points)
y_2 = line_2 + noise_2

_, ax = plt.subplots()

patch = ax.plot(pad_edges(y_2, pad_len, mode='constant', constant_values=np.nan))[0]
extrapolate_windows = (1, 100, (100, 40))
legend = [[patch], ['original']]
for i, extrapolate_window in enumerate(extrapolate_windows, 1):
    padded_y = pad_edges(y_2, pad_len, extrapolate_window=extrapolate_window)
    # offset each plot to show differences
    padded_y += 5 * i
    patch = ax.plot(padded_y)[0]
    legend[0].append(patch)
    legend[1].append(f'extrapolate_window={extrapolate_window}')

    # show only the section used for extrapolation
    padded_y[:pad_len] = np.nan
    padded_y[-pad_len:] = np.nan
    if isinstance(extrapolate_window, int):
        windows = (extrapolate_window, extrapolate_window)
    else:
        windows = extrapolate_window
    padded_y[pad_len + windows[0]:-pad_len - windows[1]] = np.nan
    section_patch = ax.plot(padded_y, 'c.')[0]

ax.set_ylim([None, 31])
legend[0].insert(1, section_patch)
legend[1].insert(1, 'fit points')
ax.legend(*legend)

plt.show()
