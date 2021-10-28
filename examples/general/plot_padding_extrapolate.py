# -*- coding: utf-8 -*-
"""
Padding with 'extrapolate' mode
-------------------------------

Since version 0.7.0 of pybaselines, a list/tuple of two separate
values can be specified for `extrapolate_window`, giving more control
over the padding.

"""

import matplotlib.pyplot as plt
import numpy as np
from pybaselines.utils import gaussian, pad_edges, _check_scalar


num_points = 1000
pad_len = 100

x = np.linspace(0, 1000, num_points)
# use an exponentially decaying line to get a large edge effect one
# one side and a small edge effect on the other
line = 5 * np.exp(-x / 200) + gaussian(x, 5, 900, 20) + gaussian(x, 5, 200, 20)
noise = np.random.default_rng(0).normal(0, 0.1, num_points)
y = line + noise

_, ax = plt.subplots()

patch = ax.plot(pad_edges(y, pad_len, mode='constant', constant_values=np.nan))[0]
extrapolate_windows = (1, 100, [100, 40])
legend = [[patch], ['original']]
for i, extrapolate_window in enumerate(extrapolate_windows, 1):
    padded_y = pad_edges(y, pad_len, extrapolate_window=extrapolate_window)
    # offset each plot to show differences
    padded_y += 5 * i
    patch = ax.plot(padded_y)[0]
    legend[0].append(patch)
    legend[1].append(f'extrapolate_window={extrapolate_window}')

    # show only the section used for extrapolation
    padded_y[:pad_len] = np.nan
    padded_y[-pad_len:] = np.nan
    windows = _check_scalar(extrapolate_window, 2, True)[0]
    padded_y[pad_len + windows[0]:-pad_len - windows[1]] = np.nan
    section_patch = ax.plot(padded_y, 'c.')[0]

ax.set_ylim([None, 31])
legend[0].insert(1, section_patch)
legend[1].insert(1, 'fit points')
ax.legend(*legend)
plt.show()
