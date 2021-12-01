# -*- coding: utf-8 -*-
"""
Algorithm convergence
---------------------

Most iterative baselines in pybaselines that allow specifying the maximum
number of iterations (`max_iter`) and minimum tolerance (`tol`) will output
a `tol_history` item in the parameter dictionary, which is a numpy array of
the measured tolerance value at each iteration. The `tol_history` parameter
can be helpful for determining appropriate `max_iter` or `tol` values.

In this example, the convergence of the :func:`.asls` and :func:`.aspls` functions
will be compared. asls is a relatively simple calculation that sets its weighting
each iteration based on whether the current baseline is above or below the input data
at each point. aspls has a much more intricate weighting based on the logistic distribution
of the residuals (data minus baseline); further, aspls also updates an additional
parameter each iteration that controls the local stiffness of the baseline.

"""
# sphinx_gallery_thumbnail_number = 4

import matplotlib.pyplot as plt
import numpy as np

from pybaselines.utils import gaussian
from pybaselines.whittaker import asls, aspls


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
baseline = 5 + 10 * np.exp(-x / 600)

noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

lam = 5e6
tol = 1e-3
max_iter = 20
fit_1, params_1 = asls(y, lam=lam, tol=tol, max_iter=max_iter)
fit_2, params_2 = aspls(y, lam=lam, tol=tol, max_iter=max_iter)

plt.plot(y)
plt.plot(fit_1, label='asls')
plt.plot(fit_2, label='aspls')
plt.legend()

# %%
# Plotting the `tol_history` parameters for the two algorithms shows
# their differences. The asls algorithm converges quite quickly due to its
# simple weighting scheme. The aspls algorithm, however, converges quite slowly
# and erratically due to its more complicated updating.

plt.figure()
plt.plot(params_1['tol_history'], label='asls')
plt.plot(params_2['tol_history'], '--', label='aspls')
plt.axhline(tol, ls=':', color='k', label='tolerance')
plt.gca().set_yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Tolerance Value')
plt.legend()

# %%
# To see whether the functions converged in a non-visual manner, the `tol_history`
# parameter can be used. If the last entry in the `tol_history` array is less than the
# indicated tolerance value, then the function converged. The length of the
# `tol_history` array is the number of iterations completed.
for function_name, params in (('asls', params_1), ('aspls', params_2)):
    tol_hist = params["tol_history"]
    print(f'{function_name}, converged: {tol_hist[-1] < tol}, {len(tol_hist)} iterations')

# %%
# Now, try increasing the maximum number of iterations to see whether aspls
# converges.

max_iter = 100
fit_3, params_3 = asls(y, lam=lam, tol=tol, max_iter=max_iter)
fit_4, params_4 = aspls(y, lam=lam, tol=tol, max_iter=max_iter)

plt.figure()
plt.plot(y)
plt.plot(fit_3, label='asls')
plt.plot(fit_4, label='aspls')

plt.legend()

plt.figure()
plt.plot(params_3['tol_history'], label='asls')
plt.plot(params_4['tol_history'], '--', label='aspls')
plt.axhline(tol, ls=':', color='k', label='tolerance')
plt.gca().set_yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Tolerance Value')
plt.legend()

plt.show()

for function_name, params in (('asls', params_3), ('aspls', params_4)):
    tol_hist = params["tol_history"]
    print(f'{function_name}, converged: {tol_hist[-1] < tol}, {len(tol_hist)} iterations')
