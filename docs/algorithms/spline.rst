================
Spline Baselines
================

The contents of :mod:`pybaselines.spline` contain algorithms for fitting
splines to the baseline.

Introduction
------------

To be added...


Algorithms
----------

mixture_model (Mixture Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.mixture_model` considers the data as a mixture model composed of
a baseline with noise and peaks. The weighting for the penalized spline fitting
the baseline is iteratively determined by fitting the residual with a normal
distribution centered at 0 (representing the noise), and a uniform distribution
for residuals >= 0 (and a third uniform distribution for residuals <= 0 if `symmetric`
is set to True) representing peaks. After fitting the total model to the residuals,
the weighting is calculated from the posterior probability for each value in the
residual belonging to the noise's normal distribution.

.. plot::
   :align: center
   :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import spline

    def create_plots():
        fig, axes = plt.subplots(
            3, 2, tight_layout={'pad': 0.1, 'w_pad': 0, 'h_pad': 0},
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )
        axes = axes.ravel()
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(
                which='both', labelbottom=False, labelleft=False,
                labeltop=False, labelright=False
            )
        return fig, axes

    def create_data():
        x = np.linspace(1, 1000, 500)
        signal = (
            gaussian(x, 6, 180, 5)
            + gaussian(x, 8, 350, 10)
            + gaussian(x, 6, 550, 5)
            + gaussian(x, 9, 800, 10)
        )
        signal_2 = (
            gaussian(x, 9, 100, 12)
            + gaussian(x, 15, 400, 8)
            + gaussian(x, 13, 700, 12)
            + gaussian(x, 9, 880, 8)
        )
        signal_3 = (
            gaussian(x, 8, 150, 10)
            + gaussian(x, 20, 120, 12)
            + gaussian(x, 16, 300, 20)
            + gaussian(x, 12, 550, 5)
            + gaussian(x, 20, 750, 12)
            + gaussian(x, 18, 800, 18)
            + gaussian(x, 15, 830, 12)
        )
        np.random.seed(1)  # set random seed
        noise = np.random.normal(0, 0.2, x.size)
        linear_baseline = 3 + 0.01 * x
        exponential_baseline = 5 + 15 * np.exp(-x / 400)
        gaussian_baseline = 5 + gaussian(x, 20, 500, 500)

        baseline_1 = linear_baseline
        baseline_2 = gaussian_baseline
        baseline_3 = exponential_baseline
        baseline_4 = 10 - 0.005 * x + gaussian(x, 5, 850, 200)
        baseline_5 = linear_baseline + 20

        y1 = signal * 2 + baseline_1 + 5 * noise
        y2 = signal + signal_2 + signal_3 + baseline_2 + noise
        y3 = signal + signal_2 + baseline_3 + noise
        y4 = signal + + signal_2 + baseline_4 + noise * 0.5
        y5 = signal * 2 - signal_2 + baseline_5 + noise

        baselines = baseline_1, baseline_2, baseline_3, baseline_4, baseline_5
        data = (y1, y2, y3, y4, y5)

        fig, axes = create_plots()
        for ax, y, baseline in zip(axes, data, baselines):
            data_handle = ax.plot(y)
            baseline_handle = ax.plot(baseline, lw=2.5)
        fit_handle = axes[-1].plot((), (), 'g--')
        axes[-1].legend(
            (data_handle[0], baseline_handle[0], fit_handle[0]),
            ('data', 'real baseline', 'estimated baseline'),
            loc='center', frameon=False
        )

        return axes, data

    for i, (ax, y) in enumerate(zip(*create_data())):
        if i in (0, 4):
            lam = 5e8
        elif i == 1:
            lam = 5e6
        else:
            lam = 1e5
        if i == 4:
            symmetric = True
            p = 0.5
        else:
            symmetric = False
            p = 0.01
        baseline = spline.mixture_model(y, lam=lam, p=p, symmetric=symmetric)
        ax.plot(baseline[0], 'g--')


irsqr (Iterative Reweighted Spline Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.irsqr` uses penalized splines and iterative reweighted least squares
to perform quantile regression on the data.

.. plot::
   :align: center
   :context: close-figs

    quantiles = {0: 0.3, 1: 0.1, 2: 0.2, 3: 0.25, 4: 0.5}
    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 0:
            lam = 1e7
        elif i == 1:
            lam = 1e6
        else:
            lam = 1e5
        baseline = spline.irsqr(y, lam=lam, quantile=quantiles[i])
        ax.plot(baseline[0], 'g--')


corner_cutting (Corner-Cutting Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.corner_cutting` iteratively removes corner points and then creates
a quadratic Bezier spline from the remaining points. Continuity between
the individual Bezier curves is maintained by adding control points halfway
between all but the first and last non-corner points.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i in (1, 3):
            max_iter = 11
        else:
            max_iter = 100

        baseline = spline.corner_cutting(y, max_iter=max_iter)
        ax.plot(baseline[0], 'g--')
