===================
Smoothing Baselines
===================

The contents of :mod:`pybaselines.smooth` contain algorithms that use smoothing
to eliminate peaks and leave only the baseline.

.. note::
   The module pybaselines.smooth was named pybaselines.window until version 0.6.0.


.. note::
   The window size used for smoothing-based algorithms is index-based, rather
   than based on the units of the data, so proper conversions must be done
   by the user to get the desired window size.


Algorithms
----------

noise_median (Noise Median method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.noise_median` estimates the baseline as the median value within
a moving window. The resulting baseline is then smoothed by convolving with a Gaussian
kernel. Note that this method does not perform well for tightly-grouped peaks.

.. plot::
   :align: center
   :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import smooth

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
        noise = np.random.default_rng(1).normal(0, 0.2, x.size)
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
        if i == 1:
            half_window = 100
            smooth_half_window = 50
        else:
            half_window = 60
            smooth_half_window = 20
        baseline = smooth.noise_median(
            y, half_window, smooth_half_window=smooth_half_window, extrapolate_window=20
        )
        ax.plot(baseline[0], 'g--')


snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.snip` iteratively takes the element-wise minimimum of each value
and the average of the values at the left and right edge of a window centered
at the value. The size of the half-window is incrementally increased from 1 to the
specified maximum size, which should be set to approximately half of the
index-based width of the largest peak or feature.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            half_window = 28
        else:
            half_window = 17
        baseline = smooth.snip(y, half_window, extrapolate_window=20)
        ax.plot(baseline[0], 'g--')


A smoother baseline can be obtained from the snip function by setting ``decreasing``
to True, which reverses the half-window size range to start at the maximum size and end at 1.
Further, smoothing can optionally be performed to make the baseline better fit noisy
data. The baselines when using decreasing window size and smoothing is shown below.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            half_window = 29
        else:
            half_window = 17
        baseline = smooth.snip(
            y, half_window, decreasing=True, smooth_half_window=3, extrapolate_window=20
        )
        ax.plot(baseline[0], 'g--')


swima (Small-Window Moving Average)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.swima` iteratively takes the element-wise minimum of either the
data (first iteration) or the previous iteration's baseline and the data/previous baseline
smoothed with a moving average. The window used for the moving average smoothing is
incrementally increased to smooth peaks until convergence is reached.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 0:
            smooth_half_window = 11
        else:
            smooth_half_window = 5
        baseline = smooth.swima(y, smooth_half_window=smooth_half_window, extrapolate_window=20
        )
        ax.plot(baseline[0], 'g--')


ipsa (Iterative Polynomial Smoothing Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.ipsa` iteratively smooths the input data using a second-order
Savitzky–Golay filter until the exit criteria is reached.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            half_window = 150
        else:
            half_window = 50
        baseline = smooth.ipsa(y, half_window, extrapolate_window=20)
        ax.plot(baseline[0], 'g--')


ria (Range Independent Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.ria` first extrapolates a linear baseline from the left and/or
right edges of the data and adds Gaussian peaks to these baselines, similar to the
:ref:`optimize_extended_range <extending-data-explanation>` function, and
records their initial areas. The data is then iteratively smoothed using a
zero-order Savitzky–Golay filter (moving average) until the area of the extended
regions after subtracting the smoothed data from the initial data is close to
their starting areas.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            width_scale = 0.3
            half_window = 40
        else:
            width_scale = 0.12
            half_window = 30
        baseline = smooth.ria(
            y, half_window=half_window, width_scale=width_scale, extrapolate_window=20
        )
        ax.plot(baseline[0], 'g--')
