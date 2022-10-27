===================
Optimizer Baselines
===================

The contents of :mod:`pybaselines.optimizers` contain algorithms that build
upon other baseline algorithms to improve their results.

Algorithms
----------

optimize_extended_range
~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`.optimize_extended_range` function is based on the `Extended Range
Penalized Least Squares (erPLS) method <https://doi.org/10.3390/s20072015>`_,
but extends its usage to all Whittaker-smoothing-based, polynomial, and spline algorithms.

In this algorithm, a linear baseline is extrapolated from the left and/or
right edges, Gaussian peaks are added to these baselines, and then the original
data plus the extensions are input into the indicated Whittaker or polynomial function.
An example of data with added baseline and Gaussian peaks is shown below.

.. _extending-data-explanation:

.. plot::
   :align: center

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian, _get_edges

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 6, 550, 5)
        + gaussian(x, 9, 800, 10)
        + gaussian(x, 9, 100, 12)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 880, 8)
    )
    baseline = 5 + 10 * np.exp(-x / 800)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)
    y = signal * 0.5 + baseline + noise

    # parameters that define the added baseline and gaussian peaks
    width_scale = 0.2
    height_scale = 0.5
    sigma_scale = 1 / 15
    added_window = int(x.shape[0] * width_scale)

    # the added baseline and guassian peaks
    added_left, added_right = _get_edges(y, added_window, extrapolate_window=30)
    added_gaussian = gaussian(
        np.linspace(-added_window / 2, added_window / 2, added_window),
        height_scale * y.max(), 0, added_window * sigma_scale
    )
    added_data = np.hstack(
        (added_gaussian + added_left, y, added_gaussian + added_right)
    )

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    added_data_handle = ax.plot(added_data, 'g--')
    # add nan values to match the shape of added_data
    data_handle = ax.plot(
        np.hstack((np.full(added_window, np.nan), y, np.full(added_window, np.nan)))
    )
    added_baseline_handle = ax.plot(
        np.hstack((added_left, np.full(y.shape[0], np.nan), added_right)), ':'
    )

    ax.legend(
        (data_handle[0], added_baseline_handle[0], added_data_handle[0]),
        ('data', 'added baseline', 'added data'), frameon=False
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


A range of ``lam`` or ``poly_order`` values are tested, and the value that best fits the
added linear regions is selected as the optimal parameter.

.. plot::
   :align: center
   :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import Baseline


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

        baselines = (baseline_1, baseline_2, baseline_3, baseline_4, baseline_5)
        data = (y1, y2, y3, y4, y5)

        return x, data, baselines


    def create_plots(data=None, baselines=None):
        fig, axes = plt.subplots(
            3, 2, tight_layout={'pad': 0.1, 'w_pad': 0, 'h_pad': 0},
            gridspec_kw={'wspace': 0, 'hspace': 0}
        )
        axes = axes.ravel()

        legend_handles = []
        if data is None:
            plot_data = False
            legend_handles.append(None)
        else:
            plot_data = True
        if baselines is None:
            plot_baselines = False
            legend_handles.append(None)
        else:
            plot_baselines = True

        for i, axis in enumerate(axes):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.tick_params(
                which='both', labelbottom=False, labelleft=False,
                labeltop=False, labelright=False
            )
            if i < 5:
                if plot_data:
                    data_handle = axis.plot(data[i])
                if plot_baselines:
                    baseline_handle = axis.plot(baselines[i], lw=2.5)
        fit_handle = axes[-1].plot((), (), 'g--')
        if plot_data:
            legend_handles.append(data_handle[0])
        if plot_baselines:
            legend_handles.append(baseline_handle[0])
        legend_handles.append(fit_handle[0])

        if None not in legend_handles:
            axes[-1].legend(
                (data_handle[0], baseline_handle[0], fit_handle[0]),
                ('data', 'real baseline', 'estimated baseline'),
                loc='center', frameon=False
            )

        return fig, axes, legend_handles


    x, data, baselines = create_data()
    baseline_fitter = Baseline(x, check_finite=False)

    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        baseline, params = baseline_fitter.optimize_extended_range(
            y, method='aspls', height_scale=0.1, pad_kwargs={'extrapolate_window': 30}
        )
        ax.plot(baseline, 'g--')


collab_pls (Collaborative Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.collab_pls` is intended for fitting multiple datasets of related data,
and can use any Whittaker-smoothing-based or spline method. The general idea is that using
multiple sets of data should be better able to estimate the overall baseline rather
than individually fitting each set of data.

There are two ways the collab_pls function can fit datasets. The dataset can be averaged
and then fit once with the selected method, and then the output weights are used to
individually fit each set of data. The other method individually fits each set of data,
averages the weighting, and then uses the averaged weights to individually fit each set
of data. The figure below shows the comparison of the baselines fit by the collab_pls
algorithm versus the individual baselines from the mpls method.

.. plot::
   :align: center
   :context: close-figs

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 800, 10)
    )
    baseline = 5 + 15 * np.exp(-x / 400)
    noise = np.random.default_rng(0).normal(0, 0.2, x.size)
    y = signal + baseline + noise
    y2 = signal * 1.3 + baseline * 2 + noise
    y3 = signal * 0.5 + baseline * 0.5 + noise * 3
    y4 = signal + baseline * 3 - 2 + noise * 2
    total_y = (y, y2, y3, y4)

    lam = 1e5
    fit_baselines = baseline_fitter.collab_pls(total_y, method='mpls', method_kwargs={'lam': lam})[0]

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    for y_values in total_y:
        data_handle = ax.plot(y_values, 'C0')  # C0 is first item in color cycle
    for baseline in fit_baselines:
        baseline_handle = ax.plot(baseline, 'g--')
    for y_values in total_y:
        individual_fit_handle = ax.plot(baseline_fitter.mpls(y_values, lam=lam)[0], 'r:')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], baseline_handle[0], individual_fit_handle[0]),
        ('data', 'collab_pls fits', 'individual fits'), frameon=False
    )
    plt.show()


There is no figure showing the fits for various baseline types for this method
since it requires multiple sets of data for each baseline type.

adaptive_minmax (Adaptive MinMax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.adaptive_minmax` uses two different polynomial orders and two different
weighting schemes to create a total of four fits. The polynomial order(s) can be
specified by the user, or else they will be estimated by the signal-to-noise
ratio of the data. The first weighting scheme is either all points weighted
equally or using user-specified weights. The second weighting scheme places
a much higher weight on points near the two ends of the data to provide better
fits in certain circumstances.

Each of the four fits uses :ref:`thresholding <thresholding-explanation>`
(the "min" part of the name) to estimate the baseline. The final baseline is
then computed as the element-wise maximum of the four fits (the "max" part of
the name).

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i < 4:
            poly_order = i + 1
        else:
            poly_order = 1
        baseline, params = baseline_fitter.adaptive_minmax(y, poly_order=poly_order, method='imodpoly')
        ax.plot(baseline, 'g--')
