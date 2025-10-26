=======================
Miscellaneous Baselines
=======================

Introduction
------------

Miscellaneous algorithms are those that do not fit in existing categories
within pybaselines.

Algorithms
----------

interp_pts (Interpolation between points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.interp_pts` interpolates between input points using line segments
or splines of different orders. The function is mainly intended for usage
with user interfaces and is not encouraged otherwise.

.. note::
   Unlike most other algorithms in pybaselines, `interp_pts` only requires inputting
   the x-values of the data rather than the y-values.


.. plot::
   :align: center
   :include-source: False
   :show-source-link: True

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import Baseline

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 800, 10)
    )
    baseline = 5 + 15 * np.exp(-x / 400)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)
    y = signal + baseline + noise

    # (x, y) pairs representing each point
    points = (
        (0, 20), (140, 15.5), (250, 12.8),
        (540, 8.75), (750, 7.23), (1001, 6.25)
    )

    baseline_fitter = Baseline(x, check_finite=False)
    linear_baseline = baseline_fitter.interp_pts(x, points)[0]
    spline_baseline = baseline_fitter.interp_pts(x, points, interp_method='cubic')[0]

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    data_handle = ax.plot(x, y)
    baseline_handle = ax.plot(x, linear_baseline, '--')
    baseline_handle_2 = ax.plot(x, spline_baseline, ':')
    points_handle = ax.plot(*list(zip(*points)), 'ro')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], baseline_handle[0], baseline_handle_2[0], points_handle[0]),
        ('data', 'linear interpolation', 'cubic spline interpolation', 'anchor points'),
        frameon=False
    )
    plt.show()


There is no figure showing the fits for various baseline types for this method
since it solely depends on the user-defined anchor points.

beads (Baseline Estimation And Denoising with Sparsity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.beads` decomposes the input data into baseline and pure, noise-free signal by
modeling the baseline as a low pass filter and by considering the signal and its derivatives
as sparse. The minimized equation for calculating the pure signal is given as:

.. math::

    \frac{1}{2} ||H(y - s)||_2^2
    + \lambda_0 \sum\limits_{i}^{N} \theta(s_i)
    + \lambda_1 \sum\limits_{i}^{N - 1} \phi(\Delta^1 s_i)
    + \lambda_2 \sum\limits_{i}^{N - 2} \phi(\Delta^2 s_i)

where :math:`y` is the measured data, :math:`s` is the calculated pure signal,
:math:`H` is a high pass filter, :math:`\theta()` is a differentiable, symmetric
or asymmetric penalty function on the calculated signal, :math:`\Delta^1` and :math:`\Delta^2`
are :ref:`finite-difference operators <difference-matrix-explanation>` of order 1
and 2, respectively, and :math:`\phi()` is a differentiable, symmetric penalty function
approximating the L1 loss (mean absolute error) applied to the first and second derivatives
of the calculated signal.

The calculated baseline, :math:`v`, upon convergence of calculating the pure signal is given by:

.. math::

    v = y - s - H(y - s)

pybaselines version 1.3.0 introduced an optional simplification of the :math:`\lambda_0`,
:math:`\lambda_1`, :math:`\lambda_2` regularization parameter selection using the procedure
recommended by the BEADS manuscript through the addition of the parameter :math:`\alpha`.
Briefly, it is assumed that each :math:`\lambda_d` value is approximately proportional to some
constant :math:`\alpha` divided by the L1 norm of the d'th derivative of the input data such
that:

.. math::

    \lambda_0 = \frac{\alpha}{||y||_1},
    \lambda_1 = \frac{\alpha}{||y'||_1},
    \lambda_2 = \frac{\alpha}{||y''||_1}

Such a parametrization allows varying just :math:`\alpha`, as well as simplified usage
within optimization frameworks to find the best value, as shown by
`Bosten, et al. <https://doi.org/10.1016/j.chroma.2023.464360>`_

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: True

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

    fit_params = [
        (500, 6, 0.01),
        (0.01, 8, 0.08),
        (80, 8, 0.01),
        (0.2, 6, 0.04),
        (100, 1, 0.01)
    ]
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        alpha, asymmetry, freq_cutoff = fit_params[i]
        baseline, params = baseline_fitter.beads(
            y, freq_cutoff=freq_cutoff, alpha=alpha, asymmetry=asymmetry, tol=1e-3
        )
        ax.plot(baseline, 'g--')

The signal with both noise and baseline removed can also be obtained from the output
of the beads method by accessing the 'signal' key in the output parameters.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the second-to-top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        alpha, asymmetry, freq_cutoff = fit_params[i]
        baseline, params = baseline_fitter.beads(
            y, freq_cutoff=freq_cutoff, alpha=alpha, asymmetry=asymmetry, tol=1e-3
        )

        ax.clear()  # remove the old plots in the axis
        data_handle = ax.plot(y)
        signal_handle = ax.plot(params['signal'])

    axes[-1].clear()  # remove the old legend
    axes[-1].legend(
        (data_handle[0], signal_handle[0]),
        ('data', 'signal from beads'), loc='center', frameon=False
    )
