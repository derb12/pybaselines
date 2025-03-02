====================
Polynomial Baselines
====================

Introduction
------------

A polynomial can be expressed as

.. math::

    p(x) = \beta_0 x^0 + \beta_1 x^1 + \beta_2 x^2 + ... + \beta_m x^m = \sum\limits_{j = 0}^m {\beta_j x^j}

where :math:`\beta` is the array of coefficients for the polynomial.

For regular polynomial fitting, the polynomial coefficients that best fit data
are gotten from minimizing the least-squares:

.. math:: \sum\limits_{i}^N w_i^2 (y_i - p(x_i))^2

where :math:`y_i` and :math:`x_i` are the measured data, :math:`p(x_i)` is
the polynomial estimate at :math:`x_i`, and :math:`w_i` is the weighting.

However, since only the baseline of the data is desired, the least-squares
approach must be modified. For polynomial-based algorithms, this is done
by 1) only fitting the data in regions where there is only baseline (termed
selective masking), 2) modifying the y-values being fit each iteration, termed
thresholding, or 3) penalyzing outliers.

.. _selective-masking-explanation:

Selective Masking
~~~~~~~~~~~~~~~~~

Selective masking is the simplest of the techniques. There
are two ways to use selective masking in pybaselines.

First, the input dataset can be trimmed/masked (easy to do with numpy) to not
include any peak regions, the masked data can be fit, and then the resulting
polynomial coefficients (must set ``return_coef`` to True) can be used to create
a polynomial that spans the entirety of the original dataset.

.. plot::
   :align: center
   :context: reset
   :include-source: True

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
    real_baseline = 5 + 15 * np.exp(-x / 400)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)
    y = signal + real_baseline + noise

    # bitwise "or" (|) and "and" (&) operators for indexing numpy array
    non_peaks = (
        (x < 150) | ((x > 210) & (x < 310))
        | ((x > 440) & (x < 650)) | (x > 840)
    )
    x_masked = x[non_peaks]
    y_masked = y[non_peaks]

    # fit only the masked x and y
    _, params = Baseline(x_masked).poly(y_masked, poly_order=3, return_coef=True)
    # recreate the polynomial using numpy and the full x-data
    baseline = np.polynomial.Polynomial(params['coef'])(x)

    # Alternatively, just use numpy:
    # baseline = np.polynomial.Polynomial.fit(x_masked, y_masked, 3)(x)

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    data_handle = ax.plot(y)
    baseline_handle = ax.plot(baseline, '--')
    masked_y = y.copy()
    masked_y[~non_peaks] = np.nan
    masked_handle = ax.plot(masked_y)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], masked_handle[0], baseline_handle[0]),
        ('data', 'non-peak regions', 'fit baseline'), frameon=False
    )
    plt.show()


The second way is to keep the original data, and input a custom weight array into the
fitting function with values equal to 0 in peak regions and 1 in baseline regions.

.. plot::
   :align: center
   :context: close-figs
   :include-source: True

    weights = np.zeros(len(y))
    weights[non_peaks] = 1
    # directly create baseline by inputting weights
    baseline = Baseline(x).poly(y, poly_order=3, weights=weights)[0]

    # Alternatively, just use numpy:
    # baseline = np.polynomial.Polynomial.fit(x, y, 3, w=weights)(x)

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    data_handle = ax.plot(y)
    baseline_handle = ax.plot(baseline, '--')
    masked_y = y.copy()
    masked_y[~non_peaks] = np.nan
    masked_handle = ax.plot(masked_y)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], masked_handle[0], baseline_handle[0]),
        ('data', 'non-peak regions', 'fit baseline'), frameon=False
    )
    plt.show()


As seen above, both ways produce the same resulting baseline, but the second way
(setting weights) is much easier and faster since the baseline is directly calculated.

The only algorithm in pybaselines that requires using selective masking is
:meth:`~.Baseline.poly`, which is normal polynomial least-squares fitting as described
above. However, all other polynomial techniques allow inputting custom weights
in order to get better fits or to reduce the number of iterations.

The use of selective masking is generally not encouraged since it is time consuming
to select the peak and non-peak regions in each set of data, and can lead to hard
to reproduce results.

.. _thresholding-explanation:

Thresholding
~~~~~~~~~~~~

Thresholding is an iterative method that first fits the data using
traditional least-squares, and then sets the next iteration's fit data
as the element-wise minimum between the current data and the current fit.
The figure below illustrates the iterative thresholding.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False

    fig, axes = plt.subplots(
        2, 2, gridspec_kw={'hspace': 0, 'wspace': 0},
        tight_layout={'pad': 0.2, 'w_pad': 0, 'h_pad': 0}
    )
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        baseline = np.polynomial.Polynomial.fit(x, y, 3)(x)
        data_handle = ax.plot(y, '-')
        baseline_handle = ax.plot(baseline, '--')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.annotate(f'iteration {i + 1}', (12, 10))

        y = np.minimum(y, baseline)

    axes[0].legend(
        (data_handle[0], baseline_handle[0]), ('data', 'fit baseline'),
        frameon=False
    )
    plt.show()


The algorithms in pybaselines that use thresholding are :meth:`~.Baseline.modpoly`,
:meth:`~.Baseline.imodpoly`, and :meth:`~.Baseline.loess` (if ``use_threshold`` is True).

Penalyzing Outliers
~~~~~~~~~~~~~~~~~~~

The algorithms in pybaselines that penalyze outliers are
:meth:`~.Baseline.penalized_poly`, which incorporate the penalty directly into the
minimized cost function, and :meth:`~.Baseline.loess` (if ``use_threshold`` is False),
which incorporates penalties by applying lower weights to outliers. Refer
to the particular algorithms below for more details.


Algorithms
----------

poly (Regular Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.poly` is simple least-squares polynomial fitting. Use selective
masking, as described above, in order to use it for baseline fitting.

Note that the plots below are just the least-squared polynomial fitting
of the data since masking is time-consuming.

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
        if i < 4:
            poly_order = i + 1
        else:
            poly_order = 1
        baseline, params = baseline_fitter.poly(y, poly_order=poly_order)
        ax.plot(baseline, 'g--')


modpoly (Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.modpoly` uses thresholding, as explained above, to iteratively fit a polynomial
baseline to data. `modpoly` is also sometimes called "ModPolyFit" in literature, and both
`modpoly` and `imodpoly` are sometimes referred to as "IPF" or "Iterative Polynomial Fit".

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
        baseline, params = baseline_fitter.modpoly(y, poly_order=poly_order, use_original=True)
        ax.plot(baseline, 'g--')


imodpoly (Improved Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.imodpoly` is an attempt to improve the modpoly algorithm for noisy data,
by including the standard deviation of the residual (data - baseline) when performing
the thresholding. The number of standard deviations included in the thresholding can
be adjusted by setting ``num_std``. `imodpoly` is also sometimes called "IModPolyFit" or "Vancouver Raman Algorithm" in literature,
and both `modpoly` and `imodpoly` are sometimes referred to as "IPF" or "Iterative Polynomial Fit".

.. note::
   If using a ``num_std`` of 0, imodpoly may still produce different results than modpoly
   due to their different exit criteria.


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
        baseline, params = baseline_fitter.imodpoly(y, poly_order=poly_order)
        ax.plot(baseline, 'g--')


penalized_poly (Penalized Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.penalized_poly` (sometimes referred to as "backcor" in literature) fits a
polynomial baseline to data using non-quadratic cost functions. Compared to the quadratic
cost function used in typical least-squares as discussed above, non-quadratic cost funtions
allow outliers above a user-defined threshold to have less effect on the fit. pentalized_poly
has three different cost functions:

* Huber
* truncated-quadratic
* Indec

In addition, each cost function can be either symmetric (to fit a baseline to data with
both positive and negative peaks) or asymmetric (for data with only positive or negative peaks).
The plots below show the symmetric and asymmetric forms of the cost functions.

.. plot::
   :align: center

    import numpy as np
    import matplotlib.pyplot as plt

    def huber(x, symmetric=True, threshold=1):
        out = np.empty_like(x)
        if symmetric:
            mask = np.abs(x) < threshold
        else:
            mask = x < threshold
        out[mask] = x[mask]**2
        out[~mask] = 2 * threshold * np.abs(x[~mask]) - threshold**2

        return out

    def truncated_quadratic(x, symmetric=True, threshold=1):
        out = np.empty_like(x)
        if symmetric:
            mask = np.abs(x) < threshold
        else:
            mask = x < threshold
        out[mask] = x[mask]**2
        out[~mask] = threshold**2

        return out

    def indec(x, symmetric=True, threshold=1):
        out = np.empty_like(x)
        if symmetric:
            mask = np.abs(x) < threshold
        else:
            mask = x < threshold
        out[mask] = x[mask]**2
        out[~mask] = (threshold**3 / (2 * np.abs(x[~mask]))) + (threshold**2) / 2

        return out

    x = np.linspace(-3, 3, 100)
    y = x * x
    s_huber = huber(x)
    a_huber = huber(x, False)
    s_tquad = truncated_quadratic(x)
    a_tquad = truncated_quadratic(x, False)
    s_indec = indec(x)
    a_indec = indec(x, False)

    fig, (ax, ax2) = plt.subplots(
        1, 2, gridspec_kw={'hspace': 0, 'wspace': 0},
        tight_layout={'pad': 0.6, 'w_pad': 0, 'h_pad': 0}
    )
    ax.plot(
        x, y, '-',
        x, s_huber, '--',
        x, s_tquad, '-.',
        x, s_indec, ':'
    )
    handles = ax2.plot(
        x, y, '-',
        x, a_huber, '--',
        x, a_tquad, '-.',
        x, a_indec, ':'
    )

    ax.axvline(1, ymax=0.7, color='black', linestyle=':')
    ax.axvline(-1, ymax=0.7, color='black', linestyle=':')
    ax.annotate('threshold', (0.3, 6.6))
    ax.set_title('Symmetric')
    ax.annotate('residual, y - baseline', (2, -1.5), annotation_clip=False)
    ax.set_ylabel('Contribution to cost function')

    ax2.legend(handles, ('quadratic', 'Huber', 'truncated-quadratic', 'Indec'), frameon=False)
    ax2.axvline(1, ymax=0.7, color='black', linestyle=':')
    ax2.set_yticks([])
    ax2.set_title('Asymmetric')

    plt.show()


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            cost_function = 'symmetric_truncated_quadratic'
            poly_order = 1
        else:
            cost_function = 'asymmetric_truncated_quadratic'
            poly_order = i + 1
        baseline, params = baseline_fitter.penalized_poly(
            y, poly_order=poly_order, threshold=1.2, cost_function=cost_function
        )
        ax.plot(baseline, 'g--')


loess (Locally Estimated Scatterplot Smoothing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.loess` (sometimes referred to as "rbe" or "robust baseline estimate" in literature)
is similar to `traditional loess/lowess <https://en.wikipedia.org/wiki/Local_regression>`_
but adapted for fitting the baseline. The baseline at each point is estimated by using
polynomial regression on the k-nearest neighbors of the point, and the effect of outliers
is reduced by iterative reweighting.

.. note::
   Although not its intended use, the loess function can be used for smoothing like
   "traditional loess", simply by settting ``symmetric_weights`` to True and ``scale`` to ~4.05.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            symmetric_weights = True
        else:
            symmetric_weights = False
        if i == 1:
            fraction = 0.55
            scale = 1.5  # reduce scale to lower the effect of grouped peaks
        else:
            fraction = 0.35
            scale = 3
        if i in (0, 4):
            poly_order = 1
        else:
            poly_order = 2

        baseline, params = baseline_fitter.loess(
            y, poly_order=poly_order, scale=scale, fraction=fraction,
            symmetric_weights=symmetric_weights
        )
        ax.plot(baseline, 'g--')


quant_reg (Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.quant_reg` fits a polynomial to the baseline using quantile regression.

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
        quantile = {0: 0.3, 1: 0.1, 2: 0.2, 3: 0.25, 4: 0.5}[i]
        baseline, params = baseline_fitter.quant_reg(
            y, poly_order=poly_order, quantile=quantile
        )
        ax.plot(baseline, 'g--')


goldindec (Goldindec Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.goldindec` fits a polynomial baseline to data using non-quadratic cost functions,
similar to :meth:`~.Baseline.penalized_poly`, except that it only allows asymmetric cost functions.
The optimal threshold value between quadratic and non-quadratic loss is iteratively optimized
based on the input `peak_ratio` value.

.. plot::
   :align: center
   :context: close-figs

    peak_ratios = [0.2, 0.6, 0.2, 0.2, 0.3]
    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            poly_order = 1
        else:
            poly_order = i + 1
        baseline, params = baseline_fitter.goldindec(
            y, poly_order=poly_order, peak_ratio=peak_ratios[i]
        )
        ax.plot(baseline, 'g--')
