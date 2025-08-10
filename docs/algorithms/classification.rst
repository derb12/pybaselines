========================
Classification Baselines
========================

Introduction
------------

Classification methods rely on classifying peak and/or baseline segments, similar to
:ref:`selective masking <selective-masking-explanation>` as explained in the polynomial
section, but make use of sophisticated techniques to determine the baseline
points rather than relying on manual selection.

All classification functions allow inputting weights to override the baseline classification,
which can be helpful, for example, to ensure a small peak is not classified as baseline without
having to alter any parameters which could otherwise reduce the effectiveness of the classification
method. The plot below shows such an example.

.. plot::
   :align: center

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import Baseline


    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 1, 180, 25)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 800, 10)
    )
    real_baseline = 5 + 5 * np.exp(-x / 400)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)
    y = signal + real_baseline + noise

    weights = np.ones(y.shape[0])
    # a weight of 0 designates that it is a peak; ensures the small peak at ~180 is
    # not classified as part of the baseline
    weights[(x > 100) & (x < 250)] = 0

    baseline_fitter = Baseline(x, check_finite=False)
    unweighted_baseline = baseline_fitter.std_distribution(y, half_window=10, num_std=1.3)[0]
    weighted_baseline = baseline_fitter.std_distribution(
        y, half_window=10, num_std=1.3, weights=weights
    )[0]

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    data_handle = ax.plot(y)
    baseline_handle = ax.plot(unweighted_baseline, '-')
    weighted_baseline_handle = ax.plot(weighted_baseline, '--')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], baseline_handle[0], weighted_baseline_handle[0]),
        ('data', 'unweighted baseline', 'weighted baseline'), frameon=False
    )
    plt.show()


Algorithms
----------

dietrich (Dietrich's Classification Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.dietrich` calculates the power spectrum of the data as the squared derivative
of the data. Then baseline points are identified by iteratively removing points where
the mean of the power spectrum is less a multiple of the standard deviation of the
power spectrum. The baseline is created by first interpolating through all baseline
points, and then iteratively fitting a polynomial to the interpolated baseline.

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
        if i == 1:
            num_std = 2.5
        else:
            num_std = 3
        baseline, params = baseline_fitter.dietrich(
            y, smooth_half_window=5, num_std=num_std, poly_order=poly_order, min_length=3
        )
        ax.plot(baseline, 'g--')


golotvin (Golotvin's Classification Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.golotvin` divides the data into sections and takes the minimum standard
deviation of all the sections as the noise's standard deviation for the entire data.
Then classifies any point where the rolling max minus min is less than a multiple of
the noise's standard deviation as belonging to the baseline.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 20
        else:
            half_window = 10
        if i in (1, 3):
            num_std = 40
        else:
            num_std = 10
        baseline, params = baseline_fitter.golotvin(y, half_window=half_window, num_std=num_std)
        ax.plot(baseline, 'g--')


std_distribution (Standard Deviation Distribution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.std_distribution` identifies baseline segments by analyzing the rolling
standard deviation distribution. The rolling standard deviations are split into two
distributions, with the smaller distribution assigned to noise. Baseline points are
then identified as any point where the rolled standard deviation is less than a multiple
of the median of the noise's standard deviation distribution.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 30
            num_std = 0.9
        elif i in (2, 3):
            half_window = 8
            num_std = 3.5
        else:
            half_window = 12
            num_std = 1.1
        baseline, params = baseline_fitter.std_distribution(y, half_window=half_window, num_std=num_std)
        ax.plot(baseline, 'g--')


fastchrom (FastChrom's Baseline Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.fastchrom` identifies baseline segments by analyzing the rolling standard
deviation distribution, similar to :meth:`~.Baseline.std_distribution`. Baseline points are
identified as any point where the rolling standard deviation is less than the specified
threshold, and peak regions are iteratively interpolated until the baseline is below the data.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            min_fwhm = y.shape[0]  # ensure it doesn't try to fill in negative peaks
        else:
            min_fwhm = None
        baseline, params = baseline_fitter.fastchrom(
            y, half_window=12, threshold=1, min_fwhm=min_fwhm
        )
        ax.plot(baseline, 'g--')


cwt_br (Continuous Wavelet Transform Baseline Recognition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.cwt_br` identifies baseline segments by performing a continous wavelet
transform (CWT) on the input data at various scales, and picks the scale with the first
local minimum in the Shannon entropy. The threshold for baseline points is obtained by fitting
a Gaussian to the histogram of the CWT at the optimal scale, and the final baseline is fit
using a weighted polynomial where identified baseline points are given a weight of 1 while all
other points have a weight of 0.


.. plot::
   :align: center
   :context: close-figs

    scales = np.arange(2, 40)
    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i < 4:
            poly_order = i + 1
            symmetric = False
        else:
            poly_order = 1
            symmetric = True
        if i in (0, 4):
            min_length = 3
        else:
            min_length = 20
        baseline, params = baseline_fitter.cwt_br(
            y, poly_order=poly_order, scales=scales, min_length=min_length,
            symmetric=symmetric, num_std=0.5
        )
        ax.plot(baseline, 'g--')


fabc (Fully Automatic Baseline Correction)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.fabc` identifies baseline segments by thresholding the squared first derivative
of the data, similar to :meth:`~.Baseline.dietrich`. However, fabc approximates the first derivative
using a continous wavelet transform with the Haar wavelet, which is more robust to noise
than the numerical derivative in Dietrich's method. The baseline is then fit using
Whittaker smoothing with all baseline points having a weight of 1 and all other points
a weight of 0.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            lam = 1e4
        elif i == 3:
            lam = 5e2
        elif i in (0, 4):
            lam = 1e6
        else:
            lam = 1e3
        if i == 1:
            num_std = 2.5
        else:
            num_std = 3
        baseline, params = baseline_fitter.fabc(y, lam=lam, scale=16, num_std=num_std, min_length=3)
        ax.plot(baseline, 'g--')


rubberband (Rubberband Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.rubberband` uses a convex hull to find local minima
of the data, which are then used to construct the baseline using either
linear interpolation or Whittaker smoothing. The rubberband method is simple and
easy to use for convex shaped data, but performs poorly for concave data. To get
around this, some commercial spectroscopy software use a `patented method
<https://patents.google.com/patent/US20060212275A1/en>`_ to coerce
the data into a convex shape so that the rubberband method still works. pybaselines
uses an alternate approach of allowing splitting the data in segments, in order
to reduce the concavity of each individual section; it is less user-friendly
than Bruker's method but works well enough for data with similar baselines and
peak positions.

.. note::
   For noisy data, rubberband performs significantly better when smoothing
   the data beforehand.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            segments = [100, 250, 300]
        elif i == 3:
            segments = [250, 380]
        else:
            segments = 1
        if i < 4:
            smooth_half_window = 5
        else:
            smooth_half_window = 0
        baseline, params = baseline_fitter.rubberband(
            y, segments=segments, lam=0, smooth_half_window=smooth_half_window
        )
        ax.plot(baseline, 'g--')


By using Whittaker smoothing (or other smoothing interpolation methods) rather than
linear interpolation to construct the baseline from the convex hull points, the
negative effects of applying the rubberband method to concave data can be
slightly reduced, as seen below.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            segments = [100, 250, 300]
        elif i == 3:
            segments = [380]
        else:
            segments = 1
        if i < 4:
            smooth_half_window = 5
        else:
            smooth_half_window = 0
        baseline, params = baseline_fitter.rubberband(
            y, segments=segments, lam=100, smooth_half_window=smooth_half_window
        )
        ax.plot(baseline, 'g--')
