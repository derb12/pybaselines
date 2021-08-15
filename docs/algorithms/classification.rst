========================
Classification Baselines
========================

The contents of :mod:`pybaselines.classification` contain algorithms that rely on
classifying peak and/or baseline segments.

Introduction
------------

Classification methods are similar to
:ref:`selective masking <selective-masking-explanation>` as explained in the polynomial
section, but use sophisticated techniques to determine the baseline points rather than
relying on manual selection.

All classification functions allow inputting weights to override the baseline classification,
which can be helpful, for example, to ensure a small peak is not classified as baseline without
having to alter any parameters which could otherwise reduce the effectiveness of the classification
method. The plot below shows such an example.

.. plot::
   :align: center

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines.classification import std_distribution

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 1, 180, 25)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 800, 10)
    )
    real_baseline = 5 + 5 * np.exp(-x / 400)
    np.random.seed(1)  # set random seed
    noise = np.random.normal(0, 0.2, x.size)
    y = signal + real_baseline + noise

    weights = np.ones(y.shape[0])
    # a weight of 0 designates that it is a peak; ensures the small peak at ~180 is
    # not classified as part of the baseline
    weights[(x > 100) & (x < 250)] = 0

    unweighted_baseline = std_distribution(y, x, half_window=10, num_std=1.3)[0]
    weighted_baseline = std_distribution(y, x, half_window=10, num_std=1.3, weights=weights)[0]

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

:func:`.dietrich` calculates the power spectrum of the data as the squared derivative
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
    from pybaselines import classification

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
        if i < 4:
            poly_order = i + 1
        else:
            poly_order = 1
        if i == 2:
            num_std = 1.95
        elif i == 3:
            num_std = 2.6
        else:
            num_std = 1.85
        baseline = classification.dietrich(y, None, 8, num_std=num_std, poly_order=poly_order)
        ax.plot(baseline[0], 'g--')


golotvin (Golotvin's Classification Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.golotvin` divides the data into sections and takes the minimum standard
deviation of all the sections as the noise's standard deviation for the entire data.
Then classifies any point where the rolling max minus min is less than a multiple of
the noise's standard deviation as belonging to the baseline.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            half_window = 25
        else:
            half_window = 10
        if i in (1, 3):
            num_std = 40
        else:
            num_std = 10
        baseline = classification.golotvin(y, None, half_window=half_window, num_std=num_std)
        ax.plot(baseline[0], 'g--')


std_distribution (Standard Deviation Distribution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.std_distribution` identifies baseline segments by analyzing the rolling
standard deviation distribution. The rolling standard deviations are split into two
distributions, with the smaller distribution assigned to noise. Baseline points are
then identified as any point where the rolled standard deviation is less than a multiple
of the median of the noise's standard deviation distribution.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 1:
            half_window = 30
            num_std = 0.9
        elif i in (2, 3):
            half_window = 8
            num_std = 3.5
        else:
            half_window = 12
            num_std = 1.1
        baseline = classification.std_distribution(y, None, half_window=half_window, num_std=num_std)
        ax.plot(baseline[0], 'g--')


std_threshold (Standard Deviation Threshold)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.std_threshold` (sometimes referred to as "FastChrom" in literature) identifies
baseline segments by analyzing the rolling standard deviation distribution, similar
to :func:`std_distribution`. Baseline points are identified as any point where the
rolling standard deviation is less than the specified threshold, and peak regions are
iteratively interpolated until the baseline is below the data.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 4:
            min_fwhm = y.shape[0]  # ensure it doesn't try to fill in negative peaks
        else:
            min_fwhm = None
        baseline = classification.std_threshold(
            y, None, half_window=12, threshold=1, min_fwhm=min_fwhm
        )
        ax.plot(baseline[0], 'g--')
