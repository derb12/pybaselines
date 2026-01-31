=======================
Morphological Baselines
=======================

Introduction
------------

`Morphological operations <https://wikipedia.org/wiki/Mathematical_morphology>`_
include dilation, erosion, opening, and closing, and they use moving windows to compute the
maximum, minimum, or a combination of the two within each window, as shown in the figure
below. The algorithms in this section use these operations to estimate the baseline.

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: True

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from scipy.ndimage import grey_closing, grey_dilation, grey_erosion, grey_opening

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 600, 12)
        + gaussian(x, 9, 800, 10)
    )
    real_baseline = 5 + gaussian(x, 5, 500, 300)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)
    y = signal + real_baseline + noise
    window = 25

    operators = (
        (grey_dilation, 'dilation'),
        (grey_erosion, 'erosion'),
        (grey_closing, 'closing'),
        (grey_opening, 'opening'),
    )

    _, (ax, ax2) = plt.subplots(nrows=2, tight_layout={'pad': 0.1})
    ax.plot(y)
    ax.plot(grey_dilation(y, window), label='dilation: max(window)')
    ax.plot(grey_erosion(y, window), label='erosion: min(window)')
    ax2.plot(y)
    ax2.plot(grey_closing(y, window), label='closing: erosion(dilation(y))')
    ax2.plot(grey_opening(y, window), label='opening: dilation(erosion(y))')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend()
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.legend()

    plt.show()


.. note::
   All morphological algorithms use a ``half_window`` parameter to define the size
   of the window used for the morphological operators. ``half_window`` is index-based,
   rather than based on the units of the data, so proper conversions must be done
   by the user to get the desired window size.


Algorithms
----------

mpls (Morphological Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mpls` uses both morphological operations and Whittaker-smoothing
to create the baseline. First, a morphological opening is performed on the
data. Then, the index of the minimum data value between each flat region of the
opened data is selected as a baseline anchor point and given a weighting of
:math:`1 - p`, while all other points are given a weight of :math:`p`. The data
and weights are then used to calculate the baseline, similar to the :meth:`~.Baseline.asls`
method.

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

    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            # few baseline points are identified, so use a higher p value so
            # that other points contribute to fitting; mpls isn't good for
            # signals with positive and negative peaks
            p = 0.1
        else:
            p = 0.001
        baseline, params = baseline_fitter.mpls(y, lam=1e5, p=p)
        ax.plot(baseline, 'g--')


mor (Morphological)
~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mor` performs a morphological opening on the data and then selects
the element-wise minimum between the opening and the average of a morphological
erosion and dilation of the opening to create the baseline.

.. note::
   The baseline from the mor method is not smooth. Smoothing is left to the
   user to perform, if desired.


.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 50
        else:
            half_window = 20
        baseline, params = baseline_fitter.mor(y, half_window)
        ax.plot(baseline, 'g--')


imor (Improved Morphological)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.imor` is an attempt to improve the mor method, and iteratively selects the element-wise
minimum between the original data and the average of a morphological erosion and dilation
of the opening of either the data (first iteration) or previous iteration's baseline to
create the baseline.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        baseline, params = baseline_fitter.imor(y, 10)
        ax.plot(baseline, 'g--')


mormol (Morphological and Mollified Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mormol` iteratively convolves the erosion of the data with a mollifying (smoothing)
kernel, to produce a smooth baseline.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 60
        else:
            half_window = 30
        baseline, params = baseline_fitter.mormol(
            y, half_window, smooth_half_window=10, pad_kwargs={'extrapolate_window': 20}
        )
        ax.plot(baseline, 'g--')


amormol (Averaging Morphological and Mollified Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.amormol` iteratively convolves a mollifying (smoothing) kernel with the
element-wise minimum of the data and the average of the morphological closing
and opening of either the data (first iteration) or previous iteration's baseline.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        baseline, params = baseline_fitter.amormol(y, 20)
        ax.plot(baseline, 'g--')


rolling_ball (Rolling Ball)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.rolling_ball` performs a morphological opening on the data and
then smooths the result with a moving average, giving a baseline that
resembles rolling a ball across the data.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 60
        else:
            half_window = 30
        baseline, params = baseline_fitter.rolling_ball(y, half_window, smooth_half_window=20)
        ax.plot(baseline, 'g--')


mwmv (Moving Window Minimum Value)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mwmv` performs a morphological erosion on the data and
then smooths the result with a moving average.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 22
        else:
            half_window = 12
        baseline, params = baseline_fitter.mwmv(y, half_window, smooth_half_window=int(4 * half_window))
        ax.plot(baseline, 'g--')


tophat (Top-hat Transformation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.tophat` performs a morphological opening on the data.

.. note::
   The baseline from the tophat method is not smooth. Smoothing is left to the
   user to perform, if desired.


.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 50
        else:
            half_window = 20
        baseline, params = baseline_fitter.tophat(y, half_window)
        ax.plot(baseline, 'g--')


mpspline (Morphology-Based Penalized Spline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mpspline` uses both morphological operations and penalized splines
to create the baseline. First, the data is smoothed by fitting a penalized
spline to the closing of the data with a window of 3. Then baseline points are
identified where the smoothed data is equal to the element-wise minimum between the
opening of the smoothed data and the average of a morphological erosion and dilation
of the opening. The baseline points are given a weighting of :math:`1 - p`, while all
other points are given a weight of :math:`p`, similar to the :meth:`~.Baseline.mpls` method.
Finally, a penalized spline is fit to the smoothed data with the assigned weighting.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            lam = 1e4
        elif i == 3:
            lam = 5e2
        else:
            lam = 1e3
        if i == 4:
            # few baseline points are identified, so use a higher p value so
            # that other points contribute to fitting, same as mpls; done so
            # that no errors occur in case no baseline points are identified
            p = 0.1
        else:
            p = 0
        baseline, params = baseline_fitter.mpspline(
            y, lam=lam, p=p, pad_kwargs={'extrapolate_window': 30}
        )
        ax.plot(baseline, 'g--')


jbcd (Joint Baseline Correction and Denoising)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.jbcd` uses regularized least-squares fitting combined with morphological operations
to simultaneously obtain the baseline and denoised signal.

Minimized function:

.. math::

    \frac{1}{2} \sum\limits_{i = 1}^N (s_i + v_i - y_i)^2
    + \alpha \sum\limits_{i = 1}^N (v_i - Op_i)^2
    + \beta \sum\limits_{i = 1}^{N - d} (\Delta^d v_i)^2
    + \gamma \sum\limits_{i = 1}^{N - d} (\Delta^d s_i)^2

where :math:`y_i` is the measured data, :math:`v_i` is the estimated baseline,
:math:`s_i` is the estimated signal, :math:`\Delta^d` is the forward-difference
operator of order d, :math:`Op_i` is the morphological opening of the measured data,
and :math:`\alpha`, :math:`\beta`, and :math:`\gamma` are regularization parameters.

Linear systems:

The initial signal, :math:`s^0`, and baseline, :math:`v^0`, are set equal to :math:`y`,
and :math:`Op`, respectively. Then the signal and baseline at iteration :math:`t`, :math:`s^t`
and :math:`v^t`, are solved for sequentially using the following two
linear equations:

.. math::

    (I + 2 \gamma D_d^{\mathsf{T}} D_d) s^t = y - v^{t-1}

.. math::

    (I + 2 \alpha I + 2 \beta D_d^{\mathsf{T}} D_d) v^t = y - s^t + 2 \alpha Op

where :math:`I` is the identity matrix and :math:`D_d` is the matrix version
of :math:`\Delta^d`, which is also the d-th derivative of the identity matrix.
After each iteration, :math:`\beta`, and :math:`\gamma` are updated by user-specified
multipliers.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 50
        else:
            half_window = 20
        baseline, params = baseline_fitter.jbcd(
            y, half_window, gamma=1, beta_mult=1.05, gamma_mult=0.95
        )
        ax.plot(baseline, 'g--')

The signal with the baseline removed and noise decreased can also be obtained from the output
of the jbcd function.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the second-to-top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            half_window = 50
        else:
            half_window = 20
        baseline, params = baseline_fitter.jbcd(
            y, half_window, gamma=1, beta_mult=1.05, gamma_mult=0.95
        )

        ax.clear()  # remove the old plots in the axis
        data_handle = ax.plot(y)
        signal_handle = ax.plot(params['signal'])

    axes[-1].clear()  # remove the old legend
    axes[-1].legend(
        (data_handle[0], signal_handle[0]),
        ('data', 'signal from jcbd'), loc='center', frameon=False
    )
