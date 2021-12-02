===================
Whittaker Baselines
===================

The contents of :mod:`pybaselines.whittaker` contain Whittaker-smoothing-based
algorithms for fitting the baseline.

Introduction
------------

Whittaker-smoothing-based (WSB) algorithms are usually referred to in literature
as weighted least squares, penalized least squares, or asymmetric least squares,
but are referred to as WSB in pybaselines to distinguish them from polynomial
techniques that also take advantage of weighted least squares (like :func:`.loess`)
and penalized least squares (like :func:`.penalized_poly`).

The general idea behind WSB algorithms is to make the baseline match the measured
data as well as it can while also penalizing the roughness of the baseline. The
resulting general function that is minimized to determine the baseline is then

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

where :math:`y_i` is the measured data, :math:`z_i` is the estimated baseline,
:math:`\lambda` is the penalty scale factor, :math:`w_i` is the weighting, and
:math:`\Delta^d` is the finite-difference operator of order d.

The resulting linear equation for solving the above minimization is:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

.. _difference-matrix-explanation:

where :math:`W` is the diagaonal matrix of the weights, and :math:`D_d` is the matrix
version of :math:`\Delta^d`, which is also the d-th derivative of the identity matrix.
For example, for an array of length 5, :math:`D_1` (first order difference matrix) is:

.. math::

    \begin{bmatrix}
    -1 & 1 & 0 & 0 & 0 \\
    0 & -1 & 1 & 0 & 0 \\
    0 & 0 & -1 & 1 & 0 \\
    0 & 0 & 0 & -1 & 1 \\
    \end{bmatrix}

and :math:`D_2` (second order difference matrix) is:

.. math::

    \begin{bmatrix}
    1 & -2 & 1 & 0 & 0 \\
    0 & 1 & -2 & 1 & 0 \\
    0 & 0 & 1 & -2 & 1 \\
    \end{bmatrix}

Most WSB techniques recommend using the second order difference matrix, although
some techniques use both the first and second order difference matrices.

The baseline is iteratively calculated using the linear system above by solving for
the baseline, :math:`z`, updating the weights, solving for the baseline using the new
weights, and repeating until some exit criteria.
The difference between WSB algorithms is the selection of weights and/or the
function that is minimized.

.. note::
   The :math:`\lambda` (``lam``) value required to fit a particular baseline for all WSB
   methods will increase as the number of data points increases, with the relationship
   being roughly :math:`\log(\lambda) \propto \log(\text{number of data points})`. For example,
   a ``lam`` value of :math:`10^3` that fits a dataset with 100 points may have to be :math:`10^7`
   to fit the same data with 1000 points, and :math:`10^{11}` for 10000 points.


Algorithms
----------

asls (Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`.asls` (sometimes called "ALS" in literature) function is the
original implementation of Whittaker smoothing for baseline fitting.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.


.. plot::
   :align: center
   :context: reset

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines import whittaker

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
            lam = 1e6
            p = 0.01
        elif i == 4:
            lam = 1e8
            p = 0.5
        else:
            lam = 1e5
            p = 0.01
        baseline = whittaker.asls(y, lam=lam, p=p)
        ax.plot(baseline[0], 'g--')


iasls (Improved Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.iasls` is an attempt to improve the asls algorithm by considering
both the roughness of the baseline and the first derivative of the residual
(data - baseline).

Minimized function:

.. math::

    \sum\limits_{i}^N (w_i (y_i - z_i))^2
    + \lambda \sum\limits_{i}^{N - 2} (\Delta^2 z_i)^2
    + \lambda_1 \sum\limits_{i}^{N - 1} (\Delta^1 (y_i - z_i))^2

Linear system:

.. math::

    (W^{\top} W + \lambda_1 D_1^{\top} D_1 + \lambda D_2^{\top} D_2) z
    = (W^{\top} W + \lambda_1 D_1^{\top} D_1) y

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 0:
            lam = 1e7
            p = 0.1
        elif i == 1:
            lam = 1e4
            p = 0.01
        elif i == 4:
            lam = 1e7
            p = 0.5
        else:
            lam = 1e3
            p = 0.01
        baseline = whittaker.iasls(y, lam=lam, lam_1=1e-4, p=p)
        ax.plot(baseline[0], 'g--')


airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.airpls` uses an exponential weighting of the negative residuals to
attempt to provide a better fit than the asls method.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        0 & y_i \ge z_i \\
        exp{\left(\frac{t (y_i - z_i)}{|\mathbf{r}^-|}\right)} & y_i < z_i
    \end{array}\right.

where :math:`t` is the iteration number and :math:`|\mathbf{r}^-|` is the l1-norm of the negative
values in the residual vector :math:`\mathbf r`, ie. :math:`\sum\limits_{y_i - z_i < 0} |y_i - z_i|`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for ax, y in zip(*create_data()):
        baseline = whittaker.airpls(y, 1e5)
        ax.plot(baseline[0], 'g--')


arpls (Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.arpls` uses a single weighting function that is designed to account
for noisy data.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \frac
        {1}
        {1 + exp{\left(\frac
            {2(r_i - (-\mu^- + 2 \sigma^-))}
            {\sigma^-}
        \right)}}

where :math:`r_i = y_i - z_i` and :math:`\mu^-` and :math:`\sigma^-` are the mean and standard
deviation, respectively, of the negative values in the residual vector :math:`\mathbf r`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for ax, y in zip(*create_data()):
        baseline = whittaker.arpls(y, 1e5)
        ax.plot(baseline[0], 'g--')


drpls (Doubly Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.drpls` uses a single weighting function that is designed to account
for noisy data, similar to arpls. Further, it takes into account both the
first and second derivatives of the baseline and uses a parameter :math:`\eta`
to adjust the fit in peak versus non-peak regions.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2
    + \lambda \sum\limits_{i}^{N - 2}(1 - \eta w_i) (\Delta^2 z_i)^2
    + \sum\limits_{i}^{N - 1} (\Delta^1 (z_i))^2

where :math:`\eta` is a value between 0 and 1 that controls the
effective value of :math:`\lambda`.

Linear system:

.. math::

    (W + D_1^{\top} D_1 + \lambda (I - \eta W) D_2^{\top} D_2) z = W y

where :math:`I` is the identity matrix.

Weighting:

.. math::

    w_i = \frac{1}{2}\left(
        1 -
        \frac
            {exp(t)(r_i - (-\mu^- + 2 \sigma^-))/\sigma^-}
            {1 + abs[exp(t)(r_i - (-\mu^- + 2 \sigma^-))/\sigma^-]}
    \right)

where :math:`r_i = y_i - z_i`, :math:`t` is the iteration number, and
:math:`\mu^-` and :math:`\sigma^-` are the mean and standard deviation,
respectively, of the negative values in the residual vector :math:`\mathbf r`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 3:
            lam = 1e5
        else:
            lam = 1e6
        baseline = whittaker.drpls(y, lam=lam)
        ax.plot(baseline[0], 'g--')


iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.iarpls` is an attempt to improve the arpls method, which has a tendency
to overestimate the baseline when fitting small peaks in noisy data, by using an
adjusted weighting formula.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \frac{1}{2}\left(
        1 -
        \frac
            {exp(t)(r_i - 2 \sigma^-)/\sigma^-}
            {\sqrt{1 + [exp(t)(r_i - 2 \sigma^-)/\sigma^-]^2}}
    \right)

where :math:`r_i = y_i - z_i`, :math:`t` is the iteration number, and
:math:`\sigma^-` is the standard deviation of the negative values in
the residual vector :math:`\mathbf r`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for ax, y in zip(*create_data()):
        baseline = whittaker.iarpls(y, 1e4)
        ax.plot(baseline[0], 'g--')


aspls (Adaptive Smoothness Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.aspls`, similar to the iarpls method, is an attempt to improve the arpls method,
which it does by using an adjusted weighting function and an additional parameter :math:`\alpha`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2
    + \lambda \sum\limits_{i}^{N - d} \alpha_i (\Delta^d z_i)^2

where

.. math::

    \alpha_i = \frac
        {abs(r_i)}
        {max(abs(\mathbf r))}

Linear system:

.. math::

    (W + \lambda \alpha D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \frac
        {1}
        {1 + exp{\left(\frac
            {0.5 (r_i - \sigma^-)}
            {\sigma^-}
        \right)}}

where :math:`r_i = y_i - z_i`  and :math:`\sigma^-` is the standard deviation
of the negative values in the residual vector :math:`\mathbf r`. (Note that the
:math:`0.5 (r_i - \sigma^-) / \sigma^-` term is different than the published
version of the asPLS, which used :math:`2 (r_i - \sigma^-) / \sigma^-`. pybaselines
uses the factor of 0.5 since it matches the results in Table 2 and Figure 5
of the asPLS paper closer than the factor of 2 and fits noisy data much better).

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for ax, y in zip(*create_data()):
        baseline = whittaker.aspls(y, 1e6)
        ax.plot(baseline[0], 'g--')


psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.psalsa` is an attempt at improving the asls method to better fit noisy data
by using an exponential decaying weighting for positive residuals.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p \cdot exp{\left(\frac{-(y_i - z_i)}{k}\right)} & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.

where :math:`k` is a factor that controls the exponential decay of the weights for baseline
values greater than the data and should be approximately the height at which a value could
be considered a peak.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 0:
            k = 2
        else:
            k = 0.5
        baseline = whittaker.psalsa(y, 1e5, k=k)
        ax.plot(baseline[0], 'g--')


derpsalsa (Derivative Peak-Screening Asymmetric Least Squares Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.derpsalsa` is an attempt at improving the asls method to better fit noisy data
by using an exponential decaying weighting for positive residuals. Further, it calculates
additional weights based on the first and second derivatives of the data.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i}^{N - d} (\Delta^d z_i)^2

Linear system:

.. math::

    (W + \lambda D_d^{\top} D_d) z = W y

Weighting:

.. math::

    w_i = w_{0i} * w_{1i} * w_{2i}

where:

.. math::

    w_{0i} = \left\{\begin{array}{cr}
        p \cdot exp{\left(\frac{-[(y_i - z_i)/k]^2}{2}\right)} & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.

.. math::

    w_{1i} = exp{\left(\frac{-[y_{sm_i}' / rms(y_{sm}')]^2}{2}\right)}

.. math::

    w_{2i} = exp{\left(\frac{-[y_{sm_i}'' / rms(y_{sm}'')]^2}{2}\right)}

:math:`k` is a factor that controls the exponential decay of the weights for baseline
values greater than the data and should be approximately the height at which a value could
be considered a peak, :math:`y_{sm}'` and :math:`y_{sm}''` are the first and second derivatives,
respectively, of the smoothed data, :math:`y_{sm}`, and :math:`rms()` is the root-mean-square operator.
:math:`w_1` and :math:`w_2` are precomputed, while :math:`w_0` is updated each iteration.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    for i, (ax, y) in enumerate(zip(*create_data())):
        if i == 0:
            k = 2
        else:
            k = 0.5
        baseline = whittaker.psalsa(y, 1e5, k=k)
        ax.plot(baseline[0], 'g--')
