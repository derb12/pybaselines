================
Spline Baselines
================

The contents of :mod:`pybaselines.spline` contain algorithms for fitting
splines to the baseline.

Introduction
------------

A spline is a piecewise joining of individual curves. There are different types of
splines, but only basis splines (B-splines) will be discussed since they are
predominantly used in pybaselines. B-splines can be expressed as:

.. math::

    z(x) = \sum\limits_{i}^N \sum\limits_{j}^M {B_j(x_i) c_j}

where :math:`N` is the number of points in :math:`x`, :math:`M` is the number of spline
basis functions, :math:`B_j(x_i)` is the j-th basis function evaluated at :math:`x_i`,
and :math:`c_j` is the coefficient for the j-th basis (which is analogous to
the height of the j-th basis). In pybaselines, the number of spline basis functions,
:math:`M`, is calculated as the number of knots, `num_knots`, plus the spline degree
minus 1.

For regular B-spline fitting, the spline coefficients that best fit the data
are gotten from minimizing the least-squares:

.. math:: \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2

where :math:`y_i` and :math:`x_i` are the measured data, and :math:`w_i` is
the weighting. In order to control the smoothness of the fitting spline, a penalty
on the finite-difference between spline coefficients is added, resulting in penalized
B-splines called P-splines (several `good <https://doi.org/10.1214/ss/1038425655>`_
`papers <https://doi.org/10.1002/wics.125>`_ exist for an introduction to P-splines).
The minimized function for P-splines is thus:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

where :math:`\lambda` is the penalty scale factor, and
:math:`\Delta^d` is the finite-difference operator of order d. Note that P-splines
use uniformly spaced knots so that the finite-difference is easy to calculate.

The resulting linear equation for solving the above minimization is:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

where :math:`W` is the diagaonal matrix of the weights, :math:`B` is the matrix
containing all of the spline basis functions, and :math:`D_d` is the matrix
version of :math:`\Delta^d` (same as :ref:`explained <difference-matrix-explanation>`
for Whittaker-smoothing-based algorithms). P-splines have similarities with Whittaker
smoothing; in fact, if the number of basis functions, :math:`M`, is set up to be equal
to the number of data points, :math:`N`, and the spline degree is set to 0, then
:math:`B` becomes the identity matrix and the above equation becomes identical
to the equation used for Whittaker smoothing.


Algorithms
----------

mixture_model (Mixture Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.mixture_model` considers the data as a mixture model composed of
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
        baseline, params = baseline_fitter.mixture_model(y, lam=lam, p=p, symmetric=symmetric)
        ax.plot(baseline, 'g--')


irsqr (Iterative Reweighted Spline Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.irsqr` uses penalized splines and iterative reweighted least squares
to perform quantile regression on the data.

.. plot::
   :align: center
   :context: close-figs

    quantiles = {0: 0.3, 1: 0.1, 2: 0.2, 3: 0.25, 4: 0.5}
    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 0:
            lam = 1e7
        elif i == 1:
            lam = 1e6
        else:
            lam = 1e5
        baseline, params = baseline_fitter.irsqr(y, lam=lam, quantile=quantiles[i])
        ax.plot(baseline, 'g--')


corner_cutting (Corner-Cutting Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.corner_cutting` iteratively removes corner points and then creates
a quadratic Bezier spline from the remaining points. Continuity between
the individual Bezier curves is maintained by adding control points halfway
between all but the first and last non-corner points.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            max_iter = 12
        elif i == 3:
            max_iter = 11
        else:
            max_iter = 100

        baseline, params = baseline_fitter.corner_cutting(y, max_iter=max_iter)
        ax.plot(baseline, 'g--')


pspline_asls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_asls` is a penalized spline version of :meth:`~.Baseline.asls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            lam = 1e4
            p = 0.01
        elif i == 4:
            lam = 1e6
            p = 0.5
        else:
            lam = 1e3
            p = 0.01
        baseline, params = baseline_fitter.pspline_asls(y, lam=lam, p=p)
        ax.plot(baseline, 'g--')



pspline_iasls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_iasls` is a penalized spline version of :meth:`~.Baseline.iasls`.

Minimized function:

.. math::

    \sum\limits_{i}^N (w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j}))^2
    + \lambda \sum\limits_{i}^{M - 2} (\Delta^2 c_i)^2
    + \lambda_1 \sum\limits_{i}^{N - 1} (\Delta^1 (y_i - \sum\limits_{j}^M {B_j(x_i) c_j}))^2

Linear system:

.. math::

    (B^{\top} W^{\top} W B + \lambda_1 B^{\top} D_1^{\top} D_1 B + \lambda D_2^{\top} D_2) c
    = (B^{\top} W^{\top} W B + \lambda_1 B^{\top} D_1^{\top} D_1) y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 0:
            lam = 1e3
            p = 0.1
        elif i == 1:
            lam = 1e2
            p = 0.01
        elif i == 4:
            lam = 1e5
            p = 0.5
        else:
            lam = 1e1
            p = 0.01
        baseline, params = baseline_fitter.pspline_iasls(y, lam=lam, p=p)
        ax.plot(baseline, 'g--')


pspline_airpls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_airpls` is a penalized spline version of :meth:`~.Baseline.airpls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            lam = 1e4
        elif i == 4:
            lam = 1e6
        else:
            lam = 1e3
        baseline, params = baseline_fitter.pspline_airpls(y, lam=lam)
        ax.plot(baseline, 'g--')


pspline_arpls (Penalized Spline Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_arpls` is a penalized spline version of :meth:`~.Baseline.arpls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

Weighting:

.. math::

    w_i = \frac
        {1}
        {1 + exp{\left(\frac
            {2(r_i - (-\mu^- + 2 \sigma^-))}
            {\sigma^-}
        \right)}}

where :math:`r_i = y_i - z_i` and :math:`\mu^-` and
:math:`\sigma^-` are the mean and standard deviation, respectively, of the negative
values in the residual vector :math:`\mathbf r`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        baseline, params = baseline_fitter.pspline_arpls(y)
        ax.plot(baseline, 'g--')


pspline_drpls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_drpls` is a penalized spline version of :meth:`~.Baseline.drpls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - 2}(1 - \eta w_{i,intp}) (\Delta^2 c_i)^2
    + \sum\limits_{i}^{M - 1} (\Delta^1 (c_i))^2

where :math:`\eta` is a value between 0 and 1 that controls the
effective value of :math:`\lambda`. :math:`w_{intp}` are the weights, :math:`w`,
after interpolating using :math:`x` and the basis midpoints in order to map the
weights from length :math:`N` to length :math:`M`.

Linear system:

.. math::

    (B^{\top}W B + D_1^{\top} D_1 + \lambda (I - \eta W_{intp}) D_2^{\top} D_2) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 3:
            lam = 1e2
        else:
            lam = 1e3
        baseline, params = baseline_fitter.pspline_drpls(y, lam=lam)
        ax.plot(baseline, 'g--')


pspline_iarpls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_iarpls` is a penalized spline version of :meth:`~.Baseline.iarpls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 3:
            lam = 1e2
        else:
            lam = 1e3
        baseline, params = baseline_fitter.pspline_iarpls(y, lam=lam)
        ax.plot(baseline, 'g--')


pspline_aspls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_aspls` is a penalized spline version of :meth:`~.Baseline.aspls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} \alpha_{i,intp} (\Delta^d c_i)^2

where

.. math::

    \alpha_i = \frac
        {abs(r_i)}
        {max(abs(\mathbf r))}

and :math:`\alpha_{intp}` is the :math:`\alpha` array after interpolating using
:math:`x` and the basis midpoints in order to map :math:`\alpha` from length
:math:`N` to length :math:`M`.

Linear system:

.. math::

    (B^{\top} W B + \lambda \alpha_{intp} D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 1:
            lam = 1e4
        elif i == 3:
            lam = 1e2
        else:
            lam = 1e3
        baseline, params = baseline_fitter.pspline_aspls(y, lam=lam)
        ax.plot(baseline, 'g--')


pspline_psalsa (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_psalsa` is a penalized spline version of :meth:`~.Baseline.psalsa`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 0:
            k = 2
        else:
            k = 0.5
        baseline, params = baseline_fitter.pspline_psalsa(y, lam=1e3, k=k)
        ax.plot(baseline, 'g--')



pspline_derpsalsa (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_derpsalsa` is a penalized spline version of :meth:`~.Baseline.derpsalsa`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 0:
            k = 2
        else:
            k = 0.5
        baseline, params = baseline_fitter.pspline_derpsalsa(y, lam=1e2, k=k)
        ax.plot(baseline, 'g--')


pspline_mpls (Penalized Spline Morphological Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline.pspline_mpls` is a penalized spline version of :meth:`~.Baseline.mpls`.

Minimized function:

.. math::

    \sum\limits_{i}^N w_i (y_i - \sum\limits_{j}^M {B_j(x_i) c_j})^2
    + \lambda \sum\limits_{i}^{M - d} (\Delta^d c_i)^2

Linear system:

.. math::

    (B^{\top} W B + \lambda D_d^{\top} D_d) c = B^{\top} W y

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
    figure, axes, handles = create_plots(data, baselines)
    for i, (ax, y) in enumerate(zip(axes, data)):
        if i == 4:
            # few baseline points are identified, so use a higher p value so
            # that other points contribute to fitting; mpls isn't good for
            # signals with positive and negative peaks
            p = 0.1
        else:
            p = 0.001
        baseline, params = baseline_fitter.pspline_mpls(y, lam=lam, p=p)
        ax.plot(baseline, 'g--')
