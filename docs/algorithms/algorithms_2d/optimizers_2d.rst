===================
Optimizer Baselines
===================

Algorithms
----------

collab_pls (Collaborative Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.collab_pls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/optimizers:collab_pls (Collaborative Penalized Least Squares)>`.
There is no figure showing a fit for for this method since it requires multiple sets of data.

adaptive_minmax (Adaptive MinMax)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.adaptive_minmax`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/optimizers:adaptive_minmax (Adaptive MinMax)>`.

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: True

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian2d
    from pybaselines import Baseline2D


    def create_data():
        x = np.linspace(-20, 20, 80)
        z = np.linspace(-20, 20, 80)
        X, Z = np.meshgrid(x, z, indexing='ij')
        signal = (
            gaussian2d(X, Z, 12, -9, -9)
            + gaussian2d(X, Z, 11, 3, 3)
            + gaussian2d(X, Z, 13, 11, 11)
            + gaussian2d(X, Z, 8, 5, -11, 1.5, 1)
            + gaussian2d(X, Z, 16, -8, 8)
        )
        baseline = 0.1 + 0.08 * X - 0.05 * Z + 0.005 * (Z + 20)**2
        noise = np.random.default_rng(0).normal(scale=0.1, size=signal.shape)
        y = signal + baseline + noise

        return x, z, y, baseline


    def create_plots(y, fit_baseline):
        X, Z = np.meshgrid(
            np.arange(y.shape[0]), np.arange(y.shape[1]), indexing='ij'
        )

        # 4 total plots: 2 contours and 2 projections
        row_names = ('Raw Data', 'Baseline Corrected')
        for i, dataset in enumerate((y, y - fit_baseline)):
            fig = plt.figure(layout='constrained', figsize=plt.figaspect(0.5))
            fig.suptitle(row_names[i])
            ax = fig.add_subplot(1, 2, 2)
            ax.contourf(X, Z, dataset, cmap='coolwarm')
            ax.set_xticks([])
            ax.set_yticks([])
            ax_2 = fig.add_subplot(1, 2, 1, projection='3d')
            ax_2.plot_surface(X, Z, dataset, cmap='coolwarm')
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            ax_2.set_zticks([])

    x, z, y, real_baseline = create_data()
    baseline_fitter = Baseline2D(x, z, check_finite=False)

    baseline, params = baseline_fitter.adaptive_minmax(y, poly_order=(2, 3))
    create_plots(y, baseline)

individual_axes (1D Baseline Correction Along Individual Axes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.individual_axes` is the single unique 2D baseline correction
algorithm that is not available as a 1D algorithm, and it applies the specified 1D
baseline algorithm along each row and/or column of the measured data. This is useful
if the axes of the data are not correlated such that no information is lost by
fitting each axis separately, or when baselines only exist along one axis.

Note that one limitation of :meth:`~.Baseline2D.individual_axes` is that it does not
handle array-like `method_kwargs`, such as when different input weights are desired
for each dataset along the rows and/or columns. However, this is an extremely niche
situation, and could be handled by simply using a for-loop to do one dimensional
baseline correction instead.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.individual_axes(
        y, method='arpls', axes=0, method_kwargs=({'lam': 1e4})
    )
    create_plots(y, baseline)


optimize_pls (Optimize Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.optimize_pls`, much like its
:ref:`1D counterpart  <algorithms/algorithms_1d/optimizers:optimize_pls (Optimize Penalized Least Squares)>`,
optimizes the regularization parameters for Whittaker smoothing and penalized spline algorithms.
For 2D, the general equation of penalized least squares given by

.. math::

    F + \lambda_r P_r + \lambda_c P_c

where :math:`F` is the fidelity of the fit, :math:`P_r` is the penalty term along the rows
whose contribution is controlled by the regularization parameter :math:`\lambda_r`, and
:math:`P_c` is the penalty term along the columns whose contribution is controlled by the
regularization parameter :math:`\lambda_c`. In general, both Whittaker
smoothing and penalized splines have a fidelity given by:

.. math::

    F = \sum\limits_{i}^M \sum\limits_{j}^N W_{ij} (Y_{ij} - V_{ij})^2

where :math:`Y` is the measured data, :math:`V` is the calculated baseline,
and :math:`W` is the weight. The penalties for Whittaker smoothing are generally:

.. math::

    P_r = \sum\limits_{i}^{M - d_r} (V_{i\bullet} \Delta^{d_r})^2

.. math::

    P_c = \sum\limits_{j}^{N - d_c} (\Delta^{d_c} V_{j\bullet})^2

:math:`\Delta^{d_r}` is the finite-difference operator of order
:math:`d_r` along each row of :math:`V`, :math:`V_{i\bullet}`, and :math:`\Delta^{d_c}` is the
finite-difference operator of order :math:`d_c` along each column of :math:`V`, :math:`V_{j\bullet}`.

Likewise, for penalized splines, the penalties are generally:

.. math::

    P_r = \sum\limits_{i}^{g - d_r} (\alpha_{i\bullet} \Delta^{d_r})^2

.. math::

    P_c = \sum\limits_{j}^{h - d_c} (\Delta^{d_c} \alpha_{j\bullet})^2

where :math:`a` are the calculated spline coefficients.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.optimize_pls(
        y, min_value=(6, 1), max_value=(9, 5), method='arpls', euclidean=True
    )
    create_plots(y, baseline)
