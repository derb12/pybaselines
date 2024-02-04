================
Spline Baselines
================

Introduction
------------

The two dimensional extension of penalized splines (P-splines) for baseline correction
within pybaselines follows the framework of Eilers, Currie, and Durbán
from `[1] <https://doi.org/10.1016/j.csda.2004.07.008>`_. The exact equations will be
omitted here (those interested should read the paper, it is very good), but the end result
is that the normal equation for solving the penalized system can be expressed as a
`generalized linear array model <https://en.wikipedia.org/wiki/Generalized_linear_array_model>`_
which allows directly using the matrices of the measured data, :math:`Y`, and the weights,
:math:`W`, rather than flattening them, which significantly reduces the required
memory and computation time.


Algorithms
----------

mixture_model (Mixture Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.mixture_model`:
:ref:`explanation for the algorithm <algorithms/spline:mixture_model (Mixture Model)>`.

.. plot::
   :align: center
   :context: reset

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

        # 4 total plots: 2 countours and 2 projections
        row_names = ('Raw Data', 'Baseline Corrected')
        for i, dataset in enumerate((y, y - fit_baseline)):
            fig = plt.figure(layout='constrained', figsize=plt.figaspect(0.5))
            fig.suptitle(row_names[i])
            ax = fig.add_subplot(1 ,2, 2)
            ax.contourf(X, Z, dataset, cmap='coolwarm')
            ax.set_xticks([])
            ax.set_yticks([])
            ax_2 = fig.add_subplot(1, 2, 1, projection='3d')
            ax_2.plot_surface(X, Z, dataset, cmap='coolwarm')
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            ax_2.set_zticks([])
            if i == 0:
                pass#ax.set_title('Contours')
                #ax_2.set_title('3D Projections')


    x, z, y, real_baseline = create_data()
    baseline_fitter = Baseline2D(x, z, check_finite=False)

    baseline, params = baseline_fitter.mixture_model(y, lam=(1e3, 1e2))
    create_plots(y, baseline)


irsqr (Iterative Reweighted Spline Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.irsqr`:
:ref:`explanation for the algorithm <algorithms/spline:irsqr (Iterative Reweighted Spline Quantile Regression)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.irsqr(y, lam=(1e3, 1e2), quantile=0.3)
    create_plots(y, baseline)


pspline_asls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_asls`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_asls (Penalized Spline Asymmetric Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_asls(y, lam=(1e3, 1e0), p=0.005)
    create_plots(y, baseline)


pspline_iasls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_iasls`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_iasls (Penalized Spline Asymmetric Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_iasls(y, lam=(1e2, 1e-2))
    create_plots(y, baseline)


pspline_airpls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_airpls`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_airpls (Penalized Spline Asymmetric Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_airpls(y, lam=(1e3, 1e-1))
    create_plots(y, baseline)


pspline_arpls (Penalized Spline Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_arpls`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_arpls (Penalized Spline Asymmetrically Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_arpls(y, lam=(1e3, 5e0))
    create_plots(y, baseline)


pspline_iarpls (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_iarpls`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_iarpls (Penalized Spline Asymmetric Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_iarpls(y, lam=(1e2, 1e0))
    create_plots(y, baseline)


pspline_psalsa (Penalized Spline Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_psalsa`:
:ref:`explanation for the algorithm <algorithms/spline:pspline_psalsa (Penalized Spline Asymmetric Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    baseline, params = baseline_fitter.pspline_psalsa(y, lam=(1e3, 5e0), k=0.5)
    create_plots(y, baseline)
