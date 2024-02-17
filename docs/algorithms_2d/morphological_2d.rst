=======================
Morphological Baselines
=======================

.. note::
   All morphological algorithms use a ``half_window`` parameter to define the size
   of the window used for the morphological operators. ``half_window`` is index-based,
   rather than based on the units of the data, so proper conversions must be done
   by the user to get the desired window size.


Algorithms
----------

mor (Morphological)
~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.mor`:
:ref:`explanation for the algorithm <algorithms/morphological:mor (Morphological)>`.

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

    baseline, params = baseline_fitter.mor(y, half_window=(6, 4))
    create_plots(y, baseline)


imor (Improved Morphological)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.imor`:
:ref:`explanation for the algorithm <algorithms/morphological:imor (Improved Morphological)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.imor(y, half_window=(4, 2), tol=5e-3)
    create_plots(y, baseline)


rolling_ball (Rolling Ball)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.rolling_ball`:
:ref:`explanation for the algorithm <algorithms/morphological:rolling_ball (Rolling Ball)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.rolling_ball(y, half_window=(8, 5), smooth_half_window=3)
    create_plots(y, baseline)


tophat (Top-hat Transformation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.tophat`:
:ref:`explanation for the algorithm <algorithms/morphological:tophat (Top-hat Transformation)>`.


.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.tophat(y, half_window=(8, 5))
    create_plots(y, baseline)
