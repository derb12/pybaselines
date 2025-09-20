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
