====================
Polynomial Baselines
====================

Introduction
------------

In 2D, a polynomial can be expressed as

.. math::

    p(x, z) = \sum\limits_{i = 0}^{d_r} \sum\limits_{j = 0}^{d_c} {\beta_{i, j} x^i z^j}

where :math:`\beta` is the matrix of coefficients for the polynomial and :math:`d_r`
and :math:`d_c` are the polynomial degrees for the rows (:math:`x`) and
columns (:math:`z`), respectively.

For regular polynomial fitting, the polynomial coefficients that best fit data
are gotten from minimizing the least-squares:

.. math:: \sum\limits_{i}^M \sum\limits_{j}^N w_{ij} (y_{ij} - p(x_i, z_j))^2

where :math:`y_{ij}`, :math:`x_i`, and :math:`z_j` are the measured data, :math:`p(x_i, z_j)` is
the polynomial estimate at :math:`x_i`, and :math:`z_j` and :math:`w_{ij}` is the weighting.


However, since only the baseline of the data is desired, the least-squares
approach must be modified. For polynomial-based algorithms, this is done
by 1) only fitting the data in regions where there is only baseline, 2)
modifying the y-values being fit each iteration, or 3) penalizing outliers.

.. note::
   For two dimensional data, polynomial algorithms take a single ``poly_order``
   parameter that can either be a single number, in which case both the rows and columns
   will use the same polynomial degree, ie. :math:`d_r = d_c`, or a sequence
   of two numbers (:math:`d_r`, :math:`d_c`) to use different polynomials along
   the rows and columns. Further, ``max_cross`` can be set to limit the polynomial
   coefficients for the cross terms.

Algorithms
----------

poly (Regular Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.poly`:
:ref:`explanation for the algorithm <algorithms/polynomial:poly (Regular Polynomial)>`.
Note that the plot below is just the least-squared polynomial fit of the data
without masking.

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

    baseline, params = baseline_fitter.poly(y, poly_order=(1, 2), max_cross=0)
    create_plots(y, baseline)


modpoly (Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.modpoly`:
:ref:`explanation for the algorithm <algorithms/polynomial:modpoly (Modified Polynomial)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.modpoly(y, poly_order=(1, 2), max_cross=0)
    create_plots(y, baseline)


imodpoly (Improved Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.imodpoly`:
:ref:`explanation for the algorithm <algorithms/polynomial:imodpoly (Improved Modified Polynomial)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.imodpoly(y, poly_order=(1, 2), max_cross=0)
    create_plots(y, baseline)


penalized_poly (Penalized Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.penalized_poly`:
:ref:`explanation for the algorithm <algorithms/polynomial:penalized_poly (Penalized Polynomial)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.penalized_poly(y, poly_order=(1, 2), max_cross=0)
    create_plots(y, baseline)


quant_reg (Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.quant_reg`:
:ref:`explanation for the algorithm <algorithms/polynomial:quant_reg (Quantile Regression)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.quant_reg(
        y, poly_order=(1, 2), max_cross=0, quantile=0.3
    )
    create_plots(y, baseline)
