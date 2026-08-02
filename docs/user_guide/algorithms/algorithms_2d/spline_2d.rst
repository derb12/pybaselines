================
Spline Baselines
================

Introduction
------------

The two dimensional extension of penalized splines (P-splines) for baseline correction
within pybaselines follows the framework of Eilers, Currie, and Durbán
from `[1] <https://doi.org/10.1016/j.csda.2004.07.008>`_.

Let the number of rows be :math:`M` and the number of columns :math:`N` within the matrix
of measured data :math:`Y`. Note that :math:`y` is the flattened array of matrix :math:`Y`
with length :math:`M * N`. Let :math:`Y` be a function of :math:`x` along the rows and :math:`z`
along the columns, ie. :math:`Y_{ij} = f(x_i, z_j)`, and :math:`B_r(x)` and :math:`B_c(z)` represent
the spline basis matrices along the rows and columns, respectively, each with a number of
knots :math:`g` and `h`. Analogous to the 1D case, the goal is to make the baseline, :math:`V` match the measured
data as well as it can  while also penalizing the difference between spline coefficients, resulting
in the following minimization:

.. math::

    \sum\limits_{i}^M \sum\limits_{j}^N W_{ij} (Y_{ij} - \sum\limits_{k}^g \sum\limits_{l}^h B_{r,k}(x_i) B_{c,l}(z_j) \alpha_{kl})^2
    + \lambda_r \sum\limits_{i}^{g - d_r} (\alpha_{i\bullet} \Delta^{d_r})^2
    + \lambda_c \sum\limits_{j}^{h - d_c} (\Delta^{d_c} \alpha_{j\bullet})^2

and

.. math::

    V = \sum\limits_{i}^g \sum\limits_{j}^h B_{r,i} B_{c,j} \alpha_{ij}


where :math:`Y_{ij}` is the measured data, :math:`\alpha` is the matrix of spline coefficients,
:math:`\lambda_r` is the penalty along the rows, :math:`\lambda_c` is the
penalty along the columns, :math:`W_{ij}` is the weighting, :math:`\Delta^{d_r}` is the finite-difference
operator of order :math:`d_r` along each row of :math:`\alpha`, :math:`\alpha_{i\bullet}`, and :math:`\Delta^{d_c}` is the
finite-difference operator of order :math:`d_c` along each column of :math:`\alpha`, :math:`\alpha_{j\bullet}`.

Let :math:`B = B_c \otimes B_r` denote the Kronecker product of the basis matrices for the columns and rows,
which represents the overall two dimensional tensor product spline basis. The resulting linear equation for
solving the above minimization is:

.. math::

    (B^{\mathsf{T}} W_{diag} B + \lambda_r I_h \otimes D_{d_r}^{\mathsf{T}} D_{d_r} + \lambda_c D_{d_c}^{\mathsf{T}} D_{d_c} \otimes I_g) \alpha = B^{\mathsf{T}} W_{diag} y

and the baseline is then:

.. math::

    v = B \alpha

where :math:`W_{diag}` is the diagaonal matrix of the flattened weights, :math:`v` is the flattened
estimated baseline, and :math:`D_d` is the matrix version of :math:`\Delta^d`, as already explained for
the :ref:`1D case <difference-matrix-explanation>`. Further, :math:`\otimes` denotes the Kronecker
product, and :math:`I_g` and :math:`I_h` are the identity matrices of length :math:`g` and
:math:`h`, respectively. After solving, the array :math:`v` can then be reshaped into the matrix :math:`V`.

.. _generalized-linear-array-model-explanation:

Since experimental data is measured on gridded data (ie. :math:`Y_{ij} = f(x_i, z_j)`), the above equation
can be optimized following `[1] <https://doi.org/10.1016/j.csda.2004.07.008>`_ and expressed as a
`generalized linear array model <https://wikipedia.org/wiki/Generalized_linear_array_model>`_
which allows directly using the matrices of the measured data, :math:`Y`, and the weights,
:math:`W`, rather than flattening them, and significantly reduces the required
memory and computation time.

Let :math:`F` be the
`face-splitting product operator <https://wikipedia.org/wiki/Khatri%E2%80%93Rao_product#Face-splitting_product>`_
of a matrix with itself such that :math:`F(B_r) = (B_r \otimes 1_{g}^{\mathsf{T}}) \odot (1_{g}^{\mathsf{T}} \otimes B_r)`
and :math:`F(B_c) = (B_c \otimes 1_{h}^{\mathsf{T}}) \odot (1_{h}^{\mathsf{T}} \otimes B_c)`, where
:math:`1_g` and :math:`1_h` are vectors of ones of length :math:`g` and :math:`h`, respectively,
and :math:`\odot` signifies elementwise multiplication. Then the linear equation can be rewritten as:

.. math::

    (F(B_r)^{\mathsf{T}} W F(B_c) + \lambda_r I_h \otimes D_{d_r}^{\mathsf{T}} D_{d_r} + \lambda_c D_{d_c}^{\mathsf{T}} D_{d_c} \otimes I_g) \alpha = B_{r}^{\mathsf{T}} (W \odot Y) B_c

and the baseline is:

.. math::

    V = B_r \alpha B_{c}^{\mathsf{T}}


Algorithms
----------

mixture_model (Mixture Model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.mixture_model`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:mixture_model (Mixture Model)>`.

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

    baseline, params = baseline_fitter.mixture_model(y, lam=(1e3, 1e2))
    create_plots(y, baseline)


irsqr (Iterative Reweighted Spline Quantile Regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.irsqr`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:irsqr (Iterative Reweighted Spline Quantile Regression)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.irsqr(y, lam=(1e3, 1e2), quantile=0.3)
    create_plots(y, baseline)


pspline_asls (Penalized Spline Version of asls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_asls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_asls (Penalized Spline Version of asls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_asls(y, lam=(1e3, 1e0), p=0.005)
    create_plots(y, baseline)


pspline_iasls (Penalized Spline Version of iasls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_iasls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_iasls (Penalized Spline Version of iasls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_iasls(y, lam=(1e2, 1e-2))
    create_plots(y, baseline)


pspline_airpls (Penalized Spline Version of airpls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_airpls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_airpls (Penalized Spline Version of airpls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_airpls(y, lam=(1e3, 1e-1))
    create_plots(y, baseline)


pspline_arpls (Penalized Spline Version of arpls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_arpls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_arpls (Penalized Spline Version of arpls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_arpls(y, lam=(1e3, 5e0))
    create_plots(y, baseline)


pspline_iarpls (Penalized Spline Version of iarpls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_iarpls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_iarpls (Penalized Spline Version of iarpls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_iarpls(y, lam=(1e2, 1e0))
    create_plots(y, baseline)


pspline_psalsa (Penalized Spline Version of psalsa)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_psalsa`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_psalsa (Penalized Spline Version of psalsa)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_psalsa(y, lam=(1e3, 5e0), k=0.5)
    create_plots(y, baseline)


pspline_brpls (Penalized Spline Version of brpls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_brpls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_brpls (Penalized Spline Version of brpls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_brpls(y, lam=(1e3, 5e0))
    create_plots(y, baseline)


pspline_lsrpls (Penalized Spline Version of lsrpls)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.pspline_lsrpls`:
:ref:`explanation for the algorithm <algorithms/algorithms_1d/spline:pspline_lsrpls (Penalized Spline Version of lsrpls)>`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: True

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.pspline_lsrpls(y, lam=(1e3, 5e0))
    create_plots(y, baseline)
