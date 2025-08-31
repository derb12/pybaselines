===================
Whittaker Baselines
===================

Introduction
------------

Excellent introductory papers on two dimensional penalized least squares are
`[1] <https://doi.org/10.1016/j.csda.2004.07.008>`_ and
`[2] <https://doi.org/10.48550/arXiv.2306.06932>`_. Whittaker-smoothing-based
algorithms are extended to two dimensional data as follows:

Let the number of rows be :math:`M` and the number of columns :math:`N` within the matrix
of measured data :math:`Y`. Note that :math:`y` is the flattened array of matrix :math:`Y`
with length :math:`M * N`. Analogous to the 1D case, the goal is to make the baseline match
the measured data as well as it can while also penalizing the roughness of the baseline, resulting
in the following minimization:

.. math::

    \sum\limits_{i}^M \sum\limits_{j}^N W_{ij} (Y_{ij} - V_{ij})^2
    + \lambda_r \sum\limits_{i}^{M - d_r} (V_{i\bullet} \Delta^{d_r})^2
    + \lambda_c \sum\limits_{j}^{N - d_c} (\Delta^{d_c} V_{j\bullet})^2

where :math:`Y_{ij}` is the measured data, :math:`V_{ij}` is the estimated baseline,
:math:`\lambda_r` is the penalty along the rows, :math:`\lambda_c` is the penalty along the columns,
:math:`W_{ij}` is the weighting, :math:`\Delta^{d_r}` is the finite-difference operator of order
:math:`d_r` along each row of :math:`V`, :math:`V_{i\bullet}`, and :math:`\Delta^{d_c}` is the
finite-difference operator of order :math:`d_c` along each column of :math:`V`, :math:`V_{j\bullet}`.

The resulting linear equation for solving the above minimization is:

.. math::

    (W_{diag} + \lambda_r I_M \otimes D_{d_r}^{\mathsf{T}} D_{d_r} + \lambda_c D_{d_c}^{\mathsf{T}} D_{d_c} \otimes I_M) v = w y


where :math:`W_{diag}` is the diagaonal matrix of the flattened weights, and :math:`D_d` is the matrix
version of :math:`\Delta^d`, as already explained for the :ref:`1D case <difference-matrix-explanation>`.
Further, :math:`\otimes` denotes the `Kronecker product <https://en.wikipedia.org/wiki/Kronecker_product>`_,
and :math:`I_M` and :math:`I_N` are the identity matrices of length :math:`M` and :math:`N`, respectively.
After solving, the array :math:`v` can then be reshaped into the matrix :math:`V`.

Since the analytical solution for 2D requires matrices of shape :math:`(M*N, M*N)`, it is quite
memory and computationally expensive to solve. Although the left hand side of the equation is
still sparse and symmetric, it cannot be solved as easily compared to the 1D case since the
bandwidth is no longer small due to the penalties along both the rows and columns (plus the
sparse solver currently available in SciPy cannot make use of the symmetric nature of the matrix;
using `Cholesky factorization <https://github.com/scikit-sparse/scikit-sparse>`_ does provide a speed
up but still does not scale well above ~500x500 sized matrices). However...

Eigendecomposition
~~~~~~~~~~~~~~~~~~

By following the excellent insights laid out by G. Biessy in `[2] <https://doi.org/10.48550/arXiv.2306.06932>`_,
the dimensionality of the system can be reduced by using eigendecomposition on each of the two
penalty matrices, :math:`D_{d_r}^{\mathsf{T}} D_{d_r}` and :math:`D_{d_c}^{\mathsf{T}} D_{d_c}`. (Note that speeding up
Whittaker smoothing using `factorization in 1D <https://doi.org/10.1016/j.csda.2006.11.038>`_ and using the
`analytical eigenvalues in nD (great paper) <https://doi.org/10.1016/j.csda.2009.09.020>`_ are established
methods, although they require using a fixed difference order, and, in the second case, of using
different boundary conditions that unfortunately do not translate well from smoothing to baseline correction).
The general eigendecomposition of the penalty matrix gives

.. math::

    D_{d}^{\mathsf{T}} D_{d} = U \Sigma U^{\mathsf{T}}

where :math:`U` is the matrix of eigenvectors and :math:`\Sigma` is a diagonal matrix
with the eigenvalues along the diagonal. Letting :math:`B = U_c \otimes U_r` denote the Kronecker
product of the eigenvector matrices of the penalty for the columns and rows, and :math:`g` and
:math:`h` denote the number of eigenvectors along the rows and columns, respectively, the linear equation
can be rewritten as:

.. math::

    (B^{\mathsf{T}} W_{diag} B + \lambda_r I_h \otimes \Sigma_r + \lambda_c \Sigma_c \otimes I_g) \alpha = B^{\mathsf{T}} W_{diag} y

and the baseline is then:

.. math::

    v = B \alpha

The beauty of this reparameterization when applied to baseline correction is twofold:

1) The number of eigenvalues required to approximate the analytical solution depends on
   the required smoothness, ie. some constant approximated by :math:`\lambda / (\text{number of data points})`
   that does not appreciably change with data size. Baselines require much less curvature than
   smoothing, so the number of eigenvalues is relatively low (from testing, ~5-10 for low order
   polynomial baselines and ~15-25 for sinusoidal baselines).
2) Since experimental data is measured on gridded data (ie. :math:`Y_{ij} = f(x_i, z_j)`), the
   above equation can be further optimized by expressing it as a
   `generalized linear array model <https://en.wikipedia.org/wiki/Generalized_linear_array_model>`_,
   following the brilliant insights of `Eilers, Currie, and Durbán <https://doi.org/10.1016/j.csda.2004.07.008>`_,
   exactly as :ref:`explained for 2D penalized splines <generalized-linear-array-model-explanation>`.

:ref:`An example <sphx_glr_generated_examples_two_d_plot_whittaker_2d_dof.py>` examines how to determine
the approximate number of eigenvalues required to represent different baselines.


.. note::
   For two dimensional data, Whittaker-smoothing-based algorithms take a single ``lam``,
   parameter that can either be a single number, in which case both the rows and columns
   will use the same smoothing parameter, ie. :math:`\lambda_r = \lambda_c`, or a sequence
   of two numbers (:math:`\lambda_r`, :math:`\lambda_c`) to use different values for the
   rows and columns.

Algorithms
----------

asls (Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.asls`:
:ref:`explanation for the algorithm <algorithms/whittaker:asls (Asymmetric Least Squares)>`.

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

    baseline, params = baseline_fitter.asls(y, lam=(1e2, 1e1), p=0.001)
    create_plots(y, baseline)


iasls (Improved Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.iasls`:
:ref:`explanation for the algorithm <algorithms/whittaker:iasls (Improved Asymmetric Least Squares)>`.
Eigendecomposition is not allowed for this method.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_data function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.iasls(y, lam=(1e3, 1e0))
    create_plots(y, baseline)


airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.airpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:airpls (Adaptive Iteratively Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.airpls(y, lam=(1e3, 1e1))
    create_plots(y, baseline)


arpls (Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.arpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:arpls (Asymmetrically Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.arpls(y, lam=(1e4, 1e2))
    create_plots(y, baseline)


drpls (Doubly Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.drpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:drpls (Doubly Reweighted Penalized Least Squares)>`.
Eigendecomposition is not allowed for this method.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.drpls(y, lam=(1e3, 1e2))
    create_plots(y, baseline)


iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.iarpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.iarpls(y, lam=(1e3, 1e2))
    create_plots(y, baseline)


aspls (Adaptive Smoothness Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.aspls`:
:ref:`explanation for the algorithm <algorithms/whittaker:aspls (Adaptive Smoothness Penalized Least Squares)>`.
Eigendecomposition is not allowed for this method.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.aspls(y, lam=(1e3, 1e2))
    create_plots(y, baseline)


psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.psalsa`:
:ref:`explanation for the algorithm <algorithms/whittaker:psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.psalsa(y, lam=(1e3, 1e2), k=0.5)
    create_plots(y, baseline)


brpls (Bayesian Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.brpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:brpls (Bayesian Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.brpls(y, lam=(1e4, 1e2))
    create_plots(y, baseline)


lsrpls (Locally Symmetric Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~.Baseline2D.lsrpls`:
:ref:`explanation for the algorithm <algorithms/whittaker:lsrpls (Locally Symmetric Reweighted Penalized Least Squares)>`.

.. plot::
   :align: center
   :context: close-figs

    # to see contents of create_plots function, look at the top-most algorithm's code
    baseline, params = baseline_fitter.lsrpls(y, lam=(1e4, 1e2))
    create_plots(y, baseline)
