====================
Polynomial Baselines
====================

The contents of :mod:`pybaselines.polynomial` contain algorithms for fitting
polynomials to the baseline.

Introduction
------------

A polynomial can be expressed as

.. math::

    p(x) = \beta_0 x^0 + \beta_1 x^1 + \beta_2 x^2 + ... + \beta_m x^m = \sum\limits_{j = 0}^m {\beta_j x^j}

where :math:`\beta` is the array of coefficients for the polynomial.

For regular polynomial fitting, the polynomial coefficients that best fit data
are gotten from minimizing the least-squares:

.. math:: \sum\limits_{i = 1}^n w_i^2 (y_i - p(x_i))^2

where :math:`y_i` and :math:`x_i` are the measured data, :math:`p(x_i)` is
the polynomial estimate at :math:`x_i`, and :math:`w_i` is the weighting.

However, since only the baseline of the data is desired, the least-squares
approach must be modified. For polynomial-based algorithms, this is done
by 1) only fitting the data in regions where there is only baseline (termed
selective masking), 2) modifying the y-values being fit each iteration, termed
thresholding, or 3) penalyzing outliers.

Selective Masking
~~~~~~~~~~~~~~~~~

Selective masking is the oldest and most basic of the techniques. There
are two ways to use selective masking in pybaselines. First, the input dataset
can be trimmed/masked (easy to do with numpy) to not include any peak regions,
the masked data can be fit, and then the resulting coefficients can be used to
create a polynomial that spans the entirety of the original dataset. The second
way is to keep the original data, and input a custom weight array into the
fitting function with values equal to 0 in peak regions and 1 in baseline regions.

The only algorithm in pybaselines that should use selective masking is
:func:`.poly`, which is normal polynomial least-squares fitting as described
above. However, other techniques allow inputting custom weights including
:func:`.modpoly` and :func:`.imodpoly`.

The use of selective masking is generally not encouraged since it is time consuming
to select the peak and non-peak regions in each set of data, and can lead to hard
to reproduce results.

Thresholding
~~~~~~~~~~~~

The algorithms in pybaselines that use thresholding are :func:`.modpoly`,
:func:`.imodpoly`, and :func:`.loess` (if `use_threshold` is True).

Penalyzing Outliers
~~~~~~~~~~~~~~~~~~~

The algorithms in pybaselines that penalyze outliers are
:func:`.penalized_poly`, which incorporate the penalty directly into the
minimized cost function, and :func:`.loess` (if `use_threshold` is False),
which incorporates penalties by applying lower weights to outliers.


Algorithms
----------

poly (Regular Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~

modpoly (Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

imodpoly (Improved Modified Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

penalized_poly (Penalized Polynomial)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

loess (Locally Estimated Scatterplot Smoothing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
