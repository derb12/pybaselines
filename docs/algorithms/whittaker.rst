===================
Whittaker Baselines
===================

The contents of :mod:`pybaselines.whittaker` contain Whittaker-smoothing-based
algorithms for fitting the baseline.

Introduction
------------

Whittaker-smoothing-based (WSB) algorithms are usually referred to in literature
as weighted least squares or penalized least squares, but are referred to as WSB
in pybaselines to distinguish them from polynomial techniques that also take
advantage of weighted least squares (like :func:`.loess`) and penalized least
squares (like :func:`.penalized_poly`).

The general functional for WSB algorithms that is minimized to determine
the baseline is

.. math:: \sum\limits_{i = 1}^n w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{n - d} (\Delta^d z_i)^2

where :math:`y_i` is the measured data, :math:`z_i` is the estimated baseline,
:math:`\lambda` is the penalty scale factor, :math:`w_i` is the weighting, and
:math:`\Delta^d` is the finite-difference differential matrix of order d. Most
WSB techniques recommend using the second order differential matrix, although
some techniques use both the first and second order differential matrices.

The difference between most WSB algorithms is the selection of weights.

Algorithms
----------

asls (Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


iasls (Improved Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


airpls (Adaptive iteratively reweighted penalized least squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


arpls (Asymmetrically reweighted penalized least squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


drpls (Doubly reweighted penalized least squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


iarpls (Improved Asymmetrically reweighted penalized least squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


aspls (Adaptive smoothness penalized least squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

