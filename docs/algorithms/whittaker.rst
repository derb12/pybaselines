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

The general idea behind WSB algorithms is to make the baseline match the measured
data as well as it can while also penalizing the roughness of the baseline. The
resulting general function that is minimized to determine the baseline is then

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

where :math:`y_i` is the measured data, :math:`z_i` is the estimated baseline,
:math:`\lambda` is the penalty scale factor, :math:`w_i` is the weighting, and
:math:`\Delta^d` is the finite-difference differential matrix of order d. Most
WSB techniques recommend using the second order differential matrix, although
some techniques use both the first and second order differential matrices.

The difference between WSB algorithms is the selection of weights and/or the
function that is minimized.

Algorithms
----------

asls (Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`.asls` (sometimes called ALS in literature) function is the
original implementation of Whittaker smoothing for baseline fitting.

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.


iasls (Improved Asymmetric Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N (w_i (y_i - z_i))^2
    + \lambda \sum\limits_{i = 1}^{N - 2} (\Delta^2 z_i)^2
    + \lambda_1 \sum\limits_{i = 1}^{N - 1} (\Delta^1 (y_i - z_i))^2

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.


airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        0 & y_i \ge z_i \\
        exp{\left(\frac{t (y_i - z_i)}{|\mathbf{d}^-|}\right)} & y_i < z_i
    \end{array}\right.

where :math:`t` is the iteration number and :math:`|\mathbf{d}^-|` is the l1-norm of the negative
values in the residual vector :math:`\mathbf d`, ie. :math:`\sum\limits_{y_i - z_i < 0} |y_i - z_i|`.

arpls (Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

Weighting:

.. math::

    w_i = \frac
        {1}
        {exp{\left(\frac
            {2(d_i - (-\mu^- + 2 \sigma^-))}
            {\sigma^-}
        \right)}}

where :math:`d_i = y_i - z_i` and :math:`\mu^-` and :math:`\sigma^-` are the mean and standard
deviation, respectively, of the negative values in the residual vector :math:`\mathbf d`.


drpls (Doubly Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2
    + \lambda \sum\limits_{i = 1}^{N - 2}(1 - \eta w_i) (\Delta^2 z_i)^2
    + \sum\limits_{i = 1}^{N - 1} (\Delta^1 (z_i))^2

where :math:`\eta` is a value between 0 and 1 that controls the
effective value of :math:`\lambda`.

Weighting:

.. math::

    w_i = \frac{1}{2}\left(
        1 -
        \frac
            {exp(t)(d_i - (-\mu^- + 2 \sigma^-))/\sigma^-}
            {1 + abs[exp(t)(d_i - (-\mu^- + 2 \sigma^-))/\sigma^-]}
    \right)

where :math:`d_i = y_i - z_i`, :math:`t` is the iteration number, and
:math:`\mu^-` and :math:`\sigma^-` are the mean and standard deviation,
respectively, of the negative values in the residual vector :math:`\mathbf d`.


iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

Weighting:

.. math::

    w_i = \frac{1}{2}\left(
        1 -
        \frac
            {exp(t)(d_i - 2 \sigma^-)/\sigma^-}
            {\sqrt{1 + [exp(t)(d_i - 2 \sigma^-)/\sigma^-]^2}}
    \right)

where :math:`d_i = y_i - z_i`, :math:`t` is the iteration number, and
:math:`\sigma^-` is the standard deviation of the negative values in
the residual vector :math:`\mathbf d`.


aspls (Adaptive Smoothness Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2
    + \lambda \sum\limits_{i = 1}^{N - d} \alpha_i (\Delta^d z_i)^2

where

.. math::

    \alpha_i = \frac
        {abs(y_i - z_i)}
        {max(abs(y_i - z_i))}

Weighting:

.. math::

    w_i = \frac
        {1}
        {1 + exp{\left[
            2(d_i - \sigma^-) / \sigma^-
        \right]}}

where :math:`d_i = y_i - z_i`  and :math:`\sigma^-` is the standard deviation
of the negative values in the residual vector :math:`\mathbf d`.


psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimized function:

.. math::

    \sum\limits_{i = 1}^N w_i (y_i - z_i)^2 + \lambda \sum\limits_{i = 1}^{N - d} (\Delta^d z_i)^2

Weighting:

.. math::

    w_i = \left\{\begin{array}{cr}
        p \cdot exp{\left(\frac{-(y_i - z_i)}{k}\right)} & y_i > z_i \\
        1 - p & y_i \le z_i
    \end{array}\right.

where :math:`k` is a factor that controls the exponential decay of the weights for baseline
values greater than the data and should be approximately the height at which a value could
be considered a peak.
