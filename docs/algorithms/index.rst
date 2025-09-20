==========
Algorithms
==========

The currently available baseline correction algorithms in pybaselines can broadly be categorized
as polynomial, whittaker, morphological, smooth, spline, classification, optimizers,
and miscellaneous (misc) methods. Note that this is simply for grouping code and helping to
explain the internals of this library and **NOT** meant as a hard-classification of the
field of baseline correction. In reality, many algorithms overlap into several categories,
and there are numerous different methods not implemented in pybaselines that do not fit in
any of those categories, which is why baseline correction in general is such an absolutely
fascinating field!

The goal of this section of the documentation is to help provide some context for each algorithm
and show how each fits various different types of data for a quick, albeit shallow, comparison among
all algorithms. The two dimensional (2D) section only explains how 1D algorithms were adapted to
fit 2D data and is thus overall more sparse on details, so it is suggested to start with reading
through the 1D section to provide a complete understanding.

Refer to the :doc:`API section <../api/index>` of the documentation for the full parameter and
reference listing for any algorithm.


.. toctree::
   :maxdepth: 2

   algorithms_1d/index
   algorithms_2d/index
