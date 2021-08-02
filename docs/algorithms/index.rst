==========
Algorithms
==========

The currently available baseline correction algorithms in pybaselines are split into
polynomial, whittaker, morphological, window, classification, optimizers, and miscellaneous (misc).
Note that this is more for grouping code and not meant as a hard-classification
of the algorithms.

This section of the documentation is to help provide some context for each algorithm.
In addition, most algorithms will have a figure that shows how well the algorithm fits
various baselines to help choose the correct algorithm for a particular baseline. Refer
to the :doc:`API section <../api/index>` of the documentation for the full parameter and
reference listing for any algorithm.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   polynomial
   whittaker
   morphological
   window
   classification
   optimizers
   misc
