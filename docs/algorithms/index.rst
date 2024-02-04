==========
Algorithms
==========

The currently available baseline correction algorithms in pybaselines can broadly be categorized
as polynomial, whittaker, morphological, smooth, spline, classification, optimizers,
and miscellaneous (misc) methods. Note that this is simply for grouping code and helping to
explain the internals of this library and **NOT** meant as a hard-classification of the
field of baseline correction (Please stop blindly copying this section in papers. There are
numerous types of baseline correction algorithms that are not included within pybaselines, which
is why baseline correction in general is such an absolutely fascinating field! Besides, miscellaneous
is obviously not an actual type of baseline correction...)

This section of the documentation is to help provide some context for each algorithm.
In addition, most algorithms will have a figure that shows how well the algorithm fits
various datasets to help choose the correct algorithm for a particular baseline. These datasets
include noisy data, data with both positive and negative peaks, data with overlapping peaks,
and concave data, and they serve as a way to quickly filter out algorithms that would not
work for a particular dataset.

Refer to the :doc:`API section <../api/index>` of the documentation for the full parameter and
reference listing for any algorithm.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   polynomial
   whittaker
   morphological
   spline
   smooth
   classification
   optimizers
   misc
