=============
2D Algorithms
=============

pybaselines extends a subset of the 1D baseline correction algorithms to work with
2D data. Note that this is only intended for data in which there is some global baseline;
otherwise, it is more appropriate and usually significantly faster to simply use the 1D
algorithms on each individual row and/or column in the data, which can be done using
:meth:`~.Baseline2D.individual_axes`.

This section of the documentation is to help provide some context for how the algorithms
were extended to work with two dimensional data. It will not be as comprehensive as the
:doc:`1D Algorithms section <../algorithms/index>`, so to help understand any algorithm,
it is suggested to start there. Refer to the :doc:`API section <../api/index>` of the
documentation for the full parameter and reference listing for any algorithm.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   polynomial_2d
   whittaker_2d
   morphological_2d
   spline_2d
   smooth_2d
   optimizers_2d
