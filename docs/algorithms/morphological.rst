=======================
Morphological Baselines
=======================

The contents of :mod:`pybaselines.morphological` contain algorithms that
use morphological operations for estimating the baseline.

Introduction
------------

`Morphological operations <https://en.wikipedia.org/wiki/Mathematical_morphology>`_
include dilation, erosion, opening, and closing. Similar to the algorithms in
:mod:`pybaselines.window`, morphological operators use moving windows and compute
the maximum, minimum, or a combination of the two within each window.

.. note::
   All morphological algorithms use a `half_window` parameter to define the size
   of the window used for the morphological operators. `half_window` is index-based,
   rather than based on the units of the data, so proper conversions must be done
   by the user to get the desired window size.


Algorithms
----------

mpls (Morphological Penalized Least Squares)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mor (Morphological)
~~~~~~~~~~~~~~~~~~~

imor (Improved Morphological)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mormol (Morphological and Mollified Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

amormol (Averaging Morphological and Mollified Baseline)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rolling_ball (Rolling Ball)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
