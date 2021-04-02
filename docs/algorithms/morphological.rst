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
