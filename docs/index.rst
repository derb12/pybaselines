pybaselines Documentation
=========================

pybaselines is a collection of algorithms for fitting the baseline of experimental data.

* For Python 3.6+
* Open Source: BSD 3-Clause License
* Source Code: https://github.com/derb12/pybaselines
* Documentation: https://pybaselines.readthedocs.io.


Baseline fitting techniques are grouped accordingly (note: when a method
is labelled as 'improved', that is the method's name, not editorialization):

a) Whittaker-smoothing-based techniques (:mod:`pybaselines.whittaker`)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)

b) Morphological (:mod:`pybaselines.morphological`)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) iamor (Iterative averaging morphological)

c) Polynomial (:mod:`pybaselines.polynomial`)

    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)

d) Window-based (:mod:`pybaselines.window`)

    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/index
   contributing
   changes
   license
   authors


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
