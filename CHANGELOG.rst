=========
Changelog
=========

Version 0.2.0 (2021-04-02)
--------------------------

New Features
~~~~~~~~~~~~

* Added the morphological and mollified (mormol) function to pybaselines.morphological,
  which uses a combination of morphology for baseline estimation and mollification for
  smoothing.
* Added the loess function to pybaselines.polynomial, which does local robust polynomial
  fitting. Allows using symmetric or asymmetric weighting, or using thresholding, similar
  to the modpoly and imodpoly functions.
* Added the penalized_poly function to pybaselines.polynomial, which fits a polynomial baseline
  using a non-quadratic cost function. The non-quadratic cost functions include
  huber, truncated-quadratic, and indec, and can be either symmetric or asymmetric.
* Added options for padding data when doing convolution or window-based
  operations to reduce edge effects and give better results.

Bug Fixes
~~~~~~~~~

* Fixed the mollification kernel used for the morphological.iamor (now amormol) function.
* Fixed a miscalculation with the weighting for whittaker.aspls.

Other Changes
~~~~~~~~~~~~~

* Slightly sped up several functions in whittaker.py by precomputing terms.
* Added tests for all baseline algorithms

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Renamed morphology.iamor to morphology.amormol (averaging morphological and
  mollified baseline) to make it more clear that mormol and amormol are similar methods.
* Renamed penalized_least_squares.py to whittaker.py, to be more specific, since other
  techniques also use penalized least squares for polynomial fitting.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Updated the example program to match the changes to pybaselines.
* Setup initial documentation.


Version 0.1.0 (2021-03-22)
--------------------------

* Initial release on PyPI.
