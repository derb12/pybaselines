=========
Changelog
=========

Version 0.3.0 (2021-04-29)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

New Features
~~~~~~~~~~~~

* Added the small-window moving average (swima) baseline to pybaselines.window,
  which iteratively smooths the data with a moving average to eliminate peaks
  and obtain the baseline.
* Added the rolling_ball function to pybaselines.morphological, which applies
  a minimum and then maximum moving window, and subsequently smooths the result,
  giving a baseline that resembles rolling a ball across the data. Also allows
  giving an array of half-window values to allow the ball to change size as it
  moves across the data.
* Added the adaptive_minmax algorithm to pybaselines.optimizers, which uses the
  modpoly or imodpoly functions and performs polynomial fits with two different
  orders and two different weighting schemes and then uses the maximum values of
  all the baselines.
* Added the Peaked Signal's Asymmetric Least Squares Algorithm (psalsa)
  function to pybaselines.whittaker, which uses exponentially decaying weighting
  to better fit noisy data.
* The imodpoly and loess functions in pybaselines.polynomial now use `num_std`
  to specify the number of standard deviations to use when thresholding.
* The pybaselines.polynomial.penalized_poly function now allows weights to be used.
  Also made the default threshold value scale with the data better.
* Added higher order filters for pybaselines.window.snip to allow for more
  complicated baselines. Also allow inputting a sequence of ints for
  `max_half_window` to better fit asymmetric peaks.

Bug Fixes
~~~~~~~~~

* Fixed a bug that would not allow even morphological half windows,
  since it is not needed for the half windows, only the full windows.
* Fixed the thresholding for pybaselines.polynomial.imodpoly, which was incorrectly
  not adding the standard deviation to the baseline when thresholding.
* Fixed weighting for pybaselines.whittaker.airpls so that weights no longer
  get values greater than 1.
* Removed the append and prepend keywords for np.diff in the
  pybaselines.morphological.mpls function, since the keywords
  were not added until numpy version 1.16, which is higher than
  the minimum stated version for pybaselines.

Other Changes
~~~~~~~~~~~~~

* Allow utils.pad_edges to work with a pad_length of 0 (no padding).
* Added a 'min_half_window' parameter for pybaselines.morphological.optimize_window
  so that small window sizes can be skipped to speed up the calculation.
* Changed the default method from 'aspls' to 'asls' for optimizers.optimize_extended_range.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Removed the 'smooth' keyword argument for pybaselines.window.snip. Smoothing is
  now performed if the given smooth half window is greater than 0.
* pybaselines.polynomial.loess no longer has an `include_stdev` keyword argument.
  Equivalent behavior can be obtained by setting `num_std` to 0.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Updated the documentation to include simple explanations for some techniques.


Version 0.2.0 (2021-04-02)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

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
