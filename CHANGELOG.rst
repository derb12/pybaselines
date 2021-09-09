=========
Changelog
=========

Version 0.6.0 (2021-09-09)
--------------------------

This is a minor version with new features, deprecations, and documentation improvements.

New Features
~~~~~~~~~~~~

* Added goldindec to pybaselines.polynomial, which uses a non-quadratic cost
  function with a shrinking threshold to fit the baseline.
* Added the morphological penalized spline (mpspline) algorithm to
  pybaselines.morphological, which uses morphology to identify baseline points
  and then fits the points using a penalized spline.
* Added the derivative peak-screening asymmetric least squares algorithm (derpsalsa)
  to pybaselines.whittaker, which includes additional weights based on the first and
  second derivatives of the data.
* Added the fastchrom algorithm to pybaselines.classification, which identifies baseline
  points as where the rolling standard deviation is less than the specified threshold.
* Added the module pybaselines.spline, which contains algorithms that use splines
  to create the baseline.
* Added the mixture model algorithm (mixture_model) to pybaselines.spline, which uses
  a weighted penalized spline to fit the baseline, where weights are calculated based
  on the probability each point belongs to the noise.
* Added iterative reweighted spline quantile regression (irsqr) to pybaselines.spline,
  which uses penalized splines and iterative reweighted least squares to perform
  quantile regression on the data.
* Added the corner-cutting algorithm (corner_cutting) to pybaselines.spline, which
  iteratively removes corner points and then fits a quadratic Bezier spline with the
  remaining points.

Other Changes
~~~~~~~~~~~~~

* Increased the minimum SciPy version to 0.17 in order to use bounds with
  scipy.optimize.curve_fit.
* Changed the default `extrapolate_window` value in pybaselines.utils.pad_edges to
  the input window length, rather than ``2 * window length + 1``.
* Slightly sped up pybaselines.optimizers.adaptive_minmax when `poly_order` is
  None by using the numpy array's min and max methods rather than the built-in
  functions.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Renamed pybaselines.window to pybaselines.smooth to make its usage more
  clear. Using pybaselines.window will still work for now, but will begin emitting
  a DeprecationWarning in a later version (maybe version 0.8 or 0.9) and will
  be removed shortly thereafter.
* Removed the constant utils.PERMC_SPEC that was deprecated in version 0.4.1.
* Deprecated the function pybaselines.morphological.optimize_window, which will
  be removed in version 0.8.0. Use pybaselines.utils.optimize_window instead.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Fixed the plot for morphological.mpls in the documentation.
* Fixed the weighting formula for whittaker.arpls in the documentation.
* Fixed a typo for the cost function in the docstring of misc.beads.
* Updated the example program for all of the newly added algorithms.


Version 0.5.1 (2021-08-10)
--------------------------

This is a minor patch with bug fixes and minor changes.

Bug Fixes
~~~~~~~~~

* Added classification to the main pybaselines namespace so that calling
  pybaselines.classification works correctly.

Other Changes
~~~~~~~~~~~~~

* Changed the default `tol` for pybaselines.polynomial.quant_reg to 1e-6
  to get better results.
* Directly use the input `eps` value for pybaselines.polynomial.quant_reg
  rather than its square.


Version 0.5.0 (2021-08-02)
--------------------------

This is a minor version with new features, bug fixes, and deprecations.

New Features
~~~~~~~~~~~~

* Added quantile regression (quant_reg) to pybaselines.polynomial, which uses quantile
  regression to fit a polynomial to the baseline.
* Added the top-hat transformation (tophat) to pybaselines.morphological, which estimates
  the baseline using the morphological opening.
* Added the moving-window minimum value (mwmv) pybaseline.morphological, which estimates the
  baseline using the rolling minimum values.
* Added the baseline estimation and denoising with sparsity (beads) method to pybaselines.misc,
  which decomposes the input data into baseline and pure, noise-free signal by modeling the
  baseline as a low pass filter and by considering the signal and its derivatives as sparse.
* Added the module pybaselines.classification, which contains algorithms that
  classify baseline and/or peak segments to create the baseline.
* Added Dietrich's classification method (dietrich) to pybaselines.classification,
  which classifies baseline points by analyzing the power spectrum of the data's
  derivative and then iteratively fits the points with a polynomial.
* Added Golotvin's classification method (golotvin) to pybaselines.classification,
  which breaks the data into segments, uses the minimum standard deviation of all
  the segments to define the standard deviation of the entire data, and then
  classifies baseline points using that value.
* Added the standard deviation distribution method (std_distribution) to
  pybaselines.classification, which classifies baseline segments by grouping the
  rolling standard deviation values into a distribution for the baseline and a
  distribution for the signal.
* Added Numba as an optional dependency. Currently, the functions pybaselines.polynomial.loess,
  pybaselines.classification.std_distribution, and pybaselines.misc.beads are faster when Numba
  is installed.
* When Numba is installed, the pybaselines.polynomial.loess calculation is done
  in parallel, which greatly improves the speed of the calculation.
* The pybaselines.polynomial.loess function now takes a `delta` parameter, which will
  use linear interpolation rather than weighted least squares fitting for all but the
  last x-values that are less than `delta` from the last-fit x-value. Can significantly
  reduce calculation time.
* All iterative methods now return an array of the calculated tolerance value for each iteration
  in the dictionary output, which should help to pick appropriate `tol` and `max_iter` values.

Bug Fixes
~~~~~~~~~

* Added checks for airpls, drpls, and iarpls functions in pybaselines.whittaker to
  prevent nan or infinite weights in edge cases where too many iterations were done.
* The baseline returned from polynomial algorithms was the second-to-last iteration's baseline,
  rather than the last iteration's. Now the returned baseline is the last iteration's.
* Sort input weights and y0 (if `use_original` is True) for pybaselines.polynomial.loess
  after sorting the x-values, rather than leaving them unsorted.

Other Changes
~~~~~~~~~~~~~

* Added a custom ParameterWarning for when a user-input parameter is valid but
  outside the recommended range and could cause issues with a calculation.
* Changed the default `conserve_memory` value in polynomial.loess to True, since
  it is just as fast as False when Numba is installed and is safer.
* pybaselines.optimizers.collab_pls now includes the parameters from each function
  call in the dictionary output as items in lists.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The key for the averaged weights for pybaselines.optimizers.collab_pls is now
  'average_weights' to avoid clashing with the 'weights' key from the called function.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Most algorithms in the documentation now include several plots showing how
  the algorithm fits different types of baselines.
* Added more in-depth explanations for all baseline correction algorithms.


Version 0.4.1 (2021-06-10)
--------------------------

This is a minor patch with new features, bug fixes, and pending deprecations.

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

* Switched to using banded solvers for all Whittaker-smoothing-based algorithms
  (all functions in pybaselines.whittaker as well as pybaselines.morphological.mpls),
  which reduced their computation time by ~60-85% compared to version 0.4.0.
* Added pentapy as an optional dependency. All Whittaker-smoothing-based functions
  will use pentapy's solver, which is faster than SciPy's solve_banded and solveh_banded
  functions, if pentapy is installed and the system is pentadiagonal (`diff_order` is 2).
  All Whittaker functions with pentapy installed take ~80-95% less time compared to
  pybaselines version 0.4.0.

Bug Fixes
~~~~~~~~~

* The `alpha` item in the dictionary output of whittaker.aspls is now the full alpha
  array rather than a single value.
* The weighting for several Whittaker-smoothing-based functions was made more robust
  and less likely to create nan weights.

Other Changes
~~~~~~~~~~~~~

* Increased the default `max_iter` for whittaker.aspls to 100.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The constant pybaselines.utils.PERMC_SPEC is no longer used. It will be removed
  in version 0.6.0.


Version 0.4.0 (2021-05-30)
--------------------------

This is a minor version with new features, bug fixes, and deprecations.

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

* Significantly reduced both the calculation time and memory usage of polynomial.loess.
  For example, getting the baseline for a dataset with 20,000 points now takes ~12 seconds
  and ~0.7 GB of memory compared to ~55 seconds and ~3 GB of memory in version 0.3.0.
* Added a `conserve_memory` parameter to polynomial.loess that will recalculate the distance
  kernels each iteration, which is slower than the default but uses very little memory. For
  example, using loess with `conserve_memory` set to True on a dataset with 20,000 points
  takes ~18 seconds while using ~0 GB of memory.
* Allow more user inputs for optimizers.optimize_extended_range to allow specifying the range
  of `lam`/`poly_order` values to test and to have more control over the added lines and
  Gaussians on the sides.
* Added a constant called PERMC_SPEC (accessed from pybaselines.utils.PERMC_SPEC),
  which is used by SciPy's sparse solver when using Whittaker-smoothing-based algorithms.
  Changed the default value to "NATURAL", which reduced the computation time of all
  Whittaker-smoothing-based algorithms by ~5-35% compared to other permc_spec options
  on the tested system.
* misc.interp_pts (formerly manual.linear_interp) now allows specifying any interpolation
  method supported by scipy.interpolate.interp1d, allowing for methods such as spline
  interpolation.

Bug Fixes
~~~~~~~~~

* Fixed poly_order calculation for optimizers.adaptive_minmax when poly_order was a
  single item within a container.
* Potential fix for namespace error with utils; accessing pybaselines.utils gave an
  attribute error in very specific envinronments, so changed the import order in
  pybaselines.__init__ to potentially fix it. Updated the quick start example in case
  the fix is not correct so that the example will still work.
* Increased minimum NumPy version to 1.14 to use rcond=None with numpy.linalg.lstsq.

Other Changes
~~~~~~~~~~~~~

* polynomial.loess now allows inputting weights, specifying a `use_original` keyword for
  thresholding to match the modpoly and imodpoly functions, and specifying a `return_coef`
  keyword to allow returning the polynomial coefficients for each x-value to recreate
  the fitted polynomial, to match all other polynomial functions.
* Changed the default `smooth_half_window` value in window.noise_median, window.snip, and
  morphological.mormol to None, rather than being fixed values. Each function sets its default
  slightly different but still follows the behavior in previous versions, except for
  window.noise_median as noted below.
* Changed default `smooth_half_window` value for window.noise_median to match specified
  `half_window` value rather than 1.
* Changed default `sigma` value for window.noise_median to scale with the specified
  `smooth_half_window`, rather than being a fixed value.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Renamed pybaselines.manual to pybaselines.misc to allow for adding any future
  miscellaneous algorithms that will not fit elsewhere.
* Renamed the manual.linear_interp function to misc.interp_pts to reflect its more
  general interpolation usage.
* The parameter dictionary returned from Whittaker-smoothing-based functions
  no longer includes 'roughness' and 'fidelity' values since the values were not used
  elsewhere.


Version 0.3.0 (2021-04-29)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

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

New Features/Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

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
