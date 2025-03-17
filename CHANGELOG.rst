=========
Changelog
=========

Version 1.2.0 (2025-03-17)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

New Features
~~~~~~~~~~~~

* Added the locally symmetric reweighted penalized least squares (``lsrpls``) Whittaker smoothing
  algorithm and its penalized spline version ``pspline_lsrpls``.
* Added the Bayesian reweighted penalized least squares (``brpls``) Whittaker smoothing
  algorithm and its penalized spline version ``pspline_brpls``.
* Added the 4S peak filling (``peak_filling``) algorithm, which truncates the data and then iteratively
  selects the minimum of a directional moving average and the current data point.
* ``Baseline`` and ``Baseline2D`` objects keep the computed pseudo-inverse of the Vandermonde for
  polynomial methods if weights are not given, which speeds up most polynomial methods for repeated
  fits.

Bug Fixes
~~~~~~~~~

* All methods of ``Baseline`` and ``Baseline2D`` are now thread-safe as long as non-data arguments
  are the same for each method call.
* Fixed incorrect indexing in ``rubberband``.
* Removed using ``copy=False`` for numpy.array calls since it raised an error in Numpy versions
  2.0 or later if a copy had to be made.
* Fixed an issue converting sparse matrices to banded matrices when solving penalized splines
  where a column could be omitted if the last diagonal value was zero. Only relevant if Numba
  is not installed and using SciPy versions 1.15 and newer.
* Corrected ``airpls`` weighting; the weighting equation for airpls was misprinted in its journal
  article, so changed to the correct weighting scheme.
* Improved overflow avoidance for ``iarpls``, ``airpls``, and ``drpls``.
* Removed internal parallel processing for ``loess`` since it was problematic for both threaded and
  multiprocessing uses.
* Fixed an issue when flattening 3D arrays with shape (N, M, 1) to (N, M) where the shape would be
  output instead of the flattened array.

Other Changes
~~~~~~~~~~~~~

* Officially list Python 3.13 as supported, as well as the experimental free-threaded
  Python 3.13 build.
* Updated lowest supported Python version to 3.9
* Updated lowest supported dependency versions: NumPy 1.20, SciPy 1.6,
  pentapy 1.1, and Numba 0.53
* Allow inputting ``assymetric_coef`` for ``aspls`` and ``pspline_aspls`` to modify shape of the
  weighting curve.
* Added ``normalize_weights`` for ``airpls`` to normalize weights between 0 and 1, which is set to
  True by default. The new, corrected ``airpls`` weighting makes all negative residuals have weights
  greater than 1, so this option can help to avoid numerical issues or overflow. Set to False to ensure
  matching the literature implementation.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* All optimizer algorithms other than ``Baseline2D.individual_axes`` now return the parameter
  dictionary from the underlying method within the ``method_params`` key in order to avoid
  key overlap between the optimizer's parameters and the method parameters.
* The default ``delta`` for ``loess`` was changed from 0 to ``0.01 * (max(x_data) - min(x_data))``.
* The ``delta`` parameter for ``loess`` is now used on the actual input `x_data` rather
  than the `x_data` after scaling to the domain [-1, 1] to make it easier to use.
* The ``optimal_parameter`` key for the ``optimize_extended_range`` method no longer returns the
  log10 of the optimal value when fitting a non polynomial method. For example, it now returns
  10000 rather than 4 if the optimal ``lam`` value was 10000.
* Deprecated passing ``tol`` and ``max_iter`` to ``mpls`` and ``pspline_mpls`` since the keywords
  were not internally used. The keywords will be removed in version 1.4.
* Deprecated the ``pentapy_solver`` attribute of ``Baseline`` and ``Baseline2D`` in
  favor of the ``banded_solver`` attribute to control the solvers used for banded linear systems.
  The attribute will be removed in version 1.4.
* Deprecated passing additional keyword arguments for padding to multiple methods, and will remove
  the functionality in version 1.4. Keyword arguments for padding should now be grouped into
  the ``pad_kwargs`` parameter instead.
* Deprecated passing additional keyword arguments for estimating the half-window parameter if none was
  given for morphological methods, and will remove the functionality in version 1.4. Keyword arguments
  for estimating the half-window should now be grouped into the ``window_kwargs`` parameter instead.
* Deprecated the ``min_rmse`` key from the parameter dictionary output of ``optimize_extended_range``
  in favor of returning all calculated root mean square values from the fittings through the new ``rmse``
  key. The ``min_rmse`` key will be removed in version 1.4.
* **Pending Deprecation**: The functional interface of pybaselines will be deprecated in version 1.3, and
  will be removed in version 2.0. For example, user code using ``whittaker.arpls(...)`` should
  migrate to ``Baseline.arpls(...)``. The only items that will be kept under the main pybaselines
  namespace will be ``Baseline``, ``Baseline2D``, and ``utils``.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Added new examples to the documentation.
* Added a page describing best practices for fitting multiple datasets with pybaselines.
* Render each method on its own page in the documentation.


Version 1.1.0 (2024-02-18)
--------------------------

This is a minor version with new features, deprecations,
and documentation improvements.

New Features
~~~~~~~~~~~~

* Added two dimensional versions of various baseline correction algorithms,
  with the focus on Whittaker-smoothing-based, spline, and polynomial methods.
  These can be accessed using the new `Baseline2D` class.
* Added the `Baseline2D.individual_axes` method, which allows fitting each row and/or
  column in two dimensional data with any one dimensional method in pybaselines.
* Added a version of the rubberband method to pybaselines.classification which allows fitting
  individual segments within data to better fit concave-shaped data.
* Added the Customized Baseline Correction (custom_bc) method to
  pybaselines.optimizers, which allows fitting baselines with controllable
  levels of stiffness in different regions.
* Added a penalized spline version of mpls (pspline_mpls) to pybaselines.spline.
* Updated spline.mixture_model to use expectation-maximization rather than the previous
  nieve approach of fitting the histogram of the residuals with the probability density
  function. Should reduce calculation times.
* Added a function for penalized spline (P-spline) smoothing to pybaselines.utils,
  `pybaselines.utils.pspline_smooth`, which will return a tuple of the smoothed input and
  the knots, spline coefficients, and spline degree for any further use with
  SciPy's BSpline.

Other Changes
~~~~~~~~~~~~~

* Officially list Python 3.12 as supported.
* Updated lowest supported Python version to 3.8
* Updated lowest supported dependency versions: NumPy 1.20, SciPy 1.5,
  pentapy 1.1, and Numba 0.49
* Use SciPy's sparse arrays when the installed SciPy version is 1.12 or newer. This
  only affects user codes if using functions from the pybaselines.utils module.
* Vendor SciPy's cwt and ricker functions, which were deprecated from SciPy in version 1.12.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Deprecated passing `num_bins` to spline.mixture_model. The keyword argument will
  be removed in version 1.3.
* Removed the pybaselines.config module, which was simply used to set the pentapy solver.
  The same behavior can be done by setting the `pentapy_solver` attribute of a `Baseline`
  object after initialization.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Added a section of the documentation explaining the extension of baseline correction for
  two dimensional data.
* Added new examples for 2D baseline correction and for custom_bc.


Version 1.0.0 (2022-10-26)
--------------------------

This is a major version with new features, bug fixes, deprecations,
and documentation improvements.

New Features
~~~~~~~~~~~~

* Added a new class-based api for all algorithms, which can be accessed by using
  the `pybaselines.api.Baseline` class. All algorithms are available as methods of
  the `Baseline` class. The functional api from earlier versions is also maintained
  for backwards compatibility.
* All functions now allow inputting an `x_data` keyword, even if it is not used within
  the function, to allow for a more consistent api. Likewise, `pybaselines.misc.interp_pts`
  added an unused `data` keyword. Now, all algorithms can be called with
  the signature `baseline_algorithm(data=y_data, x_data=x_data, ...)`.
* Added a function for Whittaker smoothing to pybaselines.utils,
  `pybaselines.utils.whittaker_smooth`.
* whittaker.iasls and spline.psline_iasls now allow inputting a `diff_order` parameter.

Bug Fixes
~~~~~~~~~

* Fixed the addition of the penalty difference diagonals in spline.pspline_drpls, which
  was incorrectly treating the penalty diagonals as lower banded rather than fully banded.

Other Changes
~~~~~~~~~~~~~

* Officially list Python 3.11 as supported.
* Added default `half_window` values for snip and noise_median.
* collab_pls accomodates `alpha` for aspls and pspline_aspls; the `alpha` parameter is
  calculated for the entire dataset in the same way as the weights and is then fixed when
  fitting each of the individual data entries.
* Improved input validation.
* Improved testing base classes to reduce copied code and improve test coverage.
* Improved code handling for banded systems and penalized splines to simplify internal code.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Removed the ability to pass addtional keyword arguments to algorithms in
  pybaselines.optimizers, which was deprecated in version 0.8.0.
* Removed the deprecated pybaselines.window module, which was formally deprecated in version 0.8.
* Moved the `PENTAPY_SOLVER` constant from pybaselines.utils to the new pybaselines.config module.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Added citation guidelines to make it easier to cite pybaselines.
* Added new examples showing how to use the new `Baseline` class.
* Added a new example examining the `beads` algorithm.


Version 0.8.0 (2021-12-07)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

New Features
~~~~~~~~~~~~

* Added more efficient ways for creating the spline basis, and now solve penalized
  spline equations as a banded system rather than as a sparse system. Compared to
  version 0.7.0, spline.mixture_model, spline.irsqr, and morphological.mpspline are
  ~60-90% faster when numba is installed and ~10-70% faster without numba.
* Made several calculations in spline.mixture_model more efficient, further reducing the
  time by ~60-70% compared to the timings above without numba. The total time reduction
  from version 0.7.0 for spline.mixture_model without numba is ~50-90%.
* Added penalized spline versions of all Whittaker-smoothing-based algorithms
  (pspline_asls, pspline_iasls, pspline_airpls, pspline_arpls, pspline_drpls, pspline_iarpls,
  pspline_aspls, pspline_psalsa, and pspline_derpsalsa) to pybaselines.spline.

Bug Fixes
~~~~~~~~~

* Was not multiplying the penalty in whittaker.iasls by `lam_1`.
* The output weights for polynomial.quant_reg and polynomial.loess are now squared
  before returning since the square root of the weights are used internally.
* The output weights and polynomial coefficients (if `return_coef` is True) for
  polynomial.loess are now sorted to match the original order of the input x-values.
* The output weights for optimizers.optimize_extended_range are now truncated and
  sorted before returning to match the original order and length of the input x-values.
* smooth.noise_median now works with a `smooth_half_window` value of 0 to give no smoothing.

Other Changes
~~~~~~~~~~~~~

* Officially list Python 3.10 as supported.
* pybaselines is now available to install using conda from the conda-forge channel.
* Changed a factor in the weighting for whittaker.aspls to better match the
  implementation in literature.
* Allow inputting x-values for all penalized spline functions rather than assuming
  evenly spaced measurements.
* optimizers.adaptive_minmax now allows separate `constrained_fraction` and
  `constrained_weight` values for for the left and right edges.
* The error raised by optimizers.collab_pls if the input data is not 2-dimensional
  is now more explicit.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* No longer allow negative or array-like values for the penalty multipliers in
  Whittaker-smoothing-based functions, penalized spline functions, morphological.jbcd,
  or misc.beads. Array-like penalty values are technically valid; however, they change the
  symmetry of the banded linear system, so additional code will have to be added in a
  later version to ensure the setup is correct before re-allowing array-like values.
* Deprecated passing keyword arguments to all functions in pybaselines.optimizers.
  Passing additional keyword arguments will raise an error starting in version 0.10.0
  or 1.0.0, whichever comes first (the same deprecation for optimize_extended_range made
  in version 0.7.0 is also pushed back to 0.10.0 or 1.0.0).
* For spline algorithms, the min and max x-values are now included as inner knots when
  creating the spline basis rather than counting them as the first outer knots. To match
  the number of knots from previous versions, the `num_knots` parameter should add 2 to
  the `num_knots` used in previous versions.
* Formally deprecated pybaselines.window, which was replaced by pybaselines.smooth in
  version 0.6.0. pybaselines.window will be removed in version 1.0.
* Removed optimize_window from pybaselines.morphological, which was deprecated in
  version 0.6.0
* Removed the code for allowing array-like `half_window` or `smooth_half_window` values
  for morphological.rolling_ball, which was deprecated in version 0.7.0.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Added more examples to the documentation for fitting noisy data and exploring
  penalized spline parameters.
* Added an introduction for the splines category in the algorithms section of the
  documentation.


Version 0.7.0 (2021-10-28)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

Notice: beginning in version 0.8.0, a DeprecationWarning will be emitted
when using any function from the pybaselines.window module. Use the
pybaselines.smooth module instead.

New Features
~~~~~~~~~~~~

* Added the range independent algorithm (ria) to pybaselines.smooth, which extends
  the left and/or right edges, similar to optimizers.optimize_extended_range, and
  iteratively smooths until the area of the extended regions is recovered.
* Added the joint baseline correction and denoising algorithm (jbcd) to
  pybaselines.morphological, which uses regularized least-squares fitting combined
  with morphological operations to simultaneously obtain the baseline and denoised signal.
* Added the iterative polynomial smoothing algorithm (ipsa) to pybaselines.smooth, which
  iteratively smooths the input data using a second-order Savitzky–Golay filter.
* Added the continuous wavelet transform baseline recognition algorithm (cwt_br) to
  pybaselines.classification, which uses a continuous wavelet transform to classify
  the baseline points and iterative polynomial fitting to create the baseline.
* Added the fully automatic baseline correction algorithm (fabc) to
  pybaselines.classification, which is very similar to classification.dietrich, except
  that it uses a continuous wavelet transform to estimate the derivative and fits the
  baseline using Whittaker smoothing.
* Added a `min_length` parameter to most classification algorithms, which allows
  discarding any values in the baseline mask where the number of consecutive points
  designated as baseline is less than `min_length`, making the algorithms more robust.
* The `threshold` for polynomial.fastchrom can now be a Callable to allow the user to
  define their own thresholding functions based on the rolling standard deviation
  distribution.
* Allow optimizers.optimize_extended_range to use spline (mixture_model, irsqr)
  and classification (dietrich, cwt_br, fabc) functions.
* Allow optimizers.collab_pls to use spline functions (mixture_model, irsqr).

Bug Fixes
~~~~~~~~~

* Increased the minimum scipy version to 1.0 in order to use the BLAS function
  gbmv (dot product of a banded matrix and vector) for misc.beads.
* Use stable sorting when sorting the x-values for polynomial.loess and
  optimizers.optimize_extended_range to ensure that the sorting is correct.
* Fixed an issue when specifying `output` with scipy.ndimage.uniform_filter1d in scipy
  versions before version 1.1.0.
* Fixed an issue using `dtype` with numpy.arange in a numba jit wrapped function, which
  was not introduced until numba version 0.47.
* Fixed an indexing error in spline.corner_cutting which would give an erroneous index
  at which the maximum area removal occurred.
* Fixed an issue that occurred when inputting weights into spline.mixture_model.
* If weights are input into optimizers.optimize_extended_range as keyword arguments,
  the weights are now correctly sorted to match the sorting of the x-values and padded
  to account for the added portions on the left and/or right edges before using in the
  fitting function.
* Fixed the output of utils.padded_convolve when the kernel was even shaped (which
  never happens in actual application in pybaselines) or larger than the data.
* Fixed an issue caused by using an `extrapolate_window` of 1 for utils.pad_edges,
  or an `extrapolate_window` of 0 or 1 for utils._get_edges (called by
  optimizers.optimize_extended_range).

Other Changes
~~~~~~~~~~~~~

* Use scipy's expit function for whittaker.arpls and aspls, which does not emit the
  warning for exponential overflow. The warning was not needed since the overflow
  ultimately makes weights of 0 for the two functions.
* Use np.gradient for the computed derivatives in derpsalsa and dietrich, which gives
  slightly less noisy derivatives than the finite difference used by np.diff.
* Only sort x-values if they are given for polynomial.loess and
  optimizers.optimize_extended_range, which saves a little time otherwise.
* Made whittaker.airpls error handling more robust in order to catch errors from the
  solvers as well, which should catch any errors not prevented by checking the residual's
  length.
* Allow the `mode` for utils.pad_edges to be a callable padding function,
  matching numpy.pad's behavior.
* Added `tol_history` to the output parameters of classification.dietrich.
* Switched to using Scipy's convolve over Numpy's. Scipy's convolve can choose between
  the direct convolution, which is always used by Numpy, or an FFT based convolution,
  which is significantly faster for large arrays.
* Added testing for the minimum supported versions of all dependencies to
  the project's continuous integration in order to ensure that the minimum
  stated dependencies actually work.
* Allow specifying two separate extrapolate windows when padding using
  utils.pad_edges to allow better flexibility for fitting the edges.

Deprecations/Breaking Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Deprecated allowing passing additional keyword arguments to optimizers.optimize_extended_range
  since the `pad_kwargs` parameter is used by both the optimize_extended_range function
  and the internal functions it supports. Now, all keyword arguments should be placed in
  the `method_kwargs` dictionary. Passing additional keyword arguments will raise
  an error starting in version 0.9.0.
* Deprecated allowing an array for the `half_window` or `smooth_half_window` parameters in
  morphological.rolling_ball. While the array-based moving min/max functions were valid,
  when combined for the morphological opening, the output would produce invalid results
  where the opening values were greater than the input data, which should not be allowed by
  the actual morphological opening. Using an array `half_window` will raise an error in
  version 0.8.0.

Documentation/Examples
~~~~~~~~~~~~~~~~~~~~~~

* Added several new examples that explore different aspects of pybaselines.
* Use sphinx-gallery to display the example programs' code and outputs within
  the documentation.


Version 0.6.0 (2021-09-09)
--------------------------

This is a minor version with new features, bug fixes, deprecations,
and documentation improvements.

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

Bug Fixes
~~~~~~~~~

* Fixed an issue with utils.pad_edges when `mode` was "extrapolate" and `extrapolate_window`
  was 1.

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
* Added the moving-window minimum value (mwmv) pybaselines.morphological, which estimates the
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
