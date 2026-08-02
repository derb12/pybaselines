===================
Parameter Selection
===================

Most baseline algorithms in pybaselines have several parameters that can be adjusted.
While this allows for fine-tuning each algorithm to work in a wide array of cases,
it can also present a difficulty for new users. It is suggested to start by adjusting only
one or two main parameters, and then change other parameters as needed. **Due to the
variable nature of these algorithms, it is highly recommended to not assume the default
parameters will work for your data!** Below are the suggested parameters to begin
adjusting for each family of algorithms within pybaselines:

* Polynomial methods

  * ``poly_order`` controls the curvature of the baseline.

* Whittaker-smoothing methods

  * ``lam`` controls the curvature of the baseline. See
    :ref:`this example <sphx_glr_generated_examples_whittaker_plot_lam_effects.py>`
    to get an idea of how ``lam`` effects the baseline. The optimal ``lam``
    value for each algorithm is not typically the same.

* Morphological methods

  * ``half_window`` controls the general fit of the baseline. See
    :ref:`this example <sphx_glr_generated_examples_morphological_plot_half_window_effects.py>`
    to get an idea of how ``half_window`` effects the baseline. The optimal
    ``half_window`` value for each algorithm is not typically the same.

* Spline methods

  * ``lam`` controls the curvature of the baseline. The
    :ref:`Whittaker example <sphx_glr_generated_examples_whittaker_plot_lam_effects.py>`
    also generally applies to spline methods.

* Smoothing methods

  * ``half_window`` controls the general fit of the baseline. The
    :ref:`Morphological example <sphx_glr_generated_examples_morphological_plot_half_window_effects.py>`
    also generally applies to smoothing methods.

* Baseline/Peak Classification methods

  * Algorithm dependent

* Optimizers

  * Algorithm dependent

* Miscellaneous methods

  * Algorithm dependent


.. note::

  In order to make this parameter selection easier, the Examples section of the documentation
  includes code for creating simple interactive GUIs for varying
  :ref:`poly_order <sphx_glr_generated_examples_interactive_plot_interactive_poly.py>`,
  :ref:`lam <sphx_glr_generated_examples_interactive_plot_interactive_lam.py>`, and
  :ref:`half_window <sphx_glr_generated_examples_interactive_plot_interactive_hw.py>`
  for all relevant baseline correction methods.


pybaselines also provides several functions and methods for helping with parameter selection
for ``poly_order``, ``lam``, and ``half_window``. In general, the naming schemes for
these helpers use the prefix ``estimate_`` (e.g. :func:`~.utils.estimate_polyorder`) for
functions that use some simple criteria to estimate an approximate parameter, and ``optimize_``
(e.g. :meth:`~.Baseline.optimize_extended_range`) for
:doc:`optimizer-type methods <../algorithms/algorithms_1d/optimizers>` that provide method-specific
parameters by calling the underlying method and using some selection criteria.

These helper functions are discussed in more detail in the following sections.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   poly_order
   lam
   half_window
