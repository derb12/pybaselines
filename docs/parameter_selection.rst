===================
Parameter Selection
===================

Most baseline algorithms in pybaselines have several parameters that can be adjusted.
While this allows for fine-tuning each algorithm to work in a wide array of cases,
it can also present a difficulty for new users. It is suggested to start by adjusting only
one or two main parameters, and then change other parameters as needed. **Due to the
variable nature of baselines, it is highly recommended to not assume the default
parameters will work for your data!** Below are the suggested parameters to begin
adjusting for each family of algorithms within pybaselines:

* Polynomial methods

    * ``poly_order`` controls the curvature of the baseline.

* Whittaker-smoothing-based methods

    * ``lam`` controls the curvature of the baseline. See
      :ref:`this example <sphx_glr_examples_whittaker_plot_lam_effects.py>`
      to get an idea of how ``lam`` effects the baseline. The optimal ``lam``
      value for each algorithm is not typically the same.

* Morphological methods

    * ``half_window`` controls the general fit of the baseline. See
      :ref:`this example <sphx_glr_examples_morphological_plot_half_window_effects.py>`
      to get an idea of how ``half_window`` effects the baseline. The optimal
      ``half_window`` value for each algorithm is not typically the same.

* Spline methods

    * ``lam`` controls the curvature of the baseline. The
      :ref:`Whittaker example <sphx_glr_examples_whittaker_plot_lam_effects.py>`
      also generally applies to spline methods.

* Smoothing-based methods

    * ``half_window`` controls the general fit of the baseline. The
      :ref:`Morphological example <sphx_glr_examples_morphological_plot_half_window_effects.py>`
      also generally applies to smoothing methods.

* Baseline/Peak Classification methods

    * Algorithm dependent

* Optimizers

    * Algorithm dependent

* Miscellaneous methods

    * Algorithm dependent
