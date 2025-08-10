Functional Interface (Legacy)
=============================

1D Baseline Correction
----------------------

The :class:`~.Baseline` class was introduced in pybaselines version 1.0 and is the
recommened way to use all of the different baseline correction algorithms provided by
pybaselines due to not having to import multiple functions from different modules and the
benefit that comes from reusing the same Baseline object for multiples fits, as shown in
:ref:`this example <sphx_glr_generated_examples_general_plot_reuse_baseline.py>`.

With that said, pybaselines will retain its functional interface to maintain compatibility for
older code before the class-based interface was introduced, as well as to provide an easier to use
interface for users who are new to object-oriented programming or programming in general. Use of
this legacy functional interface is discouraged but will not be deprecated in the foreseeable future.

For users who want to migrate from the legacy functional interface to the recommened class-based
interface of pybaselines, simply follow the example below.

.. code-block:: python

   # replace this...
   from pybaselines.polynomial import modpoly
   from pybaselines.whittaker import asls
   fit_1, params_1 = modpoly(data, x_data=x_data)
   fit_2, params_2 = asls(data)

   # ...with this
   from pybaselines import Baseline
   baseline_fitter = Baseline(x_data=x_data)
   fit_1, params_1 = baseline_fitter.modpoly(data)
   fit_2, params_2 = baseline_fitter.asls(data)

Modules
~~~~~~~

.. autosummary::
   :toctree: ../generated/api/functional
   :nosignatures:

   pybaselines.polynomial
   pybaselines.whittaker
   pybaselines.morphological
   pybaselines.spline
   pybaselines.smooth
   pybaselines.classification
   pybaselines.optimizers
   pybaselines.misc

2D Baseline Correction
----------------------

No functional interface is provided for 2D baseline correction since the :class:`~.Baseline2D`
class was implemented after the creation on the :class:`~.Baseline` class when the 1D functional
interface was already considered legacy. There are no current plans to implement a functional
interface for 2D algorithms.
