2D Baseline Correction
======================

.. currentmodule:: pybaselines

.. autoclass:: Baseline2D
   :members: banded_solver, pentapy_solver


Polynomial Algorithms
---------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.poly
   Baseline2D.modpoly
   Baseline2D.imodpoly
   Baseline2D.penalized_poly
   Baseline2D.quant_reg

Whittaker Smoothing Algorithms
------------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.asls
   Baseline2D.iasls
   Baseline2D.airpls
   Baseline2D.arpls
   Baseline2D.drpls
   Baseline2D.iarpls
   Baseline2D.aspls
   Baseline2D.psalsa
   Baseline2D.brpls
   Baseline2D.lsrpls

Morphological Algorithms
------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.mor
   Baseline2D.imor
   Baseline2D.rolling_ball
   Baseline2D.tophat


Spline Algorithms
-----------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.mixture_model
   Baseline2D.irsqr
   Baseline2D.pspline_asls
   Baseline2D.pspline_iasls
   Baseline2D.pspline_airpls
   Baseline2D.pspline_arpls
   Baseline2D.pspline_iarpls
   Baseline2D.pspline_psalsa
   Baseline2D.pspline_brpls
   Baseline2D.pspline_lsrpls

Smoothing Algorithms
--------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.noise_median

Optimizing Algorithms
---------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline2D.collab_pls
   Baseline2D.adaptive_minmax
   Baseline2D.individual_axes
