1D Baseline Correction
======================

For performing baseline correction on one-dimensional data, the recommended way is
to use the :class:`~.Baseline` class, which provides all of the various different
algorithms under a single interface.

.. currentmodule:: pybaselines

.. autoclass:: Baseline
   :members: banded_solver, pentapy_solver


Polynomial Algorithms
---------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.poly
   Baseline.modpoly
   Baseline.imodpoly
   Baseline.penalized_poly
   Baseline.loess
   Baseline.quant_reg
   Baseline.goldindec

Whittaker Smoothing Algorithms
------------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.asls
   Baseline.iasls
   Baseline.airpls
   Baseline.arpls
   Baseline.drpls
   Baseline.iarpls
   Baseline.aspls
   Baseline.psalsa
   Baseline.derpsalsa
   Baseline.brpls
   Baseline.lsrpls

Morphological Algorithms
------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.mpls
   Baseline.mor
   Baseline.imor
   Baseline.mormol
   Baseline.amormol
   Baseline.rolling_ball
   Baseline.mwmv
   Baseline.tophat
   Baseline.mpspline
   Baseline.jbcd


Spline Algorithms
-----------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.mixture_model
   Baseline.irsqr
   Baseline.corner_cutting
   Baseline.pspline_asls
   Baseline.pspline_iasls
   Baseline.pspline_airpls
   Baseline.pspline_arpls
   Baseline.pspline_drpls
   Baseline.pspline_iarpls
   Baseline.pspline_aspls
   Baseline.pspline_psalsa
   Baseline.pspline_derpsalsa
   Baseline.pspline_mpls
   Baseline.pspline_brpls
   Baseline.pspline_lsrpls

Smoothing Algorithms
--------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.noise_median
   Baseline.snip
   Baseline.swima
   Baseline.ipsa
   Baseline.ria
   Baseline.peak_filling

Baseline/Peak Classification Algorithms
---------------------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.dietrich
   Baseline.golotvin
   Baseline.std_distribution
   Baseline.fastchrom
   Baseline.cwt_br
   Baseline.fabc
   Baseline.rubberband

Optimizing Algorithms
---------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.collab_pls
   Baseline.optimize_extended_range
   Baseline.adaptive_minmax
   Baseline.custom_bc
   Baseline.optimize_pls

Miscellaneous Algorithms
------------------------

.. autosummary::
   :toctree: ../generated/api/
   :nosignatures:

   Baseline.interp_pts
   Baseline.beads
