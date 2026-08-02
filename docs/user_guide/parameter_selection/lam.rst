===============
Selecting `lam`
===============

Estimation Functions
--------------------

pybaselines currently has no estimation functions for ``lam``. While a similar procedure
as used by :func:`~utils.estimate_polyorder` could work for ``lam`` as well, it only provides
a good estimate for a select few methods; since ``lam`` in general has more robust optimization
methods available for it compared to other parameters, no estimation function was included.

Optimization Methods
---------------------

Discuss :meth:`~Baseline.optimize_extended_range` and :meth:`~Baseline.optimize_pls`.
