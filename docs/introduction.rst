Introduction
============

pybaselines is a Python library that provides many different algorithms for
performing baseline correction on data from experimental techniques such as
Raman, FTIR, NMR, XRD, XRF, PIXE, MALDI-TOF, LIBS, etc. The aim of the project is
to provide a semi-unified API to allow quickly testing and comparing multiple baseline
correction algorithms to find the best one for a set of data.

pybaselines has 50+ baseline correction algorithms. These include popular algorithms,
such as AsLS, airPLS, ModPoly, and SNIP, as well as many lesser known algorithms. Most
algorithms are adapted directly from literature, preserving their original names whenever
possible, although there are a few that are unique to pybaselines, such as penalized spline
versions of Whittaker-smoothing-based algorithms. The full list of implemented algorithms
can be found in the :doc:`API section <api/index>`.
