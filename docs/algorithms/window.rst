================
Window Baselines
================

The contents of :mod:`pybaselines.window` contain algorithms that use
moving windows to estimate the baseline.

.. note::
   The window size used for window-based algorithms is index-based, rather
   than based on the units of the data, so proper conversions must be done
   by the user to get the desired window size.


Algorithms
----------

noise_median (Noise Median method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


swima (Small-Window Moving Average)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
