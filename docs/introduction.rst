Introduction
============

pybaselines provides many different algorithms for fitting baselines to data from
experimental techniques such as Raman, FTIR, NMR, XRD, PIXE, etc. The aim of
the project is to provide a semi-unified API to allow quickly testing and comparing
multiple baseline algorithms to find the best one for a set of data.

pybaselines has 25+ baseline algorithms. Baseline fitting techniques are grouped
accordingly (note: when a method is labelled as 'improved', that is the method's
name, not editorialization):

a) Polynomial (:mod:`pybaselines.polynomial`)

    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)
    5) loess (Locally Estimated Scatterplot Smoothing)

b) Whittaker-smoothing-based techniques (:mod:`pybaselines.whittaker`)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
    4) arpls (Asymmetrically Reweighted Penalized Least Squares)
    5) drpls (Doubly Reweighted Penalized Least Squares)
    6) iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    7) aspls (Adaptive Smoothness Penalized Least Squares)
    8) psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)

c) Morphological (:mod:`pybaselines.morphological`)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) mormol (Morphological and Mollified Baseline)
    5) amormol (Averaging Morphological and Mollified Baseline)
    6) rolling_ball (Rolling Ball Baseline)

d) Window-based (:mod:`pybaselines.window`)

    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    3) swima (Small-Window Moving Average)

e) Optimizers (:mod:`pybaselines.optimizers`)

    1) collab_pls (Collaborative Penalized Least Squares)
    2) optimize_extended_range
    3) adaptive_minmax (Adaptive MinMax)

f) Manual methods (:mod:`pybaselines.manual`)

    1) linear_interp (Linear interpolation between points)
