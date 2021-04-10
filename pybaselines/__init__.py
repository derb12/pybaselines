# -*- coding: utf-8 -*-
"""
pybaselines - A collection of algorithms for fitting the baseline of experimental data
======================================================================================

pybaselines provides different techniques for fitting baselines to experimental data.

a) Polynomial (:mod:`pybaselines.polynomial`)

    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)
    5) loess (Locally Estimated Scatterplot Smoothing)

b) Whittaker-smoothing-based techniques (:mod:`pybaselines.whittaker`)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)
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

e) Optimizers (:mod:`pybaselines.optimizers`)

    1) collab_pls (Collaborative Penalized Least Squares)
    2) optimize_extended_range
    3) adaptive_minmax (Adaptive MinMax)

f) Manual methods (:mod:`pybaselines.manual`)

    1) linear_interp (Linear interpolation between points)


@author: Donald Erb
Created on March 5, 2021

"""

__version__ = '0.2.0'


from . import manual, morphological, optimizers, polynomial, utils, whittaker, window
