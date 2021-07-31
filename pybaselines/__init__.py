# -*- coding: utf-8 -*-
"""
pybaselines - A library of baseline correction algorithms.
==========================================================

pybaselines provides different techniques for fitting baselines to experimental data.

a) Polynomial methods (:mod:`pybaselines.polynomial`)

    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)
    5) loess (Locally Estimated Scatterplot Smoothing)

b) Whittaker-smoothing-based methods (:mod:`pybaselines.whittaker`)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
    4) arpls (Asymmetrically Reweighted Penalized Least Squares)
    5) drpls (Doubly Reweighted Penalized Least Squares)
    6) iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    7) aspls (Adaptive Smoothness Penalized Least Squares)
    8) psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)

c) Morphological methods (:mod:`pybaselines.morphological`)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) mormol (Morphological and Mollified Baseline)
    5) amormol (Averaging Morphological and Mollified Baseline)
    6) rolling_ball (Rolling Ball Baseline)
    7) mwmv (Moving Window Minimum Value)
    8) tophat (Top-hat Transformation)

d) Window-based methods (:mod:`pybaselines.window`)

    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    3) swima (Small-Window Moving Average)

e) Baseline/Peak Classification methods (:mod:`pybaselines.classification`)

    1) dietrich (Dietrich's Classification Method)
    2) golotvin (Golotvin's Classification Method)
    3) std_distribution (Standard Deviation Distribution)

f) Optimizers (:mod:`pybaselines.optimizers`)

    1) collab_pls (Collaborative Penalized Least Squares)
    2) optimize_extended_range
    3) adaptive_minmax (Adaptive MinMax)

g) Miscellaneous methods (:mod:`pybaselines.misc`)

    1) interp_pts (Interpolation between points)
    2) beads (Baseline Estimation And Denoising with Sparsity)


@author: Donald Erb
Created on March 5, 2021

"""

__version__ = '0.4.1'


# import utils first since it is imported by other modules; likewise, import
# optimizers last since it imports the other modules
from . import utils, misc, morphological, polynomial, whittaker, window, optimizers
