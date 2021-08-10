# -*- coding: utf-8 -*-
"""
pybaselines - A library of baseline correction algorithms.
==========================================================

pybaselines provides different techniques for fitting baselines to experimental data.

* Polynomial methods (:mod:`pybaselines.polynomial`)

    * poly (Regular Polynomial)
    * modpoly (Modified Polynomial)
    * imodpoly (Improved Modified Polynomial)
    * penalized_poly (Penalized Polynomial)
    * loess (Locally Estimated Scatterplot Smoothing)
    * quant_reg (Quantile Regression)

* Whittaker-smoothing-based methods (:mod:`pybaselines.whittaker`)

    * asls (Asymmetric Least Squares)
    * iasls (Improved Asymmetric Least Squares)
    * airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
    * arpls (Asymmetrically Reweighted Penalized Least Squares)
    * drpls (Doubly Reweighted Penalized Least Squares)
    * iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    * aspls (Adaptive Smoothness Penalized Least Squares)
    * psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)

* Morphological methods (:mod:`pybaselines.morphological`)

    * mpls (Morphological Penalized Least Squares)
    * mor (Morphological)
    * imor (Improved Morphological)
    * mormol (Morphological and Mollified Baseline)
    * amormol (Averaging Morphological and Mollified Baseline)
    * rolling_ball (Rolling Ball Baseline)
    * mwmv (Moving Window Minimum Value)
    * tophat (Top-hat Transformation)

* Window-based methods (:mod:`pybaselines.window`)

    * noise_median (Noise Median method)
    * snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    * swima (Small-Window Moving Average)

* Baseline/Peak Classification methods (:mod:`pybaselines.classification`)

    * dietrich (Dietrich's Classification Method)
    * golotvin (Golotvin's Classification Method)
    * std_distribution (Standard Deviation Distribution)

* Optimizers (:mod:`pybaselines.optimizers`)

    * collab_pls (Collaborative Penalized Least Squares)
    * optimize_extended_range
    * adaptive_minmax (Adaptive MinMax)

* Miscellaneous methods (:mod:`pybaselines.misc`)

    * interp_pts (Interpolation between points)
    * beads (Baseline Estimation And Denoising with Sparsity)


@author: Donald Erb
Created on March 5, 2021

"""

__version__ = '0.5.1'


# import utils first since it is imported by other modules; likewise, import
# optimizers last since it imports the other modules
from . import (
    utils, classification, misc, morphological, polynomial, whittaker, window, optimizers
)
