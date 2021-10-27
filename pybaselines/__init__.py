# -*- coding: utf-8 -*-
"""
pybaselines - A library of baseline correction algorithms
==========================================================

pybaselines provides different techniques for fitting baselines to experimental data.

* Polynomial methods (:mod:`pybaselines.polynomial`)

    * poly (Regular Polynomial)
    * modpoly (Modified Polynomial)
    * imodpoly (Improved Modified Polynomial)
    * penalized_poly (Penalized Polynomial)
    * loess (Locally Estimated Scatterplot Smoothing)
    * quant_reg (Quantile Regression)
    * goldindec (Goldindec Method)

* Whittaker-smoothing-based methods (:mod:`pybaselines.whittaker`)

    * asls (Asymmetric Least Squares)
    * iasls (Improved Asymmetric Least Squares)
    * airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
    * arpls (Asymmetrically Reweighted Penalized Least Squares)
    * drpls (Doubly Reweighted Penalized Least Squares)
    * iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    * aspls (Adaptive Smoothness Penalized Least Squares)
    * psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)
    * derpsalsa (Derivative Peak-Screening Asymmetric Least Squares Algorithm)

* Morphological methods (:mod:`pybaselines.morphological`)

    * mpls (Morphological Penalized Least Squares)
    * mor (Morphological)
    * imor (Improved Morphological)
    * mormol (Morphological and Mollified Baseline)
    * amormol (Averaging Morphological and Mollified Baseline)
    * rolling_ball (Rolling Ball Baseline)
    * mwmv (Moving Window Minimum Value)
    * tophat (Top-hat Transformation)
    * mpspline (Morphology-Based Penalized Spline)
    * jbcd (Joint Baseline Correction and Denoising)

* Smoothing-based methods (:mod:`pybaselines.smooth`)

    * noise_median (Noise Median method)
    * snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    * swima (Small-Window Moving Average)
    * ipsa (Iterative Polynomial Smoothing Algorithm)
    * ria (Range Independent Algorithm)

* Spline methods (:mod:`pybaselines.spline`)

    * mixture_model (Mixture Model)
    * irsqr (Iterative Reweighted Spline Quantile Regression)
    * corner_cutting (Corner-Cutting Method)

* Baseline/Peak Classification methods (:mod:`pybaselines.classification`)

    * dietrich (Dietrich's Classification Method)
    * golotvin (Golotvin's Classification Method)
    * std_distribution (Standard Deviation Distribution)
    * fastchrom (FastChrom's Baseline Method)
    * cwt_br (Continuous Wavelet Transform Baseline Recognition)
    * fabc (Fully Automatic Baseline Correction)

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

__version__ = '0.6.0'


# import utils first since it is imported by other modules; likewise, import
# optimizers last since it imports the other modules
from . import (
    utils, classification, misc, morphological, polynomial, spline, whittaker, smooth,
    window, optimizers
)
