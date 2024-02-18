# -*- coding: utf-8 -*-
"""
=============================================
Baseline Correction for Two Dimensional Data.
=============================================

:mod:`pybaselines.two_d` provides the following algorithms for baseline correcting 2D data.

* Polynomial methods (:mod:`pybaselines.two_d.polynomial`)

    * poly (Regular Polynomial)
    * modpoly (Modified Polynomial)
    * imodpoly (Improved Modified Polynomial)
    * penalized_poly (Penalized Polynomial)
    * quant_reg (Quantile Regression)

* Whittaker-smoothing-based methods (:mod:`pybaselines.two_d.whittaker`)

    * asls (Asymmetric Least Squares)
    * iasls (Improved Asymmetric Least Squares)
    * airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
    * arpls (Asymmetrically Reweighted Penalized Least Squares)
    * drpls (Doubly Reweighted Penalized Least Squares)
    * iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
    * aspls (Adaptive Smoothness Penalized Least Squares)
    * psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)

* Morphological methods (:mod:`pybaselines.two_d.morphological`)

    * mor (Morphological)
    * imor (Improved Morphological)
    * rolling_ball (Rolling Ball Baseline)
    * tophat (Top-hat Transformation)

* Spline methods (:mod:`pybaselines.two_d.spline`)

    * mixture_model (Mixture Model)
    * irsqr (Iterative Reweighted Spline Quantile Regression)
    * pspline_asls (Penalized Spline Version of asls)
    * pspline_iasls (Penalized Spline Version of iasls)
    * pspline_airpls (Penalized Spline Version of airpls)
    * pspline_arpls (Penalized Spline Version of arpls)
    * pspline_iarpls (Penalized Spline Version of iarpls)
    * pspline_psalsa (Penalized Spline Version of psalsa)

* Smoothing-based methods (:mod:`pybaselines.two_d.smooth`)

    * noise_median (Noise Median method)

* Optimizers (:mod:`pybaselines.two_d.optimizers`)

    * collab_pls (Collaborative Penalized Least Squares)
    * adaptive_minmax (Adaptive MinMax)
    * individual_axes (1D Baseline Correction Along Individual Axes)


@author: Donald Erb
Created on January 15, 2024

"""

from .api import Baseline2D
