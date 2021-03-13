# -*- coding: utf-8 -*-
"""
pybaselines - A collection of baseline algorithms for fitting experimental data
===============================================================================

pybaselines provides different techniques for fitting baselines to experimental data.

Baseline fitting techniques are grouped accordingly (note: when a method
is labelled as 'improved', that is the method's name, not editorialization):

a) Penalized least squares (:mod:`pybaselines.penalized_least_squares`)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)

b) Morphological (:mod:`pybaselines.morphological`)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) iamor ()

c) Polynomial (:mod:`pybaselines.polynomial`)

    1) modpoly (Modified Polynomial)
    2) imodpoly (Improved Modified Polynomial)

d) Window-based (:mod:`pybaselines.window`)

    1) noise_median (Noise Median)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)


@author: Donald Erb
Created on March 5, 2021

"""

__version__ = '0.1.0'


from . import (baselines, morphological, penalized_least_squares, polynomial,
               utils)
