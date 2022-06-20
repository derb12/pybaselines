# -*- coding: utf-8 -*-
"""The main entry point for using the object oriented api of pybaselines."""

from .classification import _Classification
from .misc import _Misc
from .morphological import _Morphological
from .optimizers import _Optimizers
from .polynomial import _Polynomial
from .smooth import _Smooth
from .spline import _Spline
from .whittaker import _Whittaker


class Baseline(
    _Classification, _Misc, _Morphological, _Optimizers, _Polynomial, _Smooth, _Spline, _Whittaker
):
    """
    A class for all baseline correction algorithms.

    Attributes
    ----------
    poly_order : int
        The last polynomial order used for a polynomial algorithm. Initially is -1, denoting
        that no polynomial fitting has been performed.
    pspline : pybaselines._spline_utils.PSpline or None
        The PSpline object for setting up and solving penalized spline algorithms. Is None
        if no penalized spline setup has been performed.
    vandermonde : numpy.ndarray or None
        The Vandermonde matrix for solving polynomial equations. Is None if no polynomial
        setup has been performed.
    whittaker_system : pybaselines._banded_utils.PenalizedSystem or None
        The PenalizedSystem object for setting up and solving Whittaker-smoothing-based
        algorithms. Is None if no Whittaker setup has been performed.
    x : numpy.ndarray
        The x-values for the object. If initialized with `x_data` as None, then `x` is
        initialized the first function call to have the same length as the input `data`
        and has min and max values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x` is None during initialization, then
        is set to ``numpy.ndarray([-1, 1])``.

    """
