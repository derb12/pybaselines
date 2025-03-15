# -*- coding: utf-8 -*-
"""The main entry point for using the object oriented api of pybaselines.

Created on April 8, 2023
@author: Donald Erb

"""

from .morphological import _Morphological
from .optimizers import _Optimizers
from .polynomial import _Polynomial
from .smooth import _Smooth
from .spline import _Spline
from .whittaker import _Whittaker


class Baseline2D(
    _Morphological, _Optimizers, _Polynomial, _Smooth, _Spline, _Whittaker
):
    """
    A class for all 2D baseline correction algorithms.

    Contains all available 2D baseline correction algorithms in pybaselines as methods to
    allow a single interface for easier usage.

    Parameters
    ----------
    x_data : array-like, shape (M,), optional
        The x-values of the measured data (independant variable for the rows). Default is
        None, which will create an array from -1 to 1 during the first function call with
        length equal to the input data length.
    z_data : array-like, shape (N,), optional
        The z-values of the measured data (independant variable for the columns). Default is
        None, which will create an array from -1 to 1 during the first function call with
        length equal to the input data length.
    check_finite : bool, optional
        If True (default), will raise an error if any values in input data are not finite.
        Setting to False will skip the check. Note that errors may occur if
        `check_finite` is False and the input data contains non-finite values.
    assume_sorted : bool, optional
        If False (default), will sort the input `x_data` and `z_data` values. Otherwise,
        the input is assumed to be sorted, although they will still be checked to be in
        ascending order. Note that some methods will raise an error if `x_data` or `z_data`
        values are not unique.
    output_dtype : type or numpy.dtype, optional
        The dtype to cast the output array. Default is None, which uses the typing
        of the input data.

    Attributes
    ----------
    x : numpy.ndarray or None
        The x-values for the object. If initialized with None, then `x` is initialized the
        first function call to have the same size as the input `data.shape[-2]` and has min
        and max values of -1 and 1, respectively.
    x_domain : numpy.ndarray
        The minimum and maximum values of `x`. If `x_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).
    z : numpy.ndarray or None
        The z-values for the object. If initialized with None, then `z` is initialized the
        first function call to have the same size as the input `data.shape[-1]` and has min
        and max values of -1 and 1, respectively.
    z_domain : numpy.ndarray
        The minimum and maximum values of `z`. If `z_data` is None during initialization, then
        set to numpy.ndarray([-1, 1]).

    """

    def _get_method(self, baseline_method):
        """
        A helper function to allow accessing methods by their string.

        Parameters
        ----------
        baseline_method : str
            The name of the desired method as a string. Capitalization is ignored. For
            example, both 'asls' and 'AsLS' would return :meth:`~.Baseline2D.asls`.

        Returns
        -------
        output : Callable
            The callable method corresponding to the input string.

        Raises
        ------
        AttributeError
            Raised if the input method does not exist.

        """
        method_string = baseline_method.lower()
        if hasattr(self, method_string):
            output = getattr(self, method_string)
        else:
            raise AttributeError(f'unknown method "{baseline_method}"')

        return output
