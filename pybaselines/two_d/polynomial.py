# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on April 16, 2023
@author: Donald Erb

"""

import warnings

import numpy as np

from .._nd.polynomial import _PolynomialNDMixin
from ..utils import _convert_coef2d
from ._algorithm_setup import _Algorithm2D


class _Polynomial(_Algorithm2D, _PolynomialNDMixin):
    """A base class for all polynomial algorithms."""

    @_Algorithm2D._register(sort_keys=('weights',), reshape_keys=('weights',))
    def poly(self, data, poly_order=2, weights=None, return_coef=False, max_cross=None):
        """
        Computes a polynomial fit to the data.

        .. deprecated:: 1.3.0
            ``poly`` is deprecated and will be removed in version 1.5.0.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns. If a single value is given, will use
            that for both rows and columns. Default is 2.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the `x_data` and `z_data` values and return them in the params
            dictionary. Default is False, since the conversion takes time.
        max_cross : int, optional
            The maximum degree for the cross terms. For example, if `max_cross` is 1, then
            ``x * z**2``, ``x**2 * z``, and ``x**2 * z**2`` would all be set to 0. Default is
            None, which does not limit the cross terms.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Notes
        -----
        To only fit regions without peaks, supply a weight array with zero values
        at the indices where peaks are located. It is **NOT** recommended to use this
        method without supplying weights since it is otherwise a least-squares fit to
        the data, which is not a correct representation of the baseline.

        """
        warnings.warn(
            '"poly" is deprecated and will be removed in version 1.5.', DeprecationWarning,
            stacklevel=3
        )

        y, weight_array, pseudo_inverse = self._setup_polynomial(
            data, weights, poly_order, calc_vander=True, calc_pinv=True, max_cross=max_cross
        )
        sqrt_w = np.sqrt(weight_array)

        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self._polynomial.vandermonde @ coef
        params = {'weights': weight_array}
        if return_coef:
            params['coef'] = _convert_coef2d(
                coef, *self._polynomial.poly_order, self.x_domain, self.z_domain
            )

        return baseline, params
