# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on April 16, 2023
@author: Donald Erb


The function penalized_poly was adapted from MATLAB code from
https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction
(accessed March 18, 2021), which was licensed under the BSD-2-clause below.

License: 2-clause BSD

Copyright (c) 2012, Vincent Mazet
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

"""

import warnings

import numpy as np

from .._nd import polynomial as polynomial_nd
from ..utils import _convert_coef2d
from ._algorithm_setup import _Algorithm2D


class _Polynomial(_Algorithm2D):
    """A base class for all polynomial algorithms."""

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
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

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def modpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                use_original=False, mask_initial_peaks=False, return_coef=False, max_cross=None):
        """
        The modified polynomial (ModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns. If a single value is given, will use
            that for both rows and columns. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [1]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [2]_.
        mask_initial_peaks : bool, optional
            If True, will mask any data where the initial baseline fit + the standard
            deviation of the residual is less than measured data [3]_. Default is False.
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
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Notes
        -----
        Algorithm originally developed in [2]_ and then slightly modified in [1]_.

        References
        ----------
        .. [1] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [2] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [3] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        return polynomial_nd.modpoly(
            self, data, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights,
            use_original=use_original, mask_initial_peaks=mask_initial_peaks,
            return_coef=return_coef, max_cross=max_cross
        )

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def imodpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                 use_original=False, mask_initial_peaks=True, return_coef=False,
                 num_std=1., max_cross=None):
        """
        The improved modified polynomial (IModPoly) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns. If a single value is given, will use
            that for both rows and columns. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        use_original : bool, optional
            If False (default), will compare the baseline of each iteration with
            the y-values of that iteration [1]_ when choosing minimum values. If True,
            will compare the baseline with the original y-values given by `data` [2]_.
        mask_initial_peaks : bool, optional
            If True (default), will mask any data where the initial baseline fit +
            the standard deviation of the residual is less than measured data [3]_.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the `x_data` and `z_data` values and return them in the params
            dictionary. Default is False, since the conversion takes time.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Default
            is 1. Must be greater or equal to 0.
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
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `num_std` is less than 0.

        Notes
        -----
        Algorithm originally developed in [3]_.

        References
        ----------
        .. [1] Gan, F., et al. Baseline correction by improved iterative polynomial
            fitting with automatic threshold. Chemometrics and Intelligent
            Laboratory Systems, 2006, 82, 59-65.
        .. [2] Lieber, C., et al. Automated method for subtraction of fluorescence
            from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
            1363-1367.
        .. [3] Zhao, J., et al. Automated Autofluorescence Background Subtraction
            Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
            2007, 61(11), 1225-1232.

        """
        return polynomial_nd.imodpoly(
            self, data, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights,
            use_original=use_original, mask_initial_peaks=mask_initial_peaks,
            return_coef=return_coef, num_std=num_std, max_cross=max_cross
        )

    # adapted from
    # https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction;
    # see license above
    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def penalized_poly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                       cost_function='asymmetric_truncated_quadratic', threshold=None,
                       alpha_factor=0.99, return_coef=False, max_cross=None):
        """
        Fits a polynomial baseline using a non-quadratic cost function.

        The non-quadratic cost functions penalize residuals with larger values,
        giving a more robust fit compared to normal least-squares.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns. If a single value is given, will use
            that for both rows and columns. Default is 2.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 250.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        cost_function : str, optional
            The non-quadratic cost function to minimize. Must indicate symmetry of the
            method by prepending 'a' or 'asymmetric' for asymmetric loss, and 's' or
            'symmetric' for symmetric loss. Default is 'asymmetric_truncated_quadratic'.
            Available methods, and their associated reference, are:

            * 'asymmetric_truncated_quadratic'[1]_
            * 'symmetric_truncated_quadratic'[1]_
            * 'asymmetric_huber'[1]_
            * 'symmetric_huber'[1]_
            * 'asymmetric_indec'[2]_
            * 'symmetric_indec'[2]_

        threshold : float, optional
            The threshold value for the loss method, where the function goes from
            quadratic loss (such as used for least squares) to non-quadratic. For
            symmetric loss methods, residual values with absolute value less than
            threshold will have quadratic loss. For asymmetric loss methods, residual
            values less than the threshold will have quadratic loss. Default is None,
            which sets `threshold` to one-tenth of the standard deviation of the input
            data.
        alpha_factor : float, optional
            A value between 0 and 1 that controls the value of the penalty. Default is
            0.99. Typically should not need to change this value.
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
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `alpha_factor` is not between 0 and 1.

        Notes
        -----
        In baseline literature, this procedure is sometimes called "backcor".

        References
        ----------
        .. [1] Mazet, V., et al. Background removal from spectra by designing and
            minimising a non-quadratic cost function. Chemometrics and Intelligent
            Laboratory Systems, 2005, 76(2), 121-133.
        .. [2] Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
            Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

        """
        return polynomial_nd.penalized_poly(
            self, data, poly_order=poly_order, tol=tol, max_iter=max_iter, weights=weights,
            cost_function=cost_function, threshold=threshold, alpha_factor=alpha_factor,
            return_coef=return_coef, max_cross=max_cross
        )

    @_Algorithm2D._register(
        sort_keys=('weights',), reshape_baseline=True, reshape_keys=('weights',)
    )
    def quant_reg(self, data, poly_order=2, quantile=0.05, tol=1e-6, max_iter=250,
                  weights=None, eps=None, return_coef=False, max_cross=None):
        """
        Approximates the baseline of the data using quantile regression.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int], optional
            The polynomial orders for the rows and columns. If a single value is given, will use
            that for both rows and columns. Default is 2.
        quantile : float, optional
            The quantile at which to fit the baseline. Default is 0.05.
        tol : float, optional
            The exit criteria. Default is 1e-6. For extreme quantiles (`quantile` < 0.01
            or `quantile` > 0.99), may need to use a lower value to get a good fit.
        max_iter : int, optional
            The maximum number of iterations. Default is 250. For extreme quantiles
            (`quantile` < 0.01 or `quantile` > 0.99), may need to use a higher value to
            ensure convergence.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        eps : float, optional
            A small value added to the square of the residual to prevent dividing by 0.
            Default is None, which uses the square of the maximum-absolute-value of the
            fit each iteration multiplied by 1e-6.
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
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'coef': numpy.ndarray, shape (``poly_order[0] + 1``, ``poly_order[1] + 1``)
                Only if `return_coef` is True. The array of polynomial parameters
                for the baseline, in increasing order. Can be used to create a
                polynomial using :func:`numpy.polynomial.polynomial.polyval2d`.

        Raises
        ------
        ValueError
            Raised if `quantile` is not between 0 and 1.

        Notes
        -----
        Application of quantile regression for baseline fitting as described in [1]_.

        Performs quantile regression using iteratively reweighted least squares (IRLS)
        as described in [2]_.

        References
        ----------
        .. [1] Komsta, Ł. Comparison of Several Methods of Chromatographic
                Baseline Removal with a New Approach Based on Quantile Regression.
                Chromatographia, 2011, 73, 721-731.
        .. [2] Schnabel, S., et al. Simultaneous estimation of quantile curves using
                quantile sheets. AStA Advances in Statistical Analysis, 2013, 97, 77-87.

        """
        return polynomial_nd.quant_reg(
            self, data, poly_order=poly_order, quantile=quantile, tol=tol, max_iter=max_iter,
            weights=weights, eps=eps, return_coef=return_coef, max_cross=max_cross
        )
