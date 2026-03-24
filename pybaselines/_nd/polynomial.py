# -*- coding: utf-8 -*-
"""Polynomial techniques for fitting baselines to experimental data.

Created on March 11, 2026
@author: Donald Erb


The function penalized_poly and associated helper functions were adapted from MATLAB code from
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

import numpy as np

from .. import _weighting
from ..utils import _MIN_FLOAT, relative_difference, _convert_coef, _convert_coef2d


def modpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
            use_original=False, mask_initial_peaks=False, return_coef=False, max_cross=None):
    y, weight_array, pseudo_inverse = self._setup_polynomial(
        data, weights, poly_order, calc_vander=True, calc_pinv=True, copy_weights=True,
        max_cross=max_cross
    )

    sqrt_w = np.sqrt(weight_array)
    if use_original:
        y0 = y

    coef = pseudo_inverse @ (sqrt_w * y)
    baseline = self._polynomial.vandermonde @ coef
    if mask_initial_peaks:
        # use baseline + deviation since without deviation, half of y should be above baseline
        weight_array[baseline + np.std(y - baseline) < y] = 0
        sqrt_w = np.sqrt(weight_array)
        pseudo_inverse = np.linalg.pinv(sqrt_w[:, None] * self._polynomial.vandermonde)

    tol_history = np.empty(max_iter)
    for i in range(max_iter):
        baseline_old = baseline
        y = np.minimum(y0 if use_original else y, baseline)
        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self._polynomial.vandermonde @ coef
        calc_difference = relative_difference(baseline_old, baseline)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
    if return_coef:
        if hasattr(self, 'z'):
            params['coef'] = _convert_coef2d(
                coef, *self._polynomial.poly_order, self.x_domain, self.z_domain
            )
        else:
            params['coef'] = _convert_coef(coef, self.x_domain)

    return baseline, params


def imodpoly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
             use_original=False, mask_initial_peaks=True, return_coef=False,
             num_std=1., max_cross=None):
    if num_std < 0:
        raise ValueError('num_std must be greater than or equal to 0')

    y, weight_array, pseudo_inverse = self._setup_polynomial(
        data, weights, poly_order, calc_vander=True, calc_pinv=True,
        copy_weights=True, max_cross=max_cross
    )
    sqrt_w = np.sqrt(weight_array)
    if use_original:
        y0 = y

    coef = pseudo_inverse @ (sqrt_w * y)
    baseline = self._polynomial.vandermonde @ coef
    deviation = np.std(sqrt_w * (y - baseline))
    if mask_initial_peaks:
        weight_array[baseline + deviation < y] = 0
        sqrt_w = np.sqrt(weight_array)
        pseudo_inverse = np.linalg.pinv(sqrt_w[:, None] * self._polynomial.vandermonde)

    tol_history = np.empty(max_iter)
    for i in range(max_iter):
        y = np.minimum(y0 if use_original else y, baseline + num_std * deviation)
        coef = pseudo_inverse @ (sqrt_w * y)
        baseline = self._polynomial.vandermonde @ coef
        new_deviation = np.std(sqrt_w * (y - baseline))
        # use new_deviation as dividing term in relative difference
        calc_difference = relative_difference(new_deviation, deviation)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        deviation = new_deviation

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
    if return_coef:
        if hasattr(self, 'z'):
            params['coef'] = _convert_coef2d(
                coef, *self._polynomial.poly_order, self.x_domain, self.z_domain
            )
        else:
            params['coef'] = _convert_coef(coef, self.x_domain)

    return baseline, params


# adapted from (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction);
# see license above
def _huber_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Huber non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is::

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the huber loss function, phi(x).

    References
    ----------
    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for huber is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
        weights = (
            mask * residual * (2 * alpha - 1)
            + (~mask) * 2 * alpha * threshold * np.sign(residual)
        )
    else:
        mask = (residual < threshold)
        weights = (
            mask * residual * (2 * alpha - 1)
            + (~mask) * (2 * alpha * threshold - residual)
        )
    return weights


# adapted from (https://www.mathworks.com/matlabcentral/fileexchange/27429-background-correction);
# see license above
def _truncated_quadratic_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Truncated-Quadratic non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the truncated quadratic function, phi(x).

    References
    ----------
    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for truncated quadratic is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
    else:
        mask = (residual < threshold)
    return mask * residual * (2 * alpha - 1) - (~mask) * residual


def _indec_loss(residual, threshold=1.0, alpha_factor=0.99, symmetric=True):
    """
    The Indec non-quadratic cost function.

    Parameters
    ----------
    residual : numpy.ndarray, shape (N,)
        The residual array.
    threshold : float, optional
        Any residual values below the threshold are given quadratic loss.
        Default is 1.0.
    alpha_factor : float, optional
        The scale between 0 and 1 to multiply the cost function's alpha_max
        value (see Notes below). Default is 0.99.
    symmetric : bool, optional
        If True (default), the cost function is symmetric and applies the same
        weighting for positive and negative values. If False, will apply weights
        asymmetrically so that only positive weights are given the non-quadratic
        weigting and negative weights have normal, quadratic weighting.

    Returns
    -------
    weights : numpy.ndarray, shape (N,)
        The weight array.

    Notes
    -----
    The returned result is

        -residual + alpha_factor * alpha_max * phi'(residual)

    where phi'(x) is the derivative of the Indec function, phi(x).

    References
    ----------
    Liu, J., et al. Goldindec: A Novel Algorithm for Raman Spectrum Baseline
    Correction. Applied Spectroscopy, 2015, 69(7), 834-842.

    Mazet, V., et al. Background removal from spectra by designing and
    minimising a non-quadratic cost function. Chemometrics and Intelligent
    Laboratory Systems, 2005, 76(2), 121-133.

    """
    alpha = alpha_factor * 0.5  # alpha_max for indec is 0.5
    if symmetric:
        mask = (np.abs(residual) < threshold)
        multiple = np.sign(residual)
    else:
        mask = (residual < threshold)
        # multiple=1 is same as sign(residual) since residual is always > 0
        # for asymmetric case, but this allows not doing the sign calculation
        multiple = 1
    weights = (
        mask * residual * (2 * alpha - 1)
        - (~mask) * (
            residual + alpha * multiple * threshold**3 / np.maximum(2 * residual**2, _MIN_FLOAT)
        )
    )
    return weights


def _identify_loss_method(loss_method):
    """
    Identifies the symmetry for the given loss method.

    Parameters
    ----------
    loss_method : str
        The loss method to use. Should have the symmetry identifier as
        the prefix.

    Returns
    -------
    symmetric : bool
        True if `loss_method` had 's_' or 'symmetric_' as the prefix, else False.
    str
        The input `loss_method` value without the first section that indicated
        the symmetry.

    Raises
    ------
    ValueError
        Raised if the loss method does not have the correct form.

    """
    prefix, *split_method = loss_method.lower().split('_')
    if prefix not in ('a', 's', 'asymmetric', 'symmetric') or not split_method:
        raise ValueError('must specify loss function symmetry by prepending "a_" or "s_"')
    if prefix in ('a', 'asymmetric'):
        symmetric = False
    else:
        symmetric = True
    return symmetric, '_'.join(split_method)


def penalized_poly(self, data, poly_order=2, tol=1e-3, max_iter=250, weights=None,
                   cost_function='asymmetric_truncated_quadratic', threshold=None,
                   alpha_factor=0.99, return_coef=False, max_cross=None):
    if not 0 < alpha_factor <= 1:
        raise ValueError('alpha_factor must be between 0 and 1')
    symmetric_loss, method = _identify_loss_method(cost_function)
    loss_function = {
        'huber': _huber_loss,
        'truncated_quadratic': _truncated_quadratic_loss,
        'indec': _indec_loss
    }[method]

    y, weight_array, pseudo_inverse = self._setup_polynomial(
        data, weights, poly_order, calc_vander=True, calc_pinv=True, max_cross=max_cross
    )
    if threshold is None:
        threshold = np.std(y) / 10
    loss_kwargs = {
        'threshold': threshold, 'alpha_factor': alpha_factor, 'symmetric': symmetric_loss
    }

    sqrt_w = np.sqrt(weight_array)
    y = sqrt_w * y

    coef = pseudo_inverse @ y
    baseline = self._polynomial.vandermonde @ coef
    tol_history = np.empty(max_iter)
    for i in range(max_iter):
        baseline_old = baseline
        coef = pseudo_inverse @ (y + loss_function(y - sqrt_w * baseline, **loss_kwargs))
        baseline = self._polynomial.vandermonde @ coef
        calc_difference = relative_difference(baseline_old, baseline)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break

    params = {'weights': weight_array, 'tol_history': tol_history[:i + 1]}
    if return_coef:
        if hasattr(self, 'z'):
            params['coef'] = _convert_coef2d(
                coef, *self._polynomial.poly_order, self.x_domain, self.z_domain
            )
        else:
            params['coef'] = _convert_coef(coef, self.x_domain)

    return baseline, params


def quant_reg(self, data, poly_order=2, quantile=0.05, tol=1e-6, max_iter=250,
              weights=None, eps=None, return_coef=False, max_cross=None):
    # TODO provide a way to estimate best poly_order based on AIC like in Komsta? could be
    # useful for all polynomial methods; maybe could be an optimizer function
    if not 0 < quantile < 1:
        raise ValueError('quantile must be between 0 and 1.')

    y, weight_array = self._setup_polynomial(
        data, weights, poly_order, calc_vander=True, max_cross=max_cross
    )
    sqrt_w = np.sqrt(weight_array)
    baseline_old = y
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        coef = np.linalg.lstsq(
            self._polynomial.vandermonde * sqrt_w[:, None], y * sqrt_w, None
        )[0]
        baseline = self._polynomial.vandermonde @ coef
        # relative_difference(baseline_old, baseline, 1) gives nearly same result and
        # the l2 norm is faster to calculate, so use that instead of l1 norm
        calc_difference = relative_difference(baseline_old, baseline)
        tol_history[i] = calc_difference
        if calc_difference < tol:
            break
        sqrt_w = np.sqrt(_weighting._quantile(y, baseline, quantile, eps))
        baseline_old = baseline

    params = {'weights': sqrt_w**2, 'tol_history': tol_history[:i + 1]}
    if return_coef:
        if hasattr(self, 'z'):
            params['coef'] = _convert_coef2d(
                coef, *self._polynomial.poly_order, self.x_domain, self.z_domain
            )
        else:
            params['coef'] = _convert_coef(coef, self.x_domain)

    return baseline, params
