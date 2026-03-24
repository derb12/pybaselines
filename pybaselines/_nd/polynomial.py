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

