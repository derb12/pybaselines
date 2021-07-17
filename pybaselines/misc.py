# -*- coding: utf-8 -*-
"""Miscellaneous functions for creating baselines.

Created on April 2, 2021
@author: Donald Erb


The function beads and all related functions were adapted from MATLAB code from
https://www.mathworks.com/matlabcentral/fileexchange/49974-beads-baseline-estimation-and-denoising-with-sparsity
(accessed June 28, 2021), which was licensed under the BSD-3-clause below.

Copyright (c) 2018, Laurent Duval
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution
* Neither the name of IFP Energies nouvelles nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


The function _banded_dot_banded was adapted from bandmat (https://github.com/MattShannon/bandmat)
(accessed July 10, 2021), which was licensed under the BSD-3-clause below.

Copyright 2013, 2014, 2015, 2016, 2017, 2018 Matt Shannon

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
   2. Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
   3. The name of the author may not be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.

"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import get_blas_funcs, solve_banded, solveh_banded
from scipy.ndimage import uniform_filter1d
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu, spsolve

from ._compat import _HAS_NUMBA, jit
from .utils import _MIN_FLOAT, relative_difference


def interp_pts(x_data, baseline_points=(), interp_method='linear'):
    """
    Creates a baseline by interpolating through input points.

    Parameters
    ----------
    x_data : array-like, shape (N,)
        The x-values of the measured data.
    baseline_points : array-like, shape (n, 2)
        An array of ((x_1, y_1), (x_2, y_2), ..., (x_n, y_n)) values for
        each point representing the baseline.
    interp_method : str, optional
        The method to use for interpolation. See :class:`scipy.interpolate.interp1d`
        for all options. Default is 'linear', which connects each point with a
        line segment.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline array constructed from interpolating between
        each input baseline point.
    dict
        An empty dictionary, just to match the output of all other algorithms.

    Notes
    -----
    This method is only suggested for use within user-interfaces.

    Regions of the baseline where `x_data` is less than the minimum x-value
    or greater than the maximum x-value in `baseline_points` will be assigned
    values of 0.

    """
    x = np.asarray(x_data)
    points = np.asarray(baseline_points).T

    interpolator = interp1d(
        *points, kind=interp_method, bounds_error=False, fill_value=0
    )
    # TODO why not just use x in the interpolator call?
    baseline = interpolator(np.linspace(np.nanmin(x), np.nanmax(x), x.shape[0]))

    return baseline, {}


def _banded_dot_vector(ab, x, n_lower, n_upper, a_rows, a_columns):
    """
    Computes the dot product of the matrix `a` in banded format (`ab`) with the vector `x`.

    Parameters
    ----------
    ab : array-like, shape (`n_lower` + `n_upper` + 1, N)
        The banded matrix.
    x : array-like, shape (N,)
        The vector.
    n_lower : int
        The number of lower diagonals in `ab`.
    n_upper : int
        The number of upper diagonals in `ab`.
    a_rows : int
        The number of rows in the full `a` matrix.
    a_columns : int
        The number of columns in the full `a` matrix.

    Returns
    -------
    output : numpy.ndarray, shape (N,)
        The dot product of `ab` and `x`.

    Notes
    -----
    BLAS's symmetric version, 'sbmv', shows no significant speed increase, so just
    uses the general 'gbmv' function to simplify the function.

    The function is faster if the input `ab` matrix is Fortran-ordered (has the
    F_CONTIGUOUS numpy flag), since the underlying 'gbmv' BLAS function is
    implemented in Fortran.

    """
    matrix = np.asarray(ab)
    vector = np.asarray(x)

    gbmv = get_blas_funcs(['gbmv'], (matrix, vector))[0]
    # gbmv computes y = alpha * a * x + beta * y where a is the banded matrix
    # (in compressed form), x is the input vector, y is the output vector, and alpha
    # and beta are scalar multipliers
    output = gbmv(
        m=a_rows,  # number of rows of `a` matrix in full form
        n=a_columns,  # number of columns of `a` matrix in full form
        kl=n_lower,  # sub-diagonals
        ku=n_upper,  # super-diagonals
        alpha=1.0,  # alpha, required
        a=matrix,  # `a` matrix in compressed form
        x=vector,  # `x` vector
        # trans=False,  # tranpose a, optional; may allow later
    )

    return output


@jit(nopython=True, cache=True)
def _banded_dot_banded(a, b, a_lower, a_upper, b_lower, b_upper, symmetric_output=False):
    """
    Calculates the matrix multiplication of `a` and `b` in banded forms.

    `a` and `b` must be square matrices in their full form or else this calculation
    may be incorrect.

    Derived from bandmat: https://github.com/MattShannon/bandmat/blob/master/bandmat/tensor.pyx
    function `dot_mm`, licensed under the BSD-3-Clause.

    Parameters
    ----------
    a : [type]
        [description]
    b : [type]
        [description]
    a_lower : [type]
        [description]
    a_upper : [type]
        [description]
    b_lower : [type]
        [description]
    b_upper : [type]
        [description]
    symmetric_output : bool, optional
        Whether the output matrix is known to be symmetric. If True, will only calculate
        the matrix multiplication for the upper bands, and the lower bands will be filled
        in using the upper bands. Default is False.

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        Raised if `a` and `b` do not have the same number of rows.

    """
    # a and b must both be square banded matrices
    a_rows = a.shape[1]
    b_rows = b.shape[1]
    if a_rows != b_rows:
        raise ValueError('a and b must have the same number of rows')
    diag_length = a_rows  # main diagonal length
    output = np.zeros((a_lower + b_lower + a_upper + b_upper + 1, diag_length))
    c_upper = a_upper + b_upper

    if symmetric_output:
        lower_bound = 0  # only fills upper bands
    else:
        lower_bound = a_lower + b_lower
    # TODO could be done in parallel, but does it speed up at all?
    for o_c in range(-(a_upper + b_upper), lower_bound + 1):
        for o_a in range(-min(a_upper, b_lower - o_c), min(a_lower, b_upper + o_c) + 1):
            o_b = o_c - o_a
            row_a = a_upper + o_a
            row_b = b_upper + o_b
            row_c = c_upper + o_c
            d_a = 0
            d_b = -o_b
            d_c = -o_b
            for frame in range(max(0, -o_a, o_b), max(0, diag_length + min(0, -o_a, o_b))):
                output[row_c, frame + d_c] += a[row_a, frame + d_a] * b[row_b, frame + d_b]

    if symmetric_output:
        for row in range(1, a_lower + b_lower + 1):
            offset = a_lower + b_lower + 1 - row
            # TODO should not use negative indices since empty 0 rows can sometimes
            # be added; instead, work down from main diagonal
            output[-row, :-offset] = output[row - 1, offset:]

    return output


def _parabola(data):
    """
    Makes a parabola that fits the input data at the two endpoints.

    Used in the beads calculation so that ``data - _parabola(data)`` is close
    to 0 on the two endpoints, which gives better fits for beads.

    Parameters
    ----------
    data : array-like, shape (N,)
        The data values.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The parabola fitting the two endpoints of the input `data`.

    Notes
    -----
    Does not allow inputting x-values since beads does not work properly with unevenly
    spaced data. Fitting custom x-values makes the parabola fit better when the data is
    unevenly spaced, but the resulting beads fit is no better than when using the
    default, evenly spaced x-values.

    Sets `A` as ``min(data)`` so that
    ``max(data - _parabola(data)) - min(data - _parabola(data))`` is approximately the
    same as ``max(data) - min(data)``.

    """
    y = np.asarray(data)
    x = np.linspace(-1, 1, y.shape[0])
    # use only the endpoints; when trying to use the mean of the last few values, the
    # fit is usually not as good since beads expects the endpoints to be 0; may allow
    # setting mean_width as a parameter later
    A = y.min()
    y1 = y[0] - A
    y2 = y[-1] - A
    # mean_width = 5
    # y1 = y[:mean_width].mean() - A
    # y2 = y[-mean_width:].mean() - A

    # if parabola == p(x) = A + B * x + C * x**2, find coefficients such that
    # p(x[0]==x1) = y[0] - min(y)==y1, p(x[-1]==x2) = y[-1] - min(y)==y2, and p(x_middle==0) = 0:
    # A = min(y)
    # C = (x1 * y2 - x2 * y1) / (x1 * x2**2 - x2 * x1**2)
    # B = (y1 - C) / x1
    # then replace x1 with -1, x2 with 1, and simplify
    C = (y2 + y1) / 2
    B = C - y1

    return A + B * x + C * x**2


def _high_pass_filter(data_size, freq_cutoff=0.005, filter_type=1, full_matrix=False):
    """
    Creates the banded matrices A and B such that B(A^-1) is a high pass filter.

    Parameters
    ----------
    data_size : int
        The number of data points.
    freq_cutoff : float, optional
        The normalized cutoff frequency (0 < `freq_cutoff` < 0.5).
    filter_type : int, optional
        [description]. The order of the high pass filter is ``2 * filter_type``.
        Default is 1.
    full_matrix : bool, optional
        If True, will return the full sparse diagonal matrices of A and B. If False
        (default), will return only the bands of A and B.

    """
    if not 0 < freq_cutoff < 0.5:
        raise ValueError('cutoff frequency must be between 0 and 0.5')
    elif filter_type < 1:
        raise ValueError('filter_type must be at least 1')

    b = np.array([1, -1])
    convolve_array = np.array([-1, 2, -1])
    for _ in range(filter_type - 1):
        b = np.convolve(b, convolve_array)
    b = np.convolve(b, np.array([-1, 1]))

    a = 1
    convolve_array = np.array([1, 2, 1])
    for _ in range(filter_type):
        a = np.convolve(a, convolve_array)

    cos_freq = np.cos(2 * np.pi * freq_cutoff)
    t = ((1 - cos_freq) / max(1 + cos_freq, _MIN_FLOAT))**filter_type

    a_diags = np.repeat([b + a * t], data_size, axis=0).T
    b_diags = np.repeat([b], data_size, axis=0).T
    if full_matrix:
        offsets = np.arange(-filter_type, filter_type + 1)
        A = spdiags(a_diags, offsets, data_size, data_size, 'csr')
        B = spdiags(b_diags, offsets, data_size, data_size, 'csr')
    else:
        # add zeros on edges to create the actual banded structure;
        # creates same structure as diags(a[b]_diags, offsets).todia().data[::-1]
        for i in range(filter_type):
            offset = filter_type - i
            a_diags[i][:offset] = 0
            a_diags[-i - 1][-offset:] = 0
            b_diags[i][:offset] = 0
            b_diags[-i - 1][-offset:] = 0
        A = a_diags
        B = b_diags

    return A, B


def _theta(x, asymmetry=6, eps_0=1e-6):
    abs_x = np.abs(x)
    small_mask = abs_x <= eps_0
    small_x = x[small_mask]

    theta = (
        x[(x > eps_0)].sum() - asymmetry * x[x < -eps_0].sum()
        + (
            ((1 + asymmetry) / (4 * eps_0)) * small_x**2 + ((1 - asymmetry) / 2) * small_x
            + eps_0 * (1 + asymmetry) / 4
        ).sum()
    )
    return abs_x, ~small_mask, theta


def _beads_loss(x, use_v2=True, eps_1=1e-6):
    if use_v2:
        loss = x - eps_1 * np.log(x + eps_1)
    else:
        loss = np.sqrt(x**2 + eps_1)

    return loss


def _beads_weighting(x, use_v2=True, eps_1=1e-6):
    if use_v2:
        weight = 1 / (x + eps_1)
    else:
        weight = 1 / np.sqrt(x**2 + eps_1)

    return weight


def _abs_diff(x, smooth_half_window=0):
    """
    Computes the absolute values of the first and second derivatives of input data.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
        [description]
    smooth_half_window : int, optional
        [description]. Default is 0.

    Returns
    -------
    d1_x : numpy.ndarray, shape (N - 1,)
        The absolute value of the first derivative of `x`.
    d2_x : numpy.ndarray, shape (N - 2,)
        The absolute value of the second derivative of `x`.

    """
    # NOTE: smoothing gives faster convergence and better repeatability between
    # sparse and banded beads implementations; similar as stated by the pybeads author
    d1_x = x[1:] - x[:-1]
    if smooth_half_window > 0:
        smooth_window = 2 * smooth_half_window + 1
        d2_x = np.abs(uniform_filter1d(d1_x[1:] - d1_x[:-1], smooth_window))
        d1_x = np.abs(uniform_filter1d(d1_x, smooth_window, output=d1_x), out=d1_x)
    else:
        d2_x = np.abs(d1_x[1:] - d1_x[:-1])
        d1_x = np.abs(d1_x, out=d1_x)

    return d1_x, d2_x


def _sparse_beads(y, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6,
                  filter_type=1, use_v2_loss=True, max_iter=50, tol=1e-2, eps_0=1e-6,
                  eps_1=1e-6):
    """
    [summary]

    Parameters
    ----------
    y : [type]
        [description]
    freq_cutoff : [type]
        [description]
    lam_0 : [type]
        [description]
    lam_1 : [type]
        [description]
    lam_2 : [type]
        [description]
    asymmetry : int, optional
        [description]. Default is 6.
    filter_type : int, optional
        [description]. Default is 1.
    use_v2_loss : bool, optional
        [description]. Default is True.
    max_iter : int, optional
        [description]. Default is 50.
    tol : [type], optional
        [description]. Default is 1e-2.
    eps_0 : [type], optional
        [description]. Default is 1e-6.
    eps_1 : [type], optional
        [description]. Default is 1e-6.

    Returns
    -------
    [type]
        [description]

    """
    num_y = y.shape[0]
    d1_diags = np.zeros((5, num_y))
    d2_diags = np.zeros((5, num_y))
    offsets = np.arange(2, -3, -1)
    A, B = _high_pass_filter(num_y, freq_cutoff, filter_type, True)
    # factorize A since A is unchanged in the function and its factorization
    # is used repeatedly; much faster than calling spsolve each time
    A_factor = splu(A.tocsc(), permc_spec='NATURAL')
    BTB = B * B

    x = y
    d1_x, d2_x = _abs_diff(x)
    # line 2 of Table 3 in beads paper
    d = BTB.dot(A_factor.solve(y)) - A.dot(np.full(num_y, lam_0 * (1 - asymmetry) / 2))
    gamma = np.empty(num_y)
    gamma_factor = lam_0 * (1 + asymmetry) / 2  # 2 * lam_0 * (1 + asymmetry) / 4
    cost_old = 0
    abs_x = np.abs(x)
    big_x = abs_x > eps_0
    for i in range(max_iter):
        # calculate line 6 of Table 3 in beads paper using banded matrices rather
        # than sparse matrices since it is much faster; Gamma + D.T * Lambda * D

        # row 1 and 3 instead of 0 and 2 to account for zeros on top and bottom
        d1_diags[1][1:] = d1_diags[3][:-1] = -_beads_weighting(d1_x, use_v2_loss, eps_1)
        d1_diags[2] = -(d1_diags[1] + d1_diags[3])

        d2_diags[0][2:] = d2_diags[-1][:-2] = _beads_weighting(d2_x, use_v2_loss, eps_1)
        d2_diags[1] = 2 * (d2_diags[0] - np.roll(d2_diags[0], -1, 0)) - 4 * d2_diags[0]
        d2_diags[-2][:-1] = d2_diags[1][1:]
        d2_diags[2] = -(d2_diags[0] + d2_diags[1] + d2_diags[-1] + d2_diags[-2])

        d_diags = lam_1 * d1_diags + lam_2 * d2_diags
        gamma[~big_x] = gamma_factor / eps_0
        gamma[big_x] = gamma_factor / abs_x[big_x]
        d_diags[2] += gamma

        x = A.dot(
            spsolve(
                BTB + A.dot(spdiags(d_diags, offsets, num_y, num_y, 'csr').dot(A)),
                d, 'NATURAL'
            )
        )
        h = B.dot(A_factor.solve(y - x))
        d1_x, d2_x = _abs_diff(x)
        abs_x, big_x, theta = _theta(x, asymmetry, eps_0)
        cost = (
            0.5 * h.dot(h)
            + lam_0 * theta
            + lam_1 * _beads_loss(d1_x, use_v2_loss, eps_1).sum()
            + lam_2 * _beads_loss(d2_x, use_v2_loss, eps_1).sum()
        )
        cost_difference = relative_difference(cost_old, cost)
        if cost_difference < tol:
            break
        cost_old = cost

    diff = y - x
    baseline = diff - B.dot(A_factor.solve(diff))

    return baseline, {'signal': x, 'iterations': i, 'last_tol': cost_difference}


def _banded_beads(y, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6,
                  filter_type=1, use_v2_loss=True, max_iter=50, tol=1e-2, eps_0=1e-6,
                  eps_1=1e-6):
    """
    [summary]

    Parameters
    ----------
    y : [type]
        [description]
    freq_cutoff : [type]
        [description]
    lam_0 : [type]
        [description]
    lam_1 : [type]
        [description]
    lam_2 : [type]
        [description]
    asymmetry : int, optional
        [description]. Default is 6.
    filter_type : int, optional
        [description]. Default is 1.
    use_v2_loss : bool, optional
        [description]. Default is True.
    max_iter : int, optional
        [description]. Default is 50.
    tol : [type], optional
        [description]. Default is 1e-2.
    eps_0 : [type], optional
        [description]. Default is 1e-6.
    eps_1 : [type], optional
        [description]. Default is 1e-6.

    Returns
    -------
    [type]
        [description]

    Notes
    -----
    This function is ~75% faster than _sparse_beads (independent of data size) if Numba is
    installed due to the faster banded solvers. If Numba is not installed, the calculation
    of the dot product of banded matrices makes this calculation significantly slower than
    the sparse implementation.

    It is no faster to pre-compute the Cholesky factorization of A_lower and use
    that with scipy.linalg.cho_solve_banded compared to using A_lower in solveh_banded.

    """
    num_y = y.shape[0]
    d1_diags = np.zeros((5, num_y))
    d2_diags = np.zeros((5, num_y))
    A, B = _high_pass_filter(num_y, freq_cutoff, filter_type, False)
    # NOTE: just use A rather than A.T since A is symmetric s.t. A == A.T
    A_lower = A[filter_type:]

    # B.T == B since it is symmetric
    BTB = _banded_dot_banded(B, B, filter_type, filter_type, filter_type, filter_type, True)
    # number of lower and upper diagonals of A.T * (D.T * D) * A
    num_diags = (2 * filter_type + 2, 2 * filter_type + 2)

    # line 2 of Table 3 in beads paper
    d = (
        _banded_dot_vector(
            np.asfortranarray(BTB),
            solveh_banded(A_lower, y, check_finite=False, lower=True),
            2 * filter_type, 2 * filter_type, num_y, num_y
        )
        - _banded_dot_vector(
            A, np.full(num_y, lam_0 * (1 - asymmetry) / 2), filter_type, filter_type,
            num_y, num_y
        )
    )
    gamma = np.empty(num_y)
    gamma_factor = lam_0 * (1 + asymmetry) / 2  # 2 * lam_0 * (1 + asymmetry) / 4
    x = y
    d1_x, d2_x = _abs_diff(x)
    cost_old = 0
    abs_x = np.abs(x)
    big_x = abs_x > eps_0
    for i in range(max_iter):
        # calculate line 6 of Table 3 in beads paper using banded matrices rather
        # than sparse matrices since it is much faster; Gamma + D.T * Lambda * D

        # row 1 and 3 instead of 0 and 2 to account for zeros on top and bottom
        d1_diags[1][1:] = d1_diags[3][:-1] = -_beads_weighting(d1_x, use_v2_loss, eps_1)
        d1_diags[2] = -(d1_diags[1] + d1_diags[3])

        d2_diags[0][2:] = d2_diags[-1][:-2] = _beads_weighting(d2_x, use_v2_loss, eps_1)
        d2_diags[1] = 2 * (d2_diags[0] - np.roll(d2_diags[0], -1, 0)) - 4 * d2_diags[0]
        d2_diags[-2][:-1] = d2_diags[1][1:]
        d2_diags[2] = -(d2_diags[0] + d2_diags[1] + d2_diags[-1] + d2_diags[-2])

        d_diags = lam_1 * d1_diags + lam_2 * d2_diags

        gamma[~big_x] = gamma_factor / eps_0
        gamma[big_x] = gamma_factor / abs_x[big_x]
        d_diags[2] += gamma

        temp = _banded_dot_banded(
            _banded_dot_banded(A, d_diags, filter_type, filter_type, 2, 2),
            A, filter_type + 2, filter_type + 2, filter_type, filter_type, True
        )
        temp[2:-2] += BTB

        # cannot use solveh_banded since temp is not guaranteed to be positive-definite
        # and diagonally-dominant
        x = _banded_dot_vector(
            A,
            solve_banded(num_diags, temp, d, overwrite_ab=True, check_finite=False),
            filter_type, filter_type, num_y, num_y
        )

        abs_x, big_x, theta = _theta(x, asymmetry, eps_0)
        d1_x, d2_x = _abs_diff(x)
        h = _banded_dot_vector(
            B,
            solveh_banded(A_lower, y - x, check_finite=False, overwrite_b=True, lower=True),
            filter_type, filter_type, num_y, num_y
        )
        cost = (
            0.5 * h.dot(h)
            + lam_0 * theta
            + lam_1 * _beads_loss(d1_x, use_v2_loss, eps_1).sum()
            + lam_2 * _beads_loss(d2_x, use_v2_loss, eps_1).sum()
        )
        cost_difference = relative_difference(cost_old, cost)
        if cost_difference < tol:
            break
        cost_old = cost

    diff = y - x
    baseline = (
        diff
        - _banded_dot_vector(
            B,
            solveh_banded(A_lower, diff, check_finite=False, overwrite_ab=True, lower=True),
            filter_type, filter_type, num_y, num_y
        )
    )

    return baseline, {'signal': x, 'iterations': i, 'last_tol': cost_difference}


def beads(data, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6,
          filter_type=1, cost_function=2, max_iter=50, tol=1e-2, eps_0=1e-6,
          eps_1=1e-6, fit_parabola=True):
    """
    Baseline estimation and denoising with sparsity (BEADS).

    Parameters
    ----------
    data : [type]
        [description]
    freq_cutoff : [type]
        [description]
    lam_0 : [type]
        [description]
    lam_1 : [type]
        [description]
    lam_2 : [type]
        [description]
    asymmetry : int, optional
        [description]. Default is 6.
    filter_type : int, optional
        [description]. Default is 1.
    cost_function : {2, 1, "l1_v1", "l1_v2"}, optional
        [description]. Default is 2.
    max_iter : int, optional
        [description]. Default is 50.
    tol : [type], optional
        [description]. Default is 1e-2.
    eps_0 : [type], optional
        [description]. Default is 1e-6.
    eps_1 : [type], optional
        [description]. Default is 1e-6.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it before
        performing the beads fit as suggested in []_. This ensures the endpoints of
        the fit data are close to 0, which is required by beads. If the data is already
        close to 0 on both endpoints, set `fit_parabola` to False.

    Returns
    -------
    [type]
        [description]

    Notes
    -----
    The default `lam_0`, `lam_1`, and `lam_2` values are good starting points for a
    dataset with 1000 points. Typically, smaller values are needed for larger datasets
    and larger values for smaller datasets.

    References
    ----------
    [1]_

    [2]_

    """
    if isinstance(cost_function, str):  # to maintain parity with MATLAB version
        cost_function = cost_function.lower()
    use_v2_loss = {'l1_v1': False, 'l1_v2': True, 1: False, 2: True}[cost_function]
    y0 = np.asarray_chkfinite(data)
    if fit_parabola:
        parabola = _parabola(y0)
        y = y0 - parabola
    else:
        y = y0

    if _HAS_NUMBA:
        baseline, params = _banded_beads(
            y, freq_cutoff, lam_0, lam_1, lam_2, asymmetry, filter_type, use_v2_loss,
            max_iter, tol, eps_0, eps_1
        )
    else:
        baseline, params = _sparse_beads(
            y, freq_cutoff, lam_0, lam_1, lam_2, asymmetry, filter_type, use_v2_loss,
            max_iter, tol, eps_0, eps_1
        )

    if fit_parabola:
        baseline = baseline + parabola

    return baseline, params
