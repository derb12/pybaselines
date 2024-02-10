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
from scipy.sparse.linalg import splu, spsolve

from ._algorithm_setup import _Algorithm, _class_wrapper
from ._compat import _HAS_NUMBA, dia_object, jit
from ._validation import _check_array, _check_lam
from .utils import _MIN_FLOAT, relative_difference


class _Misc(_Algorithm):
    """A base class for all miscellaneous algorithms."""

    @_Algorithm._register
    def interp_pts(self, data=None, baseline_points=(), interp_method='linear'):
        """
        Creates a baseline by interpolating through input points.

        Parameters
        ----------
        data : array-like, optional
            The y-values. Not used by this function, but input is allowed for consistency
            with other functions.
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

        Raises
        ------
        ValueError
            Raised of `baseline_points` does not contain at least two values, signifying
            one x-y point.

        Notes
        -----
        This method is only suggested for use within user-interfaces.

        Regions of the baseline where `x_data` is less than the minimum x-value
        or greater than the maximum x-value in `baseline_points` will be assigned
        values of 0.

        """
        points = np.atleast_2d(
            _check_array(baseline_points, check_finite=self._check_finite, ensure_1d=False)
        )
        if points.shape[1] != 2:
            raise ValueError(
                'baseline_points must have shape (number of x-y pairs, 2), but '
                f'instead is {points.shape}'
            )
        interpolator = interp1d(
            points[:, 0], points[:, 1], kind=interp_method, bounds_error=False, fill_value=0
        )
        baseline = interpolator(self.x)

        return baseline, {}

    @_Algorithm._register(sort_keys=('signal',))
    def beads(self, data, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6.0,
              filter_type=1, cost_function=2, max_iter=50, tol=1e-2, eps_0=1e-6,
              eps_1=1e-6, fit_parabola=True, smooth_half_window=None):
        r"""
        Baseline estimation and denoising with sparsity (BEADS).

        Decomposes the input data into baseline and pure, noise-free signal by modeling
        the baseline as a low pass filter and by considering the signal and its derivatives
        as sparse [1]_.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        freq_cutoff : float, optional
            The cutoff frequency of the high pass filter, normalized such that
            0 < `freq_cutoff` < 0.5. Default is 0.005.
        lam_0 : float, optional
            The regularization parameter for the signal values. Default is 1.0. Higher
            values give a higher penalty.
        lam_1 : float, optional
            The regularization parameter for the first derivative of the signal. Default
            is 1.0. Higher values give a higher penalty.
        lam_2 : float, optional
            The regularization parameter for the second derivative of the signal. Default
            is 1.0. Higher values give a higher penalty.
        asymmetry : float, optional
            A number greater than 0 that determines the weighting of negative values
            compared to positive values in the cost function. Default is 6.0, which gives
            negative values six times more impact on the cost function that positive values.
            Set to 1 for a symmetric cost function, or a value less than 1 to weigh positive
            values more.
        filter_type : int, optional
            An integer describing the high pass filter type. The order of the high pass
            filter is ``2 * filter_type``. Default is 1 (second order filter).
        cost_function : {2, 1, "l1_v1", "l1_v2"}, optional
            An integer or string indicating which approximation of the l1 (absolute value)
            penalty to use. 1 or "l1_v1" will use :math:`l(x) = \sqrt{x^2 + \text{eps_1}}`
            and 2 (default) or "l1_v2" will use
            :math:`l(x) = |x| - \text{eps_1}\log{(|x| + \text{eps_1})}`.
        max_iter : int, optional
            The maximum number of iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-2.
        eps_0 : float, optional
            The cutoff threshold between absolute loss and quadratic loss. Values in the signal
            with absolute value less than `eps_0` will have quadratic loss. Default is 1e-6.
        eps_1 : float, optional
            A small, positive value used to prevent issues when the first or second order
            derivatives are close to zero. Default is 1e-6.
        fit_parabola : bool, optional
            If True (default), will fit a parabola to the data and subtract it before
            performing the beads fit as suggested in [2]_. This ensures the endpoints of
            the fit data are close to 0, which is required by beads. If the data is already
            close to 0 on both endpoints, set `fit_parabola` to False.
        smooth_half_window : int, optional
            The half-window to use for smoothing the derivatives of the data with a moving
            average and full window size of `2 * smooth_half_window + 1`. Smoothing can
            improve the convergence of the calculation, and make the calculation less sensitive
            to small changes in `lam_1` and `lam_2`, as noted in the pybeads package [3]_.
            Default is None, which will not perform any smoothing.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'signal': numpy.ndarray, shape (N,)
                The pure signal portion of the input `data` without noise or the baseline.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        Notes
        -----
        The default `lam_0`, `lam_1`, and `lam_2` values are good starting points for a
        dataset with 1000 points. Typically, smaller values are needed for larger datasets
        and larger values for smaller datasets.

        When finding the best parameters for fitting, it is usually best to find the optimal
        `freq_cutoff` for the noise in the data before adjusting any other parameters since
        it has the largest effect [2]_.

        Raises
        ------
        ValueError
            Raised if `asymmetry` is less than 0.

        References
        ----------
        .. [1] Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
            (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.
        .. [2] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms
            using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10.
        .. [3] https://github.com/skotaro/pybeads.

        """
        # TODO maybe add the log-transform from Navarro-Huerta to improve fit for data spanning
        # multiple scales, or at least mention in Notes section; also should add the function
        # in Navarro-Huerta that helps choosing the best freq_cutoff for a dataset
        y0 = self._setup_misc(data)
        if isinstance(cost_function, str):  # allow string to maintain parity with MATLAB version
            cost_function = cost_function.lower()
        use_v2_loss = {'l1_v1': False, 'l1_v2': True, 1: False, 2: True}[cost_function]
        if asymmetry <= 0:
            raise ValueError('asymmetry must be greater than 0')

        if fit_parabola:
            parabola = _parabola(y0)
            y = y0 - parabola
        else:
            y = y0
        # ensure that 0 + eps_0[1] > 0 to prevent numerical issues
        eps_0 = max(eps_0, _MIN_FLOAT)
        eps_1 = max(eps_1, _MIN_FLOAT)
        if smooth_half_window is None:
            smooth_half_window = 0

        lam_0 = _check_lam(lam_0, True)
        lam_1 = _check_lam(lam_1, True)
        lam_2 = _check_lam(lam_2, True)
        if _HAS_NUMBA:
            baseline, params = _banded_beads(
                y, freq_cutoff, lam_0, lam_1, lam_2, asymmetry, filter_type, use_v2_loss,
                max_iter, tol, eps_0, eps_1, smooth_half_window
            )
        else:
            baseline, params = _sparse_beads(
                y, freq_cutoff, lam_0, lam_1, lam_2, asymmetry, filter_type, use_v2_loss,
                max_iter, tol, eps_0, eps_1, smooth_half_window
            )

        if fit_parabola:
            baseline = baseline + parabola

        return baseline, params


_misc_wrapper = _class_wrapper(_Misc)


@_misc_wrapper
def interp_pts(x_data, baseline_points=(), interp_method='linear', data=None):
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
    data : array-like, optional
        The y-values. Not used by this function, but input is allowed for consistency
        with other functions.

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


def _banded_dot_vector(ab, x, ab_lu, a_full_shape):
    """
    Computes the dot product of the matrix `a` in banded format (`ab`) with the vector `x`.

    Parameters
    ----------
    ab : array-like, shape (`n_lower` + `n_upper` + 1, N)
        The banded matrix.
    x : array-like, shape (N,)
        The vector.
    ab_lu : Container(int, int)
        The number of lower (`n_lower`) and upper (`n_upper`) diagonals in `ab`.
    a_full_shape : Container(int, int)
        The number of rows and columns in the full `a` matrix.

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
        m=a_full_shape[0],  # number of rows of `a` matrix in full form
        n=a_full_shape[1],  # number of columns of `a` matrix in full form
        kl=ab_lu[0],  # sub-diagonals
        ku=ab_lu[1],  # super-diagonals
        alpha=1.0,  # alpha, required
        a=matrix,  # `a` matrix in compressed form
        x=vector,  # `x` vector
        # trans=False,  # tranpose a, optional; may allow later
    )

    return output


# adapted from bandmat (bandmat/tensor.pyx/dot_mm_plus_equals and dot_mm); see license above
@jit(nopython=True, cache=True)
def _numba_banded_dot_banded(a, b, c, a_lower, a_upper, b_lower, b_upper, c_upper,
                             diag_length, lower_bound):
    """
    Calculates the matrix multiplication, ``C = A @ B``, with `a`, `b`, and `c` in banded forms.

    `a` and `b` must be square matrices in their full form or else this calculation
    may be incorrect.

    Parameters
    ----------
    a : array-like, shape (`a_lu[0]` + `a_lu[1]` + 1, N)
        A banded matrix.
    b : array-like, shape (`b_lu[0]` + `b_lu[1]` + 1, N)
        The second banded matrix.
    c : numpy.ndarray, shape (D, N)
        The preallocated output matrix. Should be zeroed before passing to this function.
        Will be modified inplace.
    a_lower : int
        The number of lower diagonals in `a`.
    a_upper : int
        The number of upper diagonals in `a`.
    b_lower : int
        The number of lower diagonals in `b`.
    b_upper : int
        The number of upper diagonals in `b`.
    c_upper : int
        The number of upper diagonals in `c`.
    diag_length : int
        The length of the diagonal in the full matrix. Equal to `N`.
    lower_bound : int
        The lowest diagonal to compute in `c`. Either 0 if `c` is symmetric and only
        the upper diagonals need computed, or ``a_lower + b_lower`` to compute all bands.

    Returns
    -------
    c : numpy.ndarray
        The matrix multiplication of `a` and `b`. The number of lower diagonals is the
        minimum of `a_lu[0]` + `b_lu[0]` and `a_full_shape[0]` - 1, the number of upper
        diagonals is the minimum of `a_lu[1]` + `b_lu[1]` and `b_full_shape[1]` - 1, and
        the total shape is (lower diagonals + upper diagonals + 1, N).

    Raises
    ------
    ValueError
        Raised if `a` and `b` do not have the same number of rows or if `a_full_shape[1]`
        and `b_full_shape[0]` are not equal.

    Notes
    -----
    Derived from bandmat (https://github.com/MattShannon/bandmat/blob/master/bandmat/tensor.pyx)
    function `dot_mm`, licensed under the BSD-3-Clause.

    """
    # TODO need to revisit this later and use a different implementation than bandmat's
    # so that a and b don't have to be square matrices;
    # see https://github.com/JuliaMatrices/BandedMatrices.jl and
    # https://www.netlib.org/utk/lsi/pcwLSI/text/node153.html for other implementations

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
                c[row_c, frame + d_c] += a[row_a, frame + d_a] * b[row_b, frame + d_b]

    return c


# adapted from bandmat (bandmat/tensor.pyx/dot_mm_plus_equals and dot_mm); see license above
def _banded_dot_banded(a, b, a_lu, b_lu, a_full_shape, b_full_shape, symmetric_output=False):
    """
    Calculates the matrix multiplication, ``C = A @ B``, with `a` and `b` in banded forms.

    `a` and `b` must be square matrices in their full form or else this calculation
    may be incorrect.

    Parameters
    ----------
    a : array-like, shape (`a_lu[0]` + `a_lu[1]` + 1, N)
        A banded matrix.
    b : array-like, shape (`b_lu[0]` + `b_lu[1]` + 1, N)
        The second banded matrix.
    a_lu : Container(int, int)
        A container of intergers designating the number lower and upper diagonals of `a`.
    b_lu : Container(int, int)
        A container of intergers designating the number lower and upper diagonals of `b`.
    a_full_shape : Container(int, int)
        A container of intergers designating the number of rows and columns in the full
        matrix representation of `a`.
    b_full_shape : Container(int, int)
        A container of intergers designating the number of rows and columns in the full
        matrix representation of `b`.
    symmetric_output : bool, optional
        Whether the output matrix is known to be symmetric. If True, will only calculate
        the matrix multiplication for the upper bands, and the lower bands will be filled
        in using the upper bands. Default is False.

    Returns
    -------
    output : numpy.ndarray
        The matrix multiplication of `a` and `b`. The number of lower diagonals is the
        minimum of `a_lu[0]` + `b_lu[0]` and `a_full_shape[0]` - 1, the number of upper
        diagonals is the minimum of `a_lu[1]` + `b_lu[1]` and `b_full_shape[1]` - 1, and
        the total shape is (lower diagonals + upper diagonals + 1, N).

    Raises
    ------
    ValueError
        Raised if `a` and `b` do not have the same number of rows or if `a_full_shape[1]`
        and `b_full_shape[0]` are not equal.

    Notes
    -----
    Derived from bandmat (https://github.com/MattShannon/bandmat/blob/master/bandmat/tensor.pyx),
    licensed under the BSD-3-Clause.

    """
    # TODO also need to check cases where a_lower + b_lower is > a_full_shape[0] - 1
    # and/or a_upper + b_upper is > b_full_shape[1] - 1 and see if the loops need to change at all
    if a_full_shape[1] != b_full_shape[0]:
        raise ValueError('dimension mismatch; a_full_shape[0] and b_full_shape[1] must be equal')
    # a and b must both be square banded matrices
    a = np.asarray(a)
    b = np.asarray(b)
    a_rows = a.shape[1]
    b_rows = b.shape[1]
    if a_rows != b_rows:
        raise ValueError('a and b must have the same number of rows')
    diag_length = a_rows  # main diagonal length
    a_lower, a_upper = a_lu
    b_lower, b_upper = b_lu
    c_upper = min(a_upper + b_upper, b_full_shape[1] - 1)
    c_lower = min(a_lower + b_lower, a_full_shape[0] - 1)
    if symmetric_output:
        lower_bound = 0  # only fills upper bands
    else:
        lower_bound = a_lower + b_lower
    # create output matrix outside of this function since numba's implementation
    # of np.zeros is much slower than numpy's (https://github.com/numba/numba/issues/7259)
    output = np.zeros((c_lower + c_upper + 1, diag_length))
    _numba_banded_dot_banded(
        a, b, output, a_lower, a_upper, b_lower, b_upper, c_upper, diag_length, lower_bound
    )

    if symmetric_output:
        for row in range(1, a_lower + b_lower + 1):
            offset = a_lower + b_lower + 1 - row
            # TODO should not use negative indices since empty 0 rows can sometimes
            # be added; instead, work down from main diagonal; or should at least use
            # the output's number of lower diagonals rather than a_l + b_l
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
    x = np.linspace(-1, 1, len(y))
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


# adapted from MATLAB beads version; see license above
def _high_pass_filter(data_size, freq_cutoff=0.005, filter_type=1, full_matrix=False):
    """
    Creates the banded matrices A and B such that B(A^-1) is a high pass filter.

    Parameters
    ----------
    data_size : int
        The number of data points.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    filter_type : int, optional
        An integer describing the high pass filter type. The order of the high pass
        filter is ``2 * filter_type``. Default is 1 (second order filter).
    full_matrix : bool, optional
        If True, will return the full sparse diagonal matrices of A and B. If False
        (default), will return the banded matrix versions of A and B.

    Raises
    ------
    ValueError
        Raised if `freq_cutoff` is not between 0 and 0.5 or if `filter_type` is
        less than 1.

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.

    """
    if not 0 < freq_cutoff < 0.5:
        raise ValueError('cutoff frequency must be between 0 and 0.5')
    elif filter_type < 1:
        raise ValueError('filter_type must be at least 1')

    # use finite differences instead of convolution to calculate a and b since
    # it's faster
    filter_order = 2 * filter_type
    b = np.zeros(2 * filter_order + 1)
    b[filter_order] = -1 if filter_type % 2 else 1  # same as (-1)**filter_type
    for _ in range(filter_order):
        b = b[:-1] - b[1:]
    a = abs(b)

    cos_freq = np.cos(2 * np.pi * freq_cutoff)
    t = ((1 - cos_freq) / max(1 + cos_freq, _MIN_FLOAT))**filter_type

    a_diags = np.repeat((b + a * t).reshape(1, -1), data_size, axis=0).T
    b_diags = np.repeat(b.reshape(1, -1), data_size, axis=0).T
    if full_matrix:
        offsets = np.arange(-filter_type, filter_type + 1)
        A = dia_object((a_diags, offsets), shape=(data_size, data_size)).tocsr()
        B = dia_object((b_diags, offsets), shape=(data_size, data_size)).tocsr()
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


# adapted from MATLAB beads version; see license above
def _beads_theta(x, asymmetry=6, eps_0=1e-6):
    """
    The cost function for the pure signal `x`.

    Parameters
    ----------
    x : numpy.ndarray
        The array of the signal.
    asymmetry : int, optional
        The asymmetrical parameter that determines the weighting of negative values
        compared to positive values in the cost function. Default is 6.0, which gives
        negative values six times more impact on the cost function that positive values.
        Set to 1 for a symmetric cost function, or a value less than 1 to weigh positive
        values more.
    eps_0 : float, optional
        The cutoff threshold between absolute loss and quadratic loss. Values in `x` with
        absolute value less than `eps_0` will have quadratic loss. Default is 1e-6.

    Returns
    -------
    abs_x : numpy.ndarray
        The absolute value of `x`. Used in other parts of the beads calculation, so
        return it here to avoid having to calculate again.
    large_mask : numpy.ndarray
        The boolean array indicating which values in `abs_x` are greater than `eps_0`.
        Used in other parts of the beads calculation, so return it here to avoid
        having to calculate again.
    theta : float
        The summation of the cost function of `x`.

    Notes
    -----
    The cost function is a modification of a Huber cost function, with the `asymmetry`
    parameter dictating the cost of negative values compared to positive values.

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.

    """
    abs_x = np.abs(x)
    large_mask = abs_x > eps_0
    small_x = x[~large_mask]

    theta = (
        x[(x > eps_0)].sum() - asymmetry * x[x < -eps_0].sum()
        + (
            ((1 + asymmetry) / (4 * eps_0)) * small_x**2 + ((1 - asymmetry) / 2) * small_x
            + eps_0 * (1 + asymmetry) / 4
        ).sum()
    )
    return abs_x, large_mask, theta


# adapted from MATLAB beads version; see license above
def _beads_loss(x, use_v2=True, eps_1=1e-6):
    """
    Approximates the absolute loss cost function.

    Parameters
    ----------
    x : numpy.ndarray
        The array of the absolute value of an n-order derivative of the signal.
    use_v2 : bool, optional
        If True (default), approximates the absolute loss using logarithms. If False,
        uses the square root of the sqaured values.
    eps_1 : float, optional
        A small, positive value used to prevent issues when the first or second order
        derivatives are close to zero. Default is 1e-6.

    Returns
    -------
    loss : numpy.ndarray
        The array of loss values.

    Notes
    -----
    The input `x` should be the absolute value of the array.

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.


    """
    if use_v2:
        loss = x - eps_1 * np.log(x + eps_1)
    else:
        loss = np.sqrt(x**2 + eps_1)

    return loss


# adapted from MATLAB beads version; see license above
def _beads_weighting(x, use_v2=True, eps_1=1e-6):
    """
    Approximates the weighting from absolute loss.

    Parameters
    ----------
    x : numpy.ndarray
        The array of the absolute value of an n-order derivative of the signal.
    use_v2 : bool, optional
        If True (default), approximates the absolute loss using logarithms. If False,
        uses the square root of the sqaured values.
    eps_1 : float, optional
        A small, positive value used to prevent issues when the first or second order
        derivatives are close to zero. Default is 1e-6.

    Returns
    -------
    weight : numpy.ndarray
        The weight array.

    Notes
    -----
    The input `x` should be the absolute value of the array.

    The calculation is `f'(x)/x`, where `f'(x)` is the derivative of the function
    `f(x)`, where `f(x)` is the loss function (calculated in _beads_loss).

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.

    """
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
        The array of the signal.
    smooth_half_window : int, optional
        The half-window for smoothing. Default is 0, which does no smoothing.

    Returns
    -------
    d1_x : numpy.ndarray, shape (N - 1,)
        The absolute value of the first derivative of `x`.
    d2_x : numpy.ndarray, shape (N - 2,)
        The absolute value of the second derivative of `x`.

    """
    d1_x = x[1:] - x[:-1]
    if smooth_half_window > 0:
        smooth_window = 2 * smooth_half_window + 1
        # TODO should mode be constant with cval=0 since derivative should be 0, or
        # does reflect give better results?
        # TODO should probably just smooth the first derivative and compute the second
        # derivative from the smoothed value rather than smoothing both.
        d2_x = np.abs(uniform_filter1d(d1_x[1:] - d1_x[:-1], smooth_window))
        uniform_filter1d(d1_x, smooth_window, output=d1_x)
    else:
        d2_x = np.abs(d1_x[1:] - d1_x[:-1])
    np.abs(d1_x, out=d1_x)

    return d1_x, d2_x


# adapted from MATLAB beads version; see license above
def _sparse_beads(y, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6,
                  filter_type=1, use_v2_loss=True, max_iter=50, tol=1e-2, eps_0=1e-6,
                  eps_1=1e-6, smooth_half_window=0):
    """
    The beads algorithm using full, sparse matrices.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N data points.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    lam_0 : float, optional
        The regularization parameter for the signal values. Default is 1.0. Higher
        values give a higher penalty.
    lam_1 : float, optional
        The regularization parameter for the first derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    lam_2 : float, optional
        The regularization parameter for the second derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    asymmetry : float, optional
        The asymmetrical parameter that determines the weighting of negative values
        compared to positive values in the cost function. Default is 6.0, which gives
        negative values six times more impact on the cost function that positive values.
        Set to 1 for a symmetric cost function, or a value less than 1 to weigh positive
        values more.
    filter_type : int, optional
        An integer describing the high pass filter type. The order of the high pass
        filter is ``2 * filter_type``. Default is 1 (second order filter).
    use_v2_loss : bool, optional
        If True (default), approximates the absolute loss using logarithms. If False,
        uses the square root of the sqaured values.
    max_iter : int, optional
        The maximum number of iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-2.
    eps_0 : float, optional
        The cutoff threshold between absolute loss and quadratic loss. Values in the signal
        with absolute value less than `eps_0` will have quadratic loss. Default is 1e-6.
    eps_1 : float, optional
        A small, positive value used to prevent issues when the first or second order
        derivatives are close to zero. Default is 1e-6.
    smooth_half_window : int, optional
        The half-window to use for smoothing the derivatives of the data with a moving
        average. Default is 0, which provides no smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the baseline.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Notes
    -----
    `A` and `B` matrices are symmetric, so their transposes are never used.

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.

    https://www.mathworks.com/matlabcentral/fileexchange/49974-beads-baseline-estimation-
    and-denoising-with-sparsity.

    """
    num_y = y.shape[0]
    d1_diags = np.zeros((5, num_y))
    d2_diags = np.zeros((5, num_y))
    offsets = np.arange(2, -3, -1)
    A, B = _high_pass_filter(num_y, freq_cutoff, filter_type, True)
    # factorize A since A is unchanged in the function and its factorization
    # is used repeatedly; much faster than calling spsolve each time
    A_factor = splu(A.tocsc(), permc_spec='NATURAL')
    BTB = B @ B

    x = y
    d1_x, d2_x = _abs_diff(x, smooth_half_window)
    # line 2 of Table 3 in beads paper
    d = BTB.dot(A_factor.solve(y)) - A.dot(np.full(num_y, lam_0 * (1 - asymmetry) / 2))
    gamma = np.empty(num_y)
    gamma_factor = lam_0 * (1 + asymmetry) / 2  # 2 * lam_0 * (1 + asymmetry) / 4
    cost_old = 0
    abs_x = np.abs(x)
    big_x = abs_x > eps_0
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        # calculate line 6 of Table 3 in beads paper using banded matrices rather
        # than sparse matrices since it is much faster; Gamma + D.T @ Lambda @ D

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

        # TODO check that 'NATURAL' is the appropriate permutation scheme for this
        x = A.dot(
            spsolve(
                BTB + A.dot(dia_object((d_diags, offsets), shape=(num_y, num_y)).tocsr()).dot(A),
                d, 'NATURAL'
            )
        )

        h = B.dot(A_factor.solve(y - x))
        d1_x, d2_x = _abs_diff(x, smooth_half_window)
        abs_x, big_x, theta = _beads_theta(x, asymmetry, eps_0)
        cost = (
            0.5 * h.dot(h)
            + lam_0 * theta
            + lam_1 * _beads_loss(d1_x, use_v2_loss, eps_1).sum()
            + lam_2 * _beads_loss(d2_x, use_v2_loss, eps_1).sum()
        )
        cost_difference = relative_difference(cost_old, cost)
        tol_history[i] = cost_difference
        if cost_difference < tol:
            break
        cost_old = cost

    diff = y - x
    baseline = diff - B.dot(A_factor.solve(diff))

    return baseline, {'signal': x, 'tol_history': tol_history[:i + 1]}


# adapted from MATLAB beads version; see license above
def _banded_beads(y, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6,
                  filter_type=1, use_v2_loss=True, max_iter=50, tol=1e-2, eps_0=1e-6,
                  eps_1=1e-6, smooth_half_window=0):
    """
    The beads algorithm using banded matrices rather than full, sparse matrices.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The y-values of the measured data, with N data points.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    lam_0 : float, optional
        The regularization parameter for the signal values. Default is 1.0. Higher
        values give a higher penalty.
    lam_1 : float, optional
        The regularization parameter for the first derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    lam_2 : float, optional
        The regularization parameter for the second derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    asymmetry : float, optional
        The asymmetrical parameter that determines the weighting of negative values
        compared to positive values in the cost function. Default is 6.0, which gives
        negative values six times more impact on the cost function that positive values.
        Set to 1 for a symmetric cost function, or a value less than 1 to weigh positive
        values more.
    filter_type : int, optional
        An integer describing the high pass filter type. The order of the high pass
        filter is ``2 * filter_type``. Default is 1 (second order filter).
    use_v2_loss : bool, optional
        If True (default), approximates the absolute loss using logarithms. If False,
        uses the square root of the sqaured values.
    max_iter : int, optional
        The maximum number of iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-2.
    eps_0 : float, optional
        The cutoff threshold between absolute loss and quadratic loss. Values in the signal
        with absolute value less than `eps_0` will have quadratic loss. Default is 1e-6.
    eps_1 : float, optional
        A small, positive value used to prevent issues when the first or second order
        derivatives are close to zero. Default is 1e-6.
    smooth_half_window : int, optional
        The half-window to use for smoothing the derivatives of the data with a moving
        average. Default is 0, which provides no smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the baseline.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Notes
    -----
    This function is ~75% faster than _sparse_beads (independent of data size) if Numba is
    installed due to the faster banded solvers. If Numba is not installed, the calculation
    of the dot product of banded matrices makes this calculation significantly slower than
    the sparse implementation.

    It is no faster to pre-compute the Cholesky factorization of A_lower and use
    that with scipy.linalg.cho_solve_banded compared to using A_lower in solveh_banded.

    `A` and `B` matrices are symmetric, so their transposes are never used.

    References
    ----------
    Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
    (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.

    https://www.mathworks.com/matlabcentral/fileexchange/49974-beads-baseline-estimation-
    and-denoising-with-sparsity.

    """
    num_y = y.shape[0]
    d1_diags = np.zeros((5, num_y))
    d2_diags = np.zeros((5, num_y))
    A, B = _high_pass_filter(num_y, freq_cutoff, filter_type, False)
    # the number of lower and upper diagonals for both A and B
    ab_lu = (filter_type, filter_type)
    # the shape of A and B, and D.T @ D matrices in their full forms rather than banded forms
    full_shape = (num_y, num_y)
    A_lower = A[filter_type:]
    BTB = _banded_dot_banded(B, B, ab_lu, ab_lu, full_shape, full_shape, True)
    # number of lower and upper diagonals of A.T @ (D.T @ D) @ A
    num_diags = (2 * filter_type + 2, 2 * filter_type + 2)

    # line 2 of Table 3 in beads paper
    d = (
        _banded_dot_vector(
            np.asfortranarray(BTB),
            solveh_banded(A_lower, y, check_finite=False, lower=True),
            (2 * filter_type, 2 * filter_type), full_shape
        )
        - _banded_dot_vector(
            A, np.full(num_y, lam_0 * (1 - asymmetry) / 2), ab_lu, full_shape
        )
    )
    gamma = np.empty(num_y)
    gamma_factor = lam_0 * (1 + asymmetry) / 2  # 2 * lam_0 * (1 + asymmetry) / 4
    x = y
    d1_x, d2_x = _abs_diff(x, smooth_half_window)
    cost_old = 0
    abs_x = np.abs(x)
    big_x = abs_x > eps_0
    tol_history = np.empty(max_iter + 1)
    for i in range(max_iter + 1):
        # calculate line 6 of Table 3 in beads paper using banded matrices rather
        # than sparse matrices since it is much faster; Gamma + D.T @ Lambda @ D

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
            _banded_dot_banded(A, d_diags, ab_lu, (2, 2), full_shape, full_shape),
            A, (filter_type + 2, filter_type + 2), ab_lu, full_shape, full_shape, True
        )
        temp[2:-2] += BTB

        # cannot use solveh_banded since temp is not guaranteed to be positive-definite
        # and diagonally-dominant
        x = _banded_dot_vector(
            A,
            solve_banded(num_diags, temp, d, overwrite_ab=True, check_finite=False),
            ab_lu, full_shape
        )

        abs_x, big_x, theta = _beads_theta(x, asymmetry, eps_0)
        d1_x, d2_x = _abs_diff(x, smooth_half_window)
        h = _banded_dot_vector(
            B,
            solveh_banded(A_lower, y - x, check_finite=False, overwrite_b=True, lower=True),
            ab_lu, full_shape
        )
        cost = (
            0.5 * h.dot(h)
            + lam_0 * theta
            + lam_1 * _beads_loss(d1_x, use_v2_loss, eps_1).sum()
            + lam_2 * _beads_loss(d2_x, use_v2_loss, eps_1).sum()
        )
        cost_difference = relative_difference(cost_old, cost)
        tol_history[i] = cost_difference
        if cost_difference < tol:
            break
        cost_old = cost

    diff = y - x
    baseline = (
        diff
        - _banded_dot_vector(
            B,
            solveh_banded(A_lower, diff, check_finite=False, overwrite_ab=True, lower=True),
            ab_lu, full_shape
        )
    )

    return baseline, {'signal': x, 'tol_history': tol_history[:i + 1]}


@_misc_wrapper
def beads(data, freq_cutoff=0.005, lam_0=1.0, lam_1=1.0, lam_2=1.0, asymmetry=6.0,
          filter_type=1, cost_function=2, max_iter=50, tol=1e-2, eps_0=1e-6,
          eps_1=1e-6, fit_parabola=True, smooth_half_window=None, x_data=None):
    r"""
    Baseline estimation and denoising with sparsity (BEADS).

    Decomposes the input data into baseline and pure, noise-free signal by modeling
    the baseline as a low pass filter and by considering the signal and its derivatives
    as sparse [4]_.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    freq_cutoff : float, optional
        The cutoff frequency of the high pass filter, normalized such that
        0 < `freq_cutoff` < 0.5. Default is 0.005.
    lam_0 : float, optional
        The regularization parameter for the signal values. Default is 1.0. Higher
        values give a higher penalty.
    lam_1 : float, optional
        The regularization parameter for the first derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    lam_2 : float, optional
        The regularization parameter for the second derivative of the signal. Default
        is 1.0. Higher values give a higher penalty.
    asymmetry : float, optional
        A number greater than 0 that determines the weighting of negative values
        compared to positive values in the cost function. Default is 6.0, which gives
        negative values six times more impact on the cost function that positive values.
        Set to 1 for a symmetric cost function, or a value less than 1 to weigh positive
        values more.
    filter_type : int, optional
        An integer describing the high pass filter type. The order of the high pass
        filter is ``2 * filter_type``. Default is 1 (second order filter).
    cost_function : {2, 1, "l1_v1", "l1_v2"}, optional
        An integer or string indicating which approximation of the l1 (absolute value)
        penalty to use. 1 or "l1_v1" will use :math:`l(x) = \sqrt{x^2 + \text{eps_1}}`
        and 2 (default) or "l1_v2" will use
        :math:`l(x) = |x| - \text{eps_1}\log{(|x| + \text{eps_1})}`.
    max_iter : int, optional
        The maximum number of iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-2.
    eps_0 : float, optional
        The cutoff threshold between absolute loss and quadratic loss. Values in the signal
        with absolute value less than `eps_0` will have quadratic loss. Default is 1e-6.
    eps_1 : float, optional
        A small, positive value used to prevent issues when the first or second order
        derivatives are close to zero. Default is 1e-6.
    fit_parabola : bool, optional
        If True (default), will fit a parabola to the data and subtract it before
        performing the beads fit as suggested in [5]_. This ensures the endpoints of
        the fit data are close to 0, which is required by beads. If the data is already
        close to 0 on both endpoints, set `fit_parabola` to False.
    smooth_half_window : int, optional
        The half-window to use for smoothing the derivatives of the data with a moving
        average and full window size of `2 * smooth_half_window + 1`. Smoothing can
        improve the convergence of the calculation, and make the calculation less sensitive
        to small changes in `lam_1` and `lam_2`, as noted in the pybeads package [6]_.
        Default is None, which will not perform any smoothing.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the baseline.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    Notes
    -----
    The default `lam_0`, `lam_1`, and `lam_2` values are good starting points for a
    dataset with 1000 points. Typically, smaller values are needed for larger datasets
    and larger values for smaller datasets.

    When finding the best parameters for fitting, it is usually best to find the optimal
    `freq_cutoff` for the noise in the data before adjusting any other parameters since
    it has the largest effect [5]_.

    Raises
    ------
    ValueError
        Raised if `asymmetry` is less than 0.

    References
    ----------
    .. [4] Ning, X., et al. Chromatogram baseline estimation and denoising using sparsity
           (BEADS). Chemometrics and Intelligent Laboratory Systems, 2014, 139, 156-167.
    .. [5] Navarro-Huerta, J.A., et al. Assisted baseline subtraction in complex chromatograms
           using the BEADS algorithm. Journal of Chromatography A, 2017, 1507, 1-10.
    .. [6] https://github.com/skotaro/pybeads.

    """
