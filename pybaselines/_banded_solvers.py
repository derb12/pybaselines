# -*- coding: utf-8 -*-
"""Dedicated solvers for pentadiagonal and heptadiagonal banded linear systems.

@author: Donald Erb
Created on February 20, 2025


The functions `ptrans1` and `ptrans2` were adapted from `pentapy`
(https://github.com/GeoStat-Framework/pentapy) (last accessed March 25, 2025), which was
licensed under the MIT license below.

The MIT License (MIT)

Copyright (c) 2023 Sebastian Müller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
from scipy.linalg import solve_banded

from ._compat import jit


# adapted from pentapy (https://github.com/GeoStat-Framework/pentapy); see license above
@jit(nopython=True, cache=True)
def _ptrans_1(lhs, rhs, overwrite_ab=False, overwrite_b=False):
    """
    Pentadiagonal banded matrix solver using transformations and backward substitution.

    Solves the equation ``A @ x = rhs``, given `A` in LAPACK banded format as `lhs`.

    Parameters
    ----------
    lhs : numpy.ndarray, shape (5, N)
        The pentadiagonal matrix `A` in LAPACK banded format banded format (see
        :func:`scipy.linalg.solve_banded`).
    rhs : numpy.ndarray, shape (N,)
        The right-hand side of the equation.
    overwrite_ab : bool, optional
        Whether to overwrite `lhs` when solving. Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `rhs` when solving. Default is False.

    Returns
    -------
    out : numpy.ndarray, shape (N,)
        The solution to the linear system, `x`.
    int
        If the matrix is singular, the returned integer is the row index plus one of where
        the singularity occurred. Otherwise, returns 0.

    Notes
    -----
    Derived from ``pentapy`` (https://github.com/GeoStat-Framework/pentapy), with modifications
    following pentapy issue #11 and pentapy pull request #27 for faster indexing and better error
    handling.

    Uses `lhs` in LAPACK banded format rather than the row-wise banded format used by pentapy.

    References
    ----------
    Askar, S., et al. On Solving Pentadiagonal Linear Systems via Transformations. Mathematical
    Problems in Engineering, 2015, 232456.

    """
    num_rows = lhs.shape[1]
    if overwrite_ab:
        # alpha and beta rows in the factorization correspond to lhs[1] and lhs[0], respectively,
        # when lhs is in LAPACK format, their values are referenced before those locations are set,
        # so can safely fill with the alpha and beta values
        alpha = lhs[1]
        beta = lhs[0]
    else:
        alpha = np.zeros(num_rows)
        beta = np.zeros(num_rows)
    if overwrite_b:
        out = rhs
    else:
        out = np.zeros(num_rows)

    # First row
    mu_i = lhs[2, 0]
    if mu_i == 0.0:
        return out, 1
    alpha_i_minus_1 = lhs[1, 1] / mu_i
    beta_i_minus_1 = lhs[0, 2] / mu_i
    z_i_minus_1 = rhs[0] / mu_i
    alpha[0] = alpha_i_minus_1
    beta[0] = beta_i_minus_1
    out[0] = z_i_minus_1

    # Second row
    gamma_i = lhs[3, 0]
    mu_i = lhs[2, 1] - alpha_i_minus_1 * gamma_i
    if mu_i == 0.0:
        return out, 2
    alpha_i = (lhs[1, 2] - beta_i_minus_1 * gamma_i) / mu_i
    alpha[1] = alpha_i
    beta_i = lhs[0, 3] / mu_i
    beta[1] = beta_i
    z_i = (rhs[1] - z_i_minus_1 * gamma_i) / mu_i
    out[1] = z_i

    # Central rows
    for i in range(2, num_rows - 2):
        e_i = lhs[4, i - 2]
        gamma_i = lhs[3, i - 1] - alpha_i_minus_1 * e_i
        mu_i = lhs[2, i] - beta_i_minus_1 * e_i - alpha_i * gamma_i
        if mu_i == 0.0:
            return out, i + 1

        alpha_i_plus_1 = (lhs[1, i + 1] - beta_i * gamma_i) / mu_i
        alpha_i_minus_1 = alpha_i
        alpha_i = alpha_i_plus_1
        alpha[i] = alpha_i

        beta_i_plus_1 = lhs[0, i + 2] / mu_i
        beta_i_minus_1 = beta_i
        beta_i = beta_i_plus_1
        beta[i] = beta_i

        z_i_plus_1 = (rhs[i] - z_i_minus_1 * e_i - z_i * gamma_i) / mu_i
        z_i_minus_1 = z_i
        z_i = z_i_plus_1
        out[i] = z_i

    # Second to last row
    row = num_rows - 2
    e_i = lhs[4, row - 2]
    gamma_i = lhs[3, row - 1] - alpha_i_minus_1 * e_i
    mu_i = lhs[2, row] - beta_i_minus_1 * e_i - alpha_i * gamma_i
    if mu_i == 0.0:
        return out, num_rows - 1

    alpha_i_plus_1 = (lhs[1, row + 1] - beta_i * gamma_i) / mu_i
    # Note that none of the below section is needed since only alpha[:num_rows - 2] and
    # beta[:num_rows - 2] is required for the backward substitution (would be needed if the
    # factorization was desired), but still compute it since it's cheap
    alpha[row] = alpha_i_plus_1
    if overwrite_ab:
        # have to account for LAPACK format not having zeros in top right corner as compared
        # to row-wise banded format
        beta[row:] = 0.0
        alpha[row + 1:] = 0.0
    # End of unneeded section

    z_i_plus_1 = (rhs[row] - z_i_minus_1 * e_i - z_i * gamma_i) / mu_i
    z_i_minus_1 = z_i
    z_i = z_i_plus_1

    # Last Row
    row = num_rows - 1
    e_i = lhs[4, row - 2]
    gamma_i = lhs[3, row - 1] - alpha_i * e_i
    mu_i = lhs[2, row] - beta_i * e_i - alpha_i_plus_1 * gamma_i
    if mu_i == 0.0:
        return out, num_rows

    # Backward substitution
    z_i_plus_1 = (rhs[row] - z_i_minus_1 * e_i - z_i * gamma_i) / mu_i
    z_i -= alpha_i_plus_1 * z_i_plus_1
    out[row] = z_i_plus_1
    out[num_rows - 2] = z_i

    for i in range(num_rows - 3, -1, -1):
        out[i] -= alpha[i] * z_i + beta[i] * z_i_plus_1
        z_i_plus_1 = z_i
        z_i = out[i]

    return out, 0


# adapted from pentapy (https://github.com/GeoStat-Framework/pentapy); see license above
@jit(nopython=True, cache=True)
def _ptrans_2(lhs, rhs, overwrite_ab=False, overwrite_b=False):
    """
    Pentadiagonal banded matrix solver using transformations and forward substitution.

    Solves the equation ``A @ x = rhs``, given `A` in LAPACK banded format as `lhs`.

    Parameters
    ----------
    lhs : numpy.ndarray, shape (5, N)
        The pentadiagonal matrix `A` in LAPACK banded format banded format (see
        :func:`scipy.linalg.solve_banded`).
    rhs : numpy.ndarray, shape (N,)
        The right-hand side of the equation.
    overwrite_ab : bool, optional
        Whether to overwrite `lhs` when solving. Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `rhs` when solving. Default is False.

    Returns
    -------
    out : numpy.ndarray, shape (N,)
        The solution to the linear system, `x`.
    int
        If the matrix is singular, the returned integer is the row index plus one of where
        the singularity occurred. Otherwise, returns 0.

    Notes
    -----
    Derived from ``pentapy`` (https://github.com/GeoStat-Framework/pentapy), with modifications
    following pentapy issue #11 and pentapy pull request #27 for faster indexing and better error
    handling.

    Uses `lhs` in LAPACK banded format rather than the row-wise banded format used by pentapy.

    References
    ----------
    Askar, S., et al. On Solving Pentadiagonal Linear Systems via Transformations. Mathematical
    Problems in Engineering, 2015, 232456.

    """
    num_rows = lhs.shape[1]
    if overwrite_ab:
        # sigma and phi rows in the factorization correspond to lhs[3] and lhs[4], respectively;
        # when lhs is in LAPACK format, their values are referenced before those locations are set,
        # so can safely fill with the sigma and phi values
        sigma = lhs[3]
        phi = lhs[4]
    else:
        sigma = np.zeros(num_rows)
        phi = np.zeros(num_rows)
    if overwrite_b:
        out = rhs
    else:
        out = np.zeros(num_rows)

    # First row
    row = num_rows - 1
    psi_i = lhs[2, row]
    if psi_i == 0.0:
        return out, num_rows

    sigma_i_plus_1 = lhs[3, row - 1] / psi_i
    phi_i_plus_1 = lhs[4, row - 2] / psi_i
    w_i_plus_1 = rhs[row] / psi_i

    sigma[row] = sigma_i_plus_1
    phi[row] = phi_i_plus_1
    out[row] = w_i_plus_1

    # Second row
    row = num_rows - 2
    rho_i = lhs[1, row + 1]
    psi_i = lhs[2, row] - sigma_i_plus_1 * rho_i
    if psi_i == 0.0:
        return out, num_rows - 1

    sigma_i = (lhs[3, row - 1] - phi_i_plus_1 * rho_i) / psi_i
    phi_i = lhs[4, row - 2] / psi_i
    w_i = (rhs[row] - w_i_plus_1 * rho_i) / psi_i

    sigma[row] = sigma_i
    phi[row] = phi_i
    out[row] = w_i

    # Central rows
    for i in range(num_rows - 3, 1, -1):
        b_i = lhs[0, i + 2]
        rho_i = lhs[1, i + 1] - sigma_i_plus_1 * b_i
        psi_i = lhs[2, i] - phi_i_plus_1 * b_i - sigma_i * rho_i
        if psi_i == 0.0:
            return out, i + 1

        sigma_i_minus_1 = (lhs[3, i - 1] - phi_i * rho_i) / psi_i
        sigma_i_plus_1 = sigma_i
        sigma_i = sigma_i_minus_1
        phi_i_minus_1 = lhs[4, i - 2] / psi_i
        phi_i_plus_1 = phi_i
        phi_i = phi_i_minus_1

        w_i_minus_1 = (rhs[i] - w_i_plus_1 * b_i - w_i * rho_i) / psi_i
        w_i_plus_1 = w_i
        w_i = w_i_minus_1

        sigma[i] = sigma_i
        phi[i] = phi_i
        out[i] = w_i

    # Second to last row
    b_i = lhs[0, 3]
    rho_i = lhs[1, 2] - sigma_i_plus_1 * b_i
    psi_i = lhs[2, 1] - phi_i_plus_1 * b_i - sigma_i * rho_i
    if psi_i == 0.0:
        return out, 2

    sigma_i_minus_1 = (lhs[3, 0] - phi_i * rho_i) / psi_i
    sigma_i_plus_1 = sigma_i
    sigma_i = sigma_i_minus_1

    w_i_minus_1 = (rhs[1] - w_i_plus_1 * b_i - w_i * rho_i) / psi_i
    w_i_plus_1 = w_i
    w_i = w_i_minus_1

    # Note that none of the below section is needed since only sigma[2:] and phi[2:] is required
    # for the forward substitution (would be needed if the factorization was desired), but still
    # compute it since it's cheap
    sigma[1] = sigma_i
    if overwrite_ab:
        # have to account for LAPACK format not having zeros in bottom left corner as compared
        # to row-wise banded format
        sigma[0] = 0.0
        phi[:2] = 0.0
    # End of unneeded section

    # Last row
    b_i = lhs[0, 2]
    rho_i = lhs[1, 1] - sigma_i_plus_1 * b_i
    psi_i = lhs[2, 0] - phi_i * b_i - sigma_i * rho_i
    if psi_i == 0.0:
        return out, 1

    w_i_minus_1 = (rhs[0] - w_i_plus_1 * b_i - w_i * rho_i) / psi_i
    out[0] = w_i_minus_1

    # Forward substitution
    w_i -= sigma_i * w_i_minus_1
    out[1] = w_i
    w_i_plus_1 = out[2]
    for i in range(2, num_rows):
        # note that the phi[i] * w_i_minus_1 is added so that -= applies subtraction
        out[i] -= sigma[i] * w_i + phi[i] * w_i_minus_1
        w_i_minus_1 = w_i
        w_i = out[i]

    return out, 0


def solve_banded_penta(ab, b, solver=1, overwrite_ab=False, overwrite_b=False):
    """
    Pentadiagonal banded matrix solver using transformations.

    Solves the equation ``A @ x = rhs``, given `A` in LAPACK banded format as `lhs`.

    Parameters
    ----------
    ab : numpy.ndarray, shape (5, N)
        The pentadiagonal matrix `A` in LAPACK banded format banded format (see
        :func:`scipy.linalg.solve_banded`).
    b : numpy.ndarray, shape (N,)
        The right-hand side of the equation.
    solver : {1, 2}
        The solver to use. 1 designates the PTRANS-I algorithm from [1]_, and 2 designates
        the PTRANS-II algorithm from [1]_.
    overwrite_ab : bool, optional
        Whether to overwrite `ab` when solving. Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `b` when solving. Default is False.

    Returns
    -------
    result : numpy.ndarray, shape (N,)
        The solution to the linear system, `x`.

    Raises
    ------
    ValueError
        Raised if `ab` has less than 3 columns or if `ab` does not have 5 rows.
    numpy.linalg.LinAlgError
        Raised if `ab` is singular.

    References
    ----------
    .. [1] Askar, S., et al. On Solving Pentadiagonal Linear Systems via Transformations.
           Mathematical Problems in Engineering, 2015, 232456.

    """
    lhs = np.asarray(ab, dtype=float)
    rhs = np.asarray(b, dtype=float)
    # TODO look at how scipy handles overwrite_[ab][b] for when np.asarray already copies
    # the data
    rows, columns = ab.shape
    if rows != 5:
        raise ValueError('ab matrix must have 5 rows')
    if columns < 4:
        # solvers always directly access first 4 columns, so use solve_banded instead
        return solve_banded(
            (2, 2), lhs, rhs, overwrite_ab=overwrite_ab, overwrite_b=overwrite_b,
            check_finite=False
        )

    func = {1: _ptrans_1, 2: _ptrans_2}[solver]
    result, info = func(lhs, rhs, overwrite_ab, overwrite_b)
    if info > 0:
        raise np.linalg.LinAlgError(f'Matrix is singular at row index {info - 1}')

    return result

