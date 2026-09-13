# -*- coding: utf-8 -*-
"""Tests for pybaselines.misc.

@author: Donald Erb
Created on March 20, 2021

"""

from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.sparse import issparse, vstack

from pybaselines import _banded_utils, misc
from pybaselines.utils import gaussian
from pybaselines._compat import dia_object, diags

from .base_tests import BaseTester, ConvergenceMixin, ensure_deprecation, get_data


class MiscTester(BaseTester):
    """Base testing class for miscellaneous functions."""

    module = misc


@pytest.mark.filterwarnings('ignore:"interp_pts" is deprecated')
class TestInterpPts(MiscTester):
    """Class for testing interp_pts baseline."""

    func_name = 'interp_pts'
    required_kwargs = {'baseline_points': ((5, 10), (10, 20), (90, 100))}
    required_repeated_kwargs = {'baseline_points': ((5, 10), (10, 20), (90, 100))}

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('interp_method', ('linear', 'slinear', 'quadratic'))
    def test_unchanged_data(self, use_class, interp_method):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, interp_method=interp_method
        )

    def test_no_x(self):
        """
        Ensures that function output is the same when no x is input.

        Since `interp_pts` depends heavily on x, this test just ensures that
        the function call works without an x-input as long as `data` is input.

        """
        self.class_func(data=self.y, **self.kwargs)
        getattr(self.algorithm_base(), self.func_name)(data=self.y, **self.kwargs)

    @pytest.mark.parametrize('points', ([], [1], [[1], [1]], [1, 2, 3]))
    def test_non_2d_baseline_points_fails(self, points):
        """Ensures an error is raised if there are less than two x-y baseline points."""
        with pytest.raises(ValueError):
            self.class_func(baseline_points=points)

    def test_no_y(self):
        """Ensures the function works when no y-values are input."""
        self.class_func(**self.kwargs)
        self.func(x_data=self.x, **self.kwargs)

    def test_no_y_no_x_fails(self):
        """Ensures an error is raised when both x and y are not input."""
        with pytest.raises(TypeError):
            getattr(self.algorithm_base(), self.func_name)(**self.kwargs)
        with pytest.raises(TypeError):
            self.func(**self.kwargs)

    @ensure_deprecation(1, 5)
    def test_method_deprecation(self):
        """Ensures the deprecation warning is emitted if this method is used."""
        with pytest.warns(DeprecationWarning):
            self.class_func(data=self.y, **self.kwargs)


class TestBeads(MiscTester, ConvergenceMixin):
    """Class for testing beads baseline."""

    func_name = 'beads'
    checked_keys = ('signal', 'tol_history', 'fidelity', 'penalty', 'success')

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('cost_function', (1, 2, 'l1_v1', 'l1_v2', 'L1_V1'))
    @pytest.mark.parametrize('smooth_hw', (None, 0, 5))
    @pytest.mark.parametrize('fit_parabola', (True, False))
    def test_unchanged_data(self, use_class, cost_function, smooth_hw, fit_parabola):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, cost_function=cost_function,
            smooth_half_window=smooth_hw, fit_parabola=fit_parabola
        )

    @pytest.mark.parametrize('use_banded', (True, False))
    def test_output(self, use_banded):
        """
        Ensures that the output has the desired format.

        Tests both beads implementations.

        """
        with mock.patch.object(misc, '_HAS_NUMBA', use_banded):
            super().test_output()

    @pytest.mark.parametrize('banded_factorization', (True, False))
    @pytest.mark.parametrize('cost_function', (1, 2))
    def test_beads_algorithms(self, cost_function, banded_factorization):
        """Ensure the sparse and banded forms for beads give similar results."""
        # banded beads function always works, just is slower when numba is not installed
        fitter = self.algorithm_base()
        with mock.patch.object(misc, '_HAS_NUMBA', not misc._HAS_NUMBA):
            output_1, params_1 = fitter.beads(self.y, cost_function=cost_function)
        if not banded_factorization:
            fitter.banded_solver = 4  # will use sparse factorization
        output_2, params_2 = fitter.beads(self.y, cost_function=cost_function)

        assert_allclose(output_1, output_2, rtol=5e-6, atol=1e-10)
        assert_allclose(params_1['signal'], params_2['signal'], rtol=3e-4, atol=1e-10)

    @pytest.mark.parametrize('asymmetry', (0, -1))
    def test_bad_asymmetry_fails(self, asymmetry):
        """Ensure an asymmetry value < 1 raises a ValueError."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetry=asymmetry)

    @pytest.mark.parametrize('cost_function', (0, 3, 'l2_v2'))
    def test_unknown_cost_function_fails(self, cost_function):
        """Ensure an non-covered cost function raises a KeyError."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function=cost_function)

    @pytest.mark.parametrize('use_banded', (True, False))
    def test_tol_history(self, use_banded):
        """
        Ensures the 'tol_history' item in the parameter output is correct.

        Tests both beads implementations.

        """
        max_iter = 5
        with mock.patch.object(misc, '_HAS_NUMBA', use_banded):
            _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1

    @pytest.mark.parametrize('negative_lam', [0, 1, 2])
    def test_negative_lam_fails(self, negative_lam):
        """Ensures that a negative regularization parameter fails."""
        lams = [1, 1, 1]
        lams[negative_lam] *= -1
        with pytest.raises(ValueError):
            self.class_func(self.y, lam_0=lams[0], lam_1=lams[1], lam_2=lams[2])

    @pytest.mark.parametrize('zero_lam', [0, 1, 2])
    def test_zero_lam_passes(self, zero_lam):
        """Ensures that a zero-valued regularization parameter passes."""
        lams = [1, 1, 1]
        lams[zero_lam] = 0
        self.class_func(self.y, lam_0=lams[0], lam_1=lams[1], lam_2=lams[2])

    def test_array_lam_fails(self):
        """Ensures array-like lam_[0, 1, 2] values raise an exception."""
        array_vals = np.ones_like(self.y)
        with pytest.raises(ValueError):
            self.class_func(self.y, lam_0=array_vals)
        with pytest.raises(ValueError):
            self.class_func(self.y, lam_1=array_vals)
        with pytest.raises(ValueError):
            self.class_func(self.y, lam_2=array_vals)

    @pytest.mark.parametrize('reject_outliers', (True, False))
    def test_parabola_len(self, reject_outliers):
        """Ensures `parabola_len` allows ignoring outlier points on the edges for better beads fits.

        Reproducer code slightly modified from https://github.com/derb12/pybaselines/issues/70.

        """
        x = np.linspace(0, 1000, 1000)
        signal = (
            gaussian(x, 9, 100, 12)
            + gaussian(x, 6, 180, 5)
            + gaussian(x, 8, 350, 11)
            + gaussian(x, 15, 400, 18)
            + gaussian(x, 6, 550, 6)
            + gaussian(x, 13, 700, 8)
            + gaussian(x, 9, 800, 9)
            + gaussian(x, 9, 880, 7)
        )
        baseline = 15 - 10 * np.exp(-x / 600)
        noise = np.random.default_rng(0).normal(0, 0.05, len(x))
        y = signal + baseline + noise
        y2 = 1. * y
        y2 += gaussian(x, 9, 990, 7)
        y2[0] += 20

        fitter = self.algorithm_base(x)

        parabola_len = (3, 80) if reject_outliers else 0
        good_fit = fitter.beads(
            y, lam_0=0.015, lam_1=0.1, lam_2=1, fit_parabola=True, parabola_len=parabola_len
        )[0]
        fit = fitter.beads(
            y2, lam_0=0.015, lam_1=0.1, lam_2=1, fit_parabola=True, parabola_len=parabola_len
        )[0]

        if reject_outliers:
            assert_allclose(fit, good_fit, rtol=0.05, atol=0.1)
        else:
            # sanity check that without outlier rejection on the end points, the fit on y2 is
            # significantly worse
            with pytest.raises(AssertionError):
                assert_allclose(fit, good_fit, rtol=1, atol=10)

    @pytest.mark.parametrize('use_banded', (True, False))
    @pytest.mark.parametrize('sparse_factorization', (True, False))
    @pytest.mark.parametrize('condition', (1, 2, 3))
    def test_reference_comparison(self, condition, use_banded, sparse_factorization):
        """
        Compares against the MATLAB version written by the BEADS authors.

        The reference BEADS code given by the authors is available at
        https://www.mathworks.com/matlabcentral/fileexchange/49974-beads-baseline-estimation-and-denoising-with-sparsity.
        With the above file in the working path, data for the fits was generated using Octave
        version 11.3.0 and BEADS version 1.7.0.1 with the following code::

            y = csvread(file path here);
            f_order = 1;
            asymmetry = 6;  % or 10
            freq_cutoff = 0.08;  % or 0.02 or 0.04
            alpha = 1.;  % or 15.
            lam0 = alpha / norm(y, 1);
            lam1 = alpha / norm(diff(y, 1), 1);
            lam2 = alpha / norm(diff(y, 2), 1);
            [signal, baseline, cost] = beads(y, f_order, freq_cutoff, asymmetry, lam0, lam1, lam2);
            % then save baseline and signal

        """
        x = np.linspace(0, 1000, 100)
        baseline = -2e-5 * (x - 500)**2 + 5
        signal = (
            gaussian(x, 9, 100, 12)
            + gaussian(x, 6, 180, 5)
            + gaussian(x, 8, 350, 11)
            + gaussian(x, 15, 400, 18)
            + gaussian(x, 6, 550, 6)
            + gaussian(x, 13, 700, 8)
            + gaussian(x, 9, 800, 9)
            + gaussian(x, 9, 880, 7)
        )
        noise = np.random.default_rng(0).normal(0, 0.2, len(x))
        y = signal + baseline + noise
        freq_cutoff, alpha, asymmetry = {
            1: (0.08, 1., 6),
            2: (0.02, 15., 6),
            3: (0.04, 1., 10),
        }[condition]

        fitter = self.algorithm_base()
        if sparse_factorization:
            fitter.banded_solver = 4
        with mock.patch.object(misc, '_HAS_NUMBA', use_banded):
            # NOTE: the MATLAB beads version always does 30 iterations, fixes eps_0 and eps_1
            # to 1e-6, and uses L1_v2 cost function
            fit, params = fitter.beads(
                y, freq_cutoff=freq_cutoff, alpha=alpha, asymmetry=asymmetry,
                tol=-1, max_iter=29, lam_0=None, lam_1=None, lam_2=None, fit_parabola=False,
                cost_function=2, eps_0=1e-6, eps_1=1e-6
            )

        # baseline and signal
        expected_output = np.loadtxt(
            Path(__file__).parent.joinpath(f'data/beads_{condition}.csv'), delimiter=',',
            skiprows=1
        )
        # unfortunately need fairly high rtols; since matlab BEADS version always
        # does 30 iterations, any small deviations in the calcs for one iteration get magnified
        baseline_rtol = {1: 5e-9, 2: 1e-6, 3: 5e-9}[condition]
        signal_rtol = {1: 1e-8, 2: 2e-4, 3: 5e-6}[condition]

        assert_allclose(fit, expected_output[:, 0], rtol=baseline_rtol, atol=1e-12)
        assert_allclose(params['signal'], expected_output[:, 1], rtol=signal_rtol, atol=1e-10)


def test_banded_dot_banded():
    """Ensures the dot product of two square banded matrices is correct."""
    # random, square, non-symmetric banded matrix; tests that the number of upper and
    # lower diagonals in the output is capped by the shape of the matrix rather than the
    # number of diagonals, since matrix_1 @ matrix_1 would otherwise have more diagonals
    # than allowed in the shape
    matrix_1 = dia_object(np.array([
        [0, 1, 0, 0, 0],
        [1, 3, 4, 0, 0],
        [2, 4, 9, 8, 0],
        [9, 5, 12, -4, 19],
        [0, -29, 8, 29, 12]
    ]))
    bands_1 = _banded_utils._sparse_to_banded(matrix_1)[0]

    actual_output_1 = _banded_utils._sparse_to_banded(matrix_1 @ matrix_1)[0]
    banded_output_1 = misc._banded_dot_banded(
        bands_1, bands_1, (3, 1), (3, 1), matrix_1.shape, matrix_1.shape
    )
    assert_allclose(banded_output_1, actual_output_1, rtol=1e-16, atol=1e-16)

    # random, square, symmetric banded matrix
    matrix_2 = dia_object(np.array([
        [0, 1, 22, 0, 0, 0],
        [1, 3, 4, 5, 0, 0],
        [22, 4, 9, 97, -3, 0],
        [0, 5, 97, -4, 19, 12],
        [0, 0, -3, 19, 12, 8],
        [0, 0, 0, 12, 8, 7]
    ]))
    bands_2 = matrix_2.todia().data[::-1]

    actual_output_2 = _banded_utils._sparse_to_banded(matrix_2 @ matrix_2)[0]
    banded_output_2 = misc._banded_dot_banded(
        bands_2, bands_2, (2, 2), (2, 2), matrix_2.shape, matrix_2.shape
    )
    assert_allclose(banded_output_2, actual_output_2, rtol=1e-16, atol=1e-16)


def test_parabola():
    """Ensures the parabola fits the two endpoints and equals min(y) at the midpoint."""
    num_points = 51
    x = np.arange(num_points)
    mid_point = num_points // 2
    y = np.sin(x) + 0.1 * x + np.random.default_rng(5).uniform(0, 0.05, x.size)

    parabola = misc._parabola(y)

    assert_allclose(parabola[0], y[0], rtol=1e-14, atol=1e-14)
    assert_allclose(parabola[-1], y[-1], rtol=1e-14, atol=1e-14)
    assert_allclose(parabola[mid_point], y.min(), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('noisy_left', (True, False))
@pytest.mark.parametrize('noisy_right', (True, False))
@pytest.mark.parametrize('parabola_len', (3, (3, 60), 0, 1))
def test_parabola_len(noisy_left, noisy_right, parabola_len):
    """Ensures `parabola_len` allows ignoring outlier points on the edges when fitting a parabola.

    Reproducer code from https://github.com/derb12/pybaselines/issues/70.

    """
    x = np.linspace(0, 1000, 1000)
    signal = (
        gaussian(x, 9, 100, 12)
        + gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 11)
        + gaussian(x, 15, 400, 18)
        + gaussian(x, 6, 550, 6)
        + gaussian(x, 13, 700, 8)
        + gaussian(x, 9, 800, 9)
        + gaussian(x, 9, 880, 7)
    )
    baseline = 10 - 10 * np.exp(-x / 600)
    noise = np.random.default_rng(0).normal(0, 0.05, len(x))
    y = signal + baseline + noise
    if isinstance(parabola_len, int):
        left_edge = right_edge = parabola_len
    else:
        left_edge, right_edge = parabola_len

    if noisy_right:
        # partial peak on the end of the signal; needs a larger right edge to ignore
        y += gaussian(x, 9, 990, 7)
    if noisy_right and right_edge > 15:
        expected_right = np.median(y[-right_edge - 1:])
    else:
        expected_right = y[-1]
    if noisy_left:
        y[0] += 20
    if noisy_left and left_edge > 1:
        expected_left = np.median(y[:left_edge + 1])
    else:
        expected_left = y[0]

    A = y.min()
    y1 = expected_left - A
    y2 = expected_right - A
    C = (y2 + y1) / 2
    B = C - y1
    parabola_x = np.linspace(-1, 1, len(y))
    expected_parabola = A + B * parabola_x + C * parabola_x**2

    parabola = misc._parabola(y, parabola_len=parabola_len)

    assert_allclose(parabola, expected_parabola, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('banded_factorization', (True, False))
@pytest.mark.parametrize('banded', (True, False))
@pytest.mark.parametrize('filter_type', (1, 2))
def test_high_pass_filter_simple(filter_type, banded, banded_factorization):
    """Simple test to ensure _high_pass_filter works."""
    num_points = 5
    freq_cutoff = 0.3
    if filter_type == 1:
        desired_B_full = np.array([
            [2., -1., 0., 0., 0.],
            [-1., 2., -1., 0., 0.],
            [0., -1., 2., -1., 0.],
            [0., 0., -1., 2., -1.],
            [0., 0., 0., -1., 2.]
        ])
        desired_A_full = np.array([
            [5.78885438, 0.89442719, 0., 0., 0.],
            [0.89442719, 5.78885438, 0.89442719, 0., 0.],
            [0., 0.89442719, 5.78885438, 0.89442719, 0.],
            [0., 0., 0.89442719, 5.78885438, 0.89442719],
            [0., 0., 0., 0.89442719, 5.78885438]
        ])
    else:
        desired_B_full = np.array([
            [6., -4., 1., 0., 0.],
            [-4., 6., -4., 1., 0.],
            [1., -4., 6., -4., 1.],
            [0., 1., -4., 6., -4.],
            [0., 0., 1., -4., 6.]
        ])
        desired_A_full = np.array([
            [27.53312629, 10.35541753, 4.58885438, 0., 0.],
            [10.35541753, 27.53312629, 10.35541753, 4.58885438, 0.],
            [4.58885438, 10.35541753, 27.53312629, 10.35541753, 4.58885438],
            [0., 4.58885438, 10.35541753, 27.53312629, 10.35541753],
            [0., 0., 4.58885438, 10.35541753, 27.53312629]
        ])
    desired_A_banded = _banded_utils._sparse_to_banded(dia_object(desired_A_full))[0]
    desired_B_banded = _banded_utils._sparse_to_banded(dia_object(desired_B_full))[0]

    A_banded, A_sparse, A_factorization, B_banded, BTB = misc._high_pass_filter(
        num_points, freq_cutoff, filter_type, banded=banded,
        banded_factorization=banded_factorization
    )
    B_sparse = _banded_utils._banded_to_sparse(B_banded, lower=False)
    BTB_expected = B_sparse.T @ B_sparse
    if banded:
        BTB_expected = _banded_utils._sparse_to_banded(BTB_expected)[0]
    else:
        BTB_expected = BTB_expected.toarray()
        BTB = BTB.toarray()

    # limited by precision of the simple arrays above
    assert_allclose(A_sparse.toarray(), desired_A_full, rtol=2e-9, atol=1e-15)
    assert_allclose(B_sparse.toarray(), desired_B_full, rtol=1e-15, atol=1e-15)
    assert_allclose(A_banded, desired_A_banded, rtol=2e-9, atol=1e-15)
    assert_allclose(B_banded, desired_B_banded, rtol=1e-15, atol=1e-15)
    assert_allclose(BTB, BTB_expected, rtol=1e-16, atol=1e-16)

    # check that the full A and B matrices are symmetric
    assert_allclose(A_sparse.T.toarray(), A_sparse.toarray(), rtol=1e-16, atol=1e-16)
    assert_allclose(B_sparse.T.toarray(), B_sparse.toarray(), rtol=1e-16, atol=1e-16)

    eye = np.eye(num_points)
    expected_A_inv = np.linalg.solve(desired_A_full, eye)
    if banded_factorization:
        assert A_factorization.shape == (filter_type + 1, num_points)  # lower bands of A
    else:
        assert callable(A_factorization)
    A_inv = misc._factorized_solve(A_factorization, eye)
    assert_allclose(A_inv, expected_A_inv, atol=1e-14, rtol=5e-9)

    assert A_sparse.shape == (num_points, num_points)
    assert B_sparse.shape == (num_points, num_points)
    assert A_banded.shape == (2 * filter_type + 1, num_points)
    assert B_banded.shape == (2 * filter_type + 1, num_points)
    if banded:
        assert BTB.shape == (2 * (2 * filter_type) + 1, num_points)
    else:
        assert BTB.shape == (num_points, num_points)


@pytest.mark.parametrize('banded_factorization', (True, False))
@pytest.mark.parametrize('banded', (True, False))
@pytest.mark.parametrize('num_points', (100, 201))
@pytest.mark.parametrize('filter_type', (1, 2, 3, 4))
@pytest.mark.parametrize('freq_cutoff', (0.499999, 0.1, 0.01, 0.001, 0.00001))
def test_high_pass_filter(num_points, filter_type, freq_cutoff, banded, banded_factorization):
    """Tests various inputs for _high_pass_filter to ensure output is correct."""
    A_banded, A_sparse, A_factorization, B_banded, BTB = misc._high_pass_filter(
        num_points, freq_cutoff, filter_type, banded, banded_factorization
    )
    B_sparse = _banded_utils._banded_to_sparse(B_banded, lower=False)

    assert_allclose(
        A_banded, _banded_utils._sparse_to_banded(A_sparse)[0], rtol=1e-16, atol=1e-16
    )
    assert_allclose(
        B_banded, _banded_utils._sparse_to_banded(B_sparse)[0], rtol=1e-16, atol=1e-16
    )

    # check that the full A and B matrices are symmetric
    assert_allclose(A_sparse.T.toarray(), A_sparse.toarray(), rtol=1e-16, atol=1e-16)
    assert_allclose(B_sparse.T.toarray(), B_sparse.toarray(), rtol=1e-16, atol=1e-16)

    if banded_factorization:
        assert A_factorization.shape == (filter_type + 1, num_points)  # lower bands of A
    else:
        assert callable(A_factorization)

    assert A_sparse.shape == (num_points, num_points)
    assert B_sparse.shape == (num_points, num_points)
    assert A_banded.shape == (2 * filter_type + 1, num_points)
    assert B_banded.shape == (2 * filter_type + 1, num_points)
    if banded:
        assert BTB.shape == (2 * (2 * filter_type) + 1, num_points)
    else:
        assert BTB.shape == (num_points, num_points)

    actual_BTB = B_sparse.T @ B_sparse
    # check that B.T @ B is the same as B @ B since B is symmetric
    assert_allclose(
        actual_BTB.toarray(), (B_sparse @ B_sparse).toarray(), rtol=1e-16, atol=1e-16
    )
    if banded:
        actual_BTB_banded = _banded_utils._sparse_to_banded(actual_BTB)[0]
        assert_allclose(BTB, actual_BTB_banded, rtol=1e-16, atol=1e-16)
    else:
        assert issparse(BTB)
        assert_allclose(
            BTB.toarray(), actual_BTB.toarray(), rtol=1e-16, atol=1e-16
        )


@pytest.mark.parametrize('freq_cutoff', (0, 0.5, -0.5, 5))
def test_high_pass_filter_bad_freqcutoff_fails(freq_cutoff):
    """Ensures a ValueError is raised for incorrect freq_cutoff values."""
    with pytest.raises(ValueError):
        misc._high_pass_filter(10, freq_cutoff=freq_cutoff)


@pytest.mark.parametrize('filter_type', (0, -1))
def test_high_pass_filter_bad_filtertype_fails(filter_type):
    """Ensures a ValueError is raised for incorrect filter_type values."""
    with pytest.raises(ValueError):
        misc._high_pass_filter(10, filter_type=filter_type)


@pytest.mark.parametrize('filter_type', np.arange(1, 10))
def test_high_pass_filter_convolution_matrix_hack(filter_type):
    """Ensure the trick used for calculating the convolution matrix coefficients is correct."""
    # the actual calculation from the beads MATLAB source using convolution
    b_actual = np.array([1, -1])
    convolve_array = np.array([-1, 2, -1])
    for _ in range(filter_type - 1):
        b_actual = np.convolve(b_actual, convolve_array)
    b_actual = np.convolve(b_actual, np.array([-1, 1]))

    a_actual = 1
    convolve_array = np.array([1, 2, 1])
    for _ in range(filter_type):
        a_actual = np.convolve(a_actual, convolve_array)

    # the faster alternative using finite differences
    filter_order = 2 * filter_type
    b = np.zeros(2 * filter_order + 1)
    b[filter_order] = -1 if filter_type % 2 else 1  # same as (-1)**filter_type
    for _ in range(filter_order):
        b = b[:-1] - b[1:]
    a = abs(b)

    assert_array_equal(a, a_actual)
    assert_array_equal(b, b_actual)


@pytest.mark.parametrize('num_points', (100, 203))
@pytest.mark.parametrize('lam_1', (1e-5, 2., 1e6))
@pytest.mark.parametrize('lam_2', (1e-5, 2., 1e6))
@pytest.mark.parametrize('filter_type', (1, 2, 3))
@pytest.mark.parametrize('freq_cutoff', (0.49, 0.01, 0.001, 2e-5))
def test_beads_diff_matrix_calculation(num_points, lam_1, lam_2, filter_type, freq_cutoff):
    """
    Check that the lam * (D.T @ Lam @ D) and A.T @ M @ A calculations are correct.

    D is the stacked first and second order difference matrices, Lam is a diagonal matrix,
    and lam is a scalar. M is the output of Gamma + lam * (D.T @ Lam @ D), and can let
    Gamma just be 0 for the test.

    The actual calculation for D.T @ Lam @ D uses just the banded structure, which allows
    using arrays rather than having to use and update three separate sparse matrices (the
    full calculation is Gamma + D.T @ Lam @ D, where both Gamma and Lam are sparse matrices
    with one diagonal that gets updated each iteration), which is much faster and has no
    significant effect on memory.

    """
    full_shape = (num_points, num_points)  # the shape of the full matrices of A, B, and D.T*D
    A_banded, A, _, _, _ = misc._high_pass_filter(
        num_points, freq_cutoff, filter_type
    )
    x, y = get_data(True, num_points)
    lam_12_array = np.concatenate((
        np.full(num_points - 1, lam_1), np.full(num_points - 2, lam_2)
    ))
    diff_1_matrix = _banded_utils.difference_matrix(num_points, 1)
    diff_2_matrix = _banded_utils.difference_matrix(num_points, 2)
    d1_y = abs(np.diff(y))
    d2_y = abs(np.diff(y, 2))
    d_y = np.concatenate((d1_y, d2_y))
    diff_matrix = vstack((diff_1_matrix, diff_2_matrix))  # the full difference matrix, D

    # D.T @ diags(weight_function(derivative of y)) @ D,
    # let weight_function(d_y) just return d_y since it doesn't matter.
    # the calculation as written in the beads paper (see docstring of beads function for reference)
    true_calculation = (
        lam_1 * diff_1_matrix.T @ diags(d1_y) @ diff_1_matrix
        + lam_2 * diff_2_matrix.T @ diags(d2_y) @ diff_2_matrix
    )

    # the calculation as written in the MATLAB beads function, puts lam_1 and lam_2 within Lam
    matlab_calculation = diff_matrix.T @ diags(lam_12_array * d_y) @ diff_matrix

    assert_allclose(
        true_calculation.toarray(), matlab_calculation.toarray(), rtol=1e-15, atol=1e-15
    )

    # now do the same calculation, using the banded matrices
    diff_1_banded = np.zeros((5, num_points))
    diff_2_banded = np.zeros((5, num_points))
    # D.T @ L @ D == D_1.T @ L_1 @ D_1 + D_2.T @ L_2 @ D_2, so can calculate the
    # individual differences separately
    d1_y_output, d2_y_output = misc._abs_diff(y)
    diff_1_banded[1][1:] = diff_1_banded[3][:-1] = -d1_y_output
    diff_1_banded[2] = -(diff_1_banded[1] + diff_1_banded[3])

    diff_2_banded[0][2:] = diff_2_banded[-1][:-2] = d2_y_output
    diff_2_banded[1] = (
        2 * (diff_2_banded[0] - np.roll(diff_2_banded[0], -1, 0))
        - 4 * diff_2_banded[0]
    )
    diff_2_banded[-2][:-1] = diff_2_banded[1][1:]
    diff_2_banded[2] = -(
        diff_2_banded[0] + diff_2_banded[1] + diff_2_banded[-1] + diff_2_banded[-2]
    )

    banded_calculation = lam_1 * diff_1_banded + lam_2 * diff_2_banded

    assert_allclose(
        banded_calculation, _banded_utils._sparse_to_banded(matlab_calculation)[0],
        rtol=1e-15, atol=1e-15
    )

    # now test calculation of A.T @ M @ A where A is the D.T @ Lam @ D results
    ATMA_actual = A.T @ true_calculation @ A
    ATMA_actual_bands = _banded_utils._sparse_to_banded(ATMA_actual)[0]

    sparse_DTD = dia_object(
        (banded_calculation, np.arange(2, -3, -1)), shape=(num_points, num_points)
    )

    # for freq_cutoff ~ 0.5, values are very large, so have to raise rtol
    if freq_cutoff < 0.49:
        rtol = 1e-15
    else:
        rtol = 1e-10 if filter_type == 1 else 2e-12
    assert_allclose((A.T @ sparse_DTD @ A).toarray(), ATMA_actual.toarray(), rtol=rtol, atol=1e-15)
    # also check without transposing A since A is symmetric and that's what is used in pybaselines
    assert_allclose((A @ sparse_DTD @ A).toarray(), ATMA_actual.toarray(), rtol=rtol, atol=1e-15)

    # now check banded result; banded calculation also uses A instead of A.T
    ATMA_banded = misc._banded_dot_banded(
        misc._banded_dot_banded(
            A_banded, banded_calculation, (filter_type, filter_type), (2, 2),
            full_shape, full_shape
        ),
        A_banded, (filter_type + 2, filter_type + 2), (filter_type, filter_type),
        full_shape, full_shape
    )
    assert_allclose(ATMA_banded, ATMA_actual_bands, rtol=rtol, atol=1e-15)


@pytest.mark.parametrize('alpha', (1, 5.5))
def test_process_lams(data_fixture, alpha):
    """Ensures _process_lams correctly calculates lam values according to the L1 norms."""
    x, y = data_fixture
    lam_0_factor = alpha / np.linalg.norm(y, 1)
    lam_1_factor = alpha / np.linalg.norm(np.diff(y, 1), 1)
    lam_2_factor = alpha / np.linalg.norm(np.diff(y, 2), 1)
    lam_factors = [lam_0_factor, lam_1_factor, lam_2_factor]

    output = misc._process_lams(y, alpha, None, None, None)
    assert_allclose(output[0], lam_0_factor, rtol=1e-14, atol=1e-14)
    assert_allclose(output[1], lam_1_factor, rtol=1e-14, atol=1e-14)
    assert_allclose(output[2], lam_2_factor, rtol=1e-14, atol=1e-14)

    # only certain lam values missing
    for index in range(3):
        lams = [1, 1, 1]
        lams[index] = None
        output2 = misc._process_lams(y, alpha=alpha, lam_0=lams[0], lam_1=lams[1], lam_2=lams[2])
        for idx in range(3):
            if idx == index:
                assert_allclose(output2[idx], lam_factors[idx], rtol=1e-14, atol=1e-14)
            else:
                assert_allclose(output2[idx], lams[idx], rtol=1e-14, atol=1e-14)

    # ensure it respects when lam values are 0
    for index in range(3):
        lams = [1, 1, 1]
        lams[index] = 0
        output2 = misc._process_lams(y, alpha=alpha, lam_0=lams[0], lam_1=lams[1], lam_2=lams[2])
        for idx in range(3):
            if idx == index:
                assert_allclose(output2[idx], 0, rtol=1e-14, atol=1e-14)
            else:
                assert_allclose(output2[idx], lams[idx], rtol=1e-14, atol=1e-14)

    # all lams given, so they should just be checked and then output
    lam_0 = 5
    lam_1 = 10
    lam_2 = 3
    output3 = misc._process_lams(y, alpha, lam_0, lam_1, lam_2)
    assert_allclose(output3[0], lam_0, rtol=1e-14, atol=1e-14)
    assert_allclose(output3[1], lam_1, rtol=1e-14, atol=1e-14)
    assert_allclose(output3[2], lam_2, rtol=1e-14, atol=1e-14)


def test_process_lams_non_positive_alpha(data_fixture):
    """Ensures non-positive alpha values raise an error."""
    x, y = data_fixture
    with pytest.raises(ValueError):
        misc._process_lams(y, alpha=0, lam_0=1, lam_1=1, lam_2=1)
    with pytest.raises(ValueError):
        misc._process_lams(y, alpha=-1., lam_0=1, lam_1=1, lam_2=1)


def test_process_lams_all_zero(data_fixture):
    """Ensures an error is raised if all three lam values are zero."""
    x, y = data_fixture
    with pytest.raises(ValueError):
        misc._process_lams(y, alpha=0, lam_0=0, lam_1=0, lam_2=0)


@pytest.mark.parametrize('negative_lam', [0, 1, 2])
def test_process_lams_negative_lam_fails(data_fixture, negative_lam):
    """Ensures that a negative regularization parameter fails."""
    x, y = data_fixture
    lams = [1, 1, 1]
    lams[negative_lam] *= -1
    with pytest.raises(ValueError):
        misc._process_lams(y, alpha=1, lam_0=lams[0], lam_1=lams[1], lam_2=lams[2])
