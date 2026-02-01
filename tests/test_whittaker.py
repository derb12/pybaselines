# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.sparse.linalg import spsolve

from pybaselines import _banded_utils, _weighting, whittaker
from pybaselines.utils import relative_difference
from pybaselines._compat import diags, identity

from .base_tests import BaseTester, InputWeightsMixin, RecreationMixin, ensure_deprecation


def sparse_iasls(data, lam, p=1e-2, lam_1=1e-4, max_iter=50, tol=1e-3, diff_order=2):
    """A sparse version of iasls for testing that the banded version is implemented correctly."""
    y = np.asarray(data)
    num_y = len(y)
    weight_array = np.ones(num_y)
    d1_penalty = lam_1 * _banded_utils.diff_penalty_matrix(num_y, 1)
    penalty_matrix = lam * _banded_utils.diff_penalty_matrix(num_y, diff_order) + d1_penalty
    weight_matrix = diags(weight_array)
    for _ in range(max_iter + 1):
        lhs = weight_matrix.T @ weight_matrix + penalty_matrix
        baseline = spsolve(lhs, (weight_matrix.T @ weight_matrix + d1_penalty) @ y)
        new_weights = _weighting._asls(y, baseline, p)
        calc_difference = relative_difference(weight_array, new_weights)
        if calc_difference < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    return baseline


def sparse_drpls(data, lam, eta=0.5, diff_order=2, tol=1e-3, max_iter=50):
    """A sparse version of drpls for testing that the banded version is implemented correctly."""
    y = np.asarray(data)
    num_y = len(y)
    weight_array = np.ones(num_y)
    penalty_matrix = lam * _banded_utils.diff_penalty_matrix(num_y, diff_order)
    d1_penalty = _banded_utils.diff_penalty_matrix(num_y, 1)
    identity_matrix = identity(num_y)
    weight_matrix = diags(weight_array)
    for i in range(max_iter + 1):
        lhs = weight_matrix + d1_penalty + (identity_matrix - eta * weight_matrix) @ penalty_matrix
        baseline = spsolve(lhs, weight_array * y)
        new_weights, _ = _weighting._drpls(y, baseline, i + 1)
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)

    return baseline


def sparse_aspls(data, lam, diff_order=2, tol=1e-3, max_iter=100, asymmetric_coef=2.,
                 alternate_weighting=True):
    """A sparse version of aspls for testing that the banded version is implemented correctly."""
    y = np.asarray(data)
    num_y = len(y)
    weight_array = np.ones(num_y)

    penalty_matrix = lam * _banded_utils.diff_penalty_matrix(num_y, diff_order)
    alpha_matrix = identity(num_y)
    weight_matrix = diags(weight_array)
    for _ in range(max_iter + 1):
        lhs = weight_matrix + alpha_matrix @ penalty_matrix
        baseline = spsolve(lhs, weight_array * y)
        new_weights, residual, _ = _weighting._aspls(
            y, baseline, asymmetric_coef, alternate_weighting
        )
        if relative_difference(weight_array, new_weights) < tol:
            break
        weight_array = new_weights
        weight_matrix.setdiag(weight_array)
        abs_d = np.abs(residual)
        alpha_matrix.setdiag(abs_d / abs_d.max())

    return baseline


class WhittakerTester(BaseTester, InputWeightsMixin, RecreationMixin):
    """Base testing class for whittaker functions."""

    module = whittaker
    checked_keys = ('weights', 'tol_history')

    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_scipy_solvers(self, diff_order):
        """Ensure the two SciPy solvers give similar results."""
        original_solver = self.algorithm.banded_solver
        try:
            self.algorithm.banded_solver = 3  # use solveh_banded if allowed
            solveh_output = self.class_func(self.y, diff_order=diff_order)[0]
            self.algorithm.banded_solver = 4  # force use solve_banded
            solve_output = self.class_func(self.y, diff_order=diff_order)[0]

            assert_allclose(solveh_output, solve_output, rtol=1e-6, atol=1e-8)
        finally:
            self.algorithm.banded_solver = original_solver

    @pytest.mark.parametrize('pentadiagonal_solver', (1, 2))
    def test_pentadiagonal_solver(self, pentadiagonal_solver):
        """Ensure pentadiagonal solvers give similar result to SciPy's solvers."""
        original_solver = self.algorithm.banded_solver
        try:
            self.algorithm.banded_solver = pentadiagonal_solver
            # mock having numba so the solver is used even if numba is not installed
            with mock.patch.object(_banded_utils, '_HAS_NUMBA', True):
                pentadiagonal_output = self.class_func(self.y, diff_order=2)[0]

            self.algorithm.banded_solver = 3  # use solveh_banded if allowed
            solveh_output = self.class_func(self.y, diff_order=2)[0]
            self.algorithm.banded_solver = 4  # force use solve_banded
            solve_output = self.class_func(self.y, diff_order=2)[0]

            assert_allclose(pentadiagonal_output, solveh_output, rtol=1e-4, atol=1e-8)
            assert_allclose(pentadiagonal_output, solve_output, rtol=1e-4, atol=1e-8)
        finally:
            self.algorithm.banded_solver = original_solver

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


class TestAsLS(WhittakerTester):
    """Class for testing asls baseline."""

    func_name = 'asls'

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)


class TestIAsLS(WhittakerTester):
    """Class for testing iasls baseline."""

    func_name = 'iasls'

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {2: 1e6, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=1)

    @pytest.mark.parametrize('diff_order', (2, 3))
    @pytest.mark.parametrize('lam_1', (1e-3, 1e-1))
    @pytest.mark.parametrize('p', (0.1, 0.3))
    def test_sparse_comparison(self, diff_order, lam_1, p):
        """
        Ensures the banded version of the implementation is correct.

        Since iasls uses a more involved linear equation, ensure that the banded implementation
        matches a simpler sparse implementation.
        """
        max_iter = 100
        tol = 1e-3
        lam = {2: 1e6, 3: 1e10}[diff_order]
        sparse_output = sparse_iasls(
            self.y, lam=lam, lam_1=lam_1, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol
        )
        banded_output = self.class_func(
            self.y, lam=lam, lam_1=lam_1, p=p, diff_order=diff_order, max_iter=max_iter, tol=tol
        )[0]

        assert_allclose(banded_output, sparse_output, rtol=5e-4, atol=1e-8)


class TestAirPLS(WhittakerTester):
    """Class for testing airpls baseline."""

    func_name = 'airpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e3, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    # ignore the ParameterWarning that can occur from the fit not being good at high iterations
    @pytest.mark.filterwarnings('ignore::UserWarning')
    def test_avoid_nonfinite_weights(self, no_noise_data_fixture):
        """
        Ensures that the function gracefully exits when errors occur.

        When there are no negative residuals, which occurs when a low tol value is used with
        a high max_iter value, the weighting function would produce values all ~0, which
        can fail the solvers. The returned baseline should be the last iteration that was
        successful, and thus should not contain nan or +/- inf.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.
        Uses np.isfinite on the dot product of the baseline since the dot product is fast,
        would propogate the nan or inf, and will create only a single value to check
        for finite-ness.

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=3000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()


class TestArPLS(WhittakerTester):
    """Class for testing arpls baseline."""

    func_name = 'arpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_avoid_overflow_warning(self, no_noise_data_fixture):
        """
        Ensures no warning is emitted for exponential overflow.

        The weighting is 1 / (1 + exp(values)), so if values is too high,
        exp(values) is inf, which should usually emit an overflow warning.
        However, the resulting weight is 0, which is fine, so the warning is
        not needed and should be avoided. This test ensures the overflow warning
        is not emitted, and also ensures that the output is all finite, just in
        case the weighting was not actually stable.

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()


class TestDrPLS(WhittakerTester):
    """Class for testing drpls baseline."""

    func_name = 'drpls'

    @pytest.mark.parametrize('eta', (-1, 2))
    def test_outside_eta_fails(self, eta):
        """Ensures eta values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, eta=eta)

    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {2: 1e5, 3: 1e9}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=1)

    def test_avoid_nonfinite_weights(self, no_noise_data_fixture):
        """
        Ensures that the function does not create non-finite weights.

        drpls should not experience overflow since there is a cap on the iteration used
        within the exponential, so no warnings or errors should be emitted even when using
        a very high max_iter and low tol.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()

    @pytest.mark.parametrize('diff_order', (2, 3))
    @pytest.mark.parametrize('eta', (0.3, 0.5))
    def test_sparse_comparison(self, diff_order, eta):
        """
        Ensures the banded version of the implementation is correct.

        Since drpls uses a more involved linear equation, ensure that the banded implementation
        matches a simpler sparse implementation.
        """
        lam = {2: 1e5, 3: 1e9}[diff_order]
        max_iter = 100
        tol = 1e-3
        banded_output = self.class_func(
            self.y, lam=lam, eta=eta, diff_order=diff_order, max_iter=max_iter, tol=tol
        )[0]
        sparse_output = sparse_drpls(
            self.y, lam=lam, eta=eta, diff_order=diff_order, max_iter=max_iter, tol=tol
        )

        assert_allclose(banded_output, sparse_output, rtol=1.5e-6, atol=1e-10)


class TestIArPLS(WhittakerTester):
    """Class for testing iarpls baseline."""

    func_name = 'iarpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_avoid_nonfinite_weights(self, no_noise_data_fixture):
        """
        Ensures that the function does not create non-finite weights.

        iarpls should not experience overflow since there is a cap on the iteration used
        within the exponential, so no warnings or errors should be emitted even when using
        a very high max_iter and low tol.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.
        Uses np.isfinite on the dot product of the baseline since the dot product is fast,
        would propogate the nan or inf, and will create only a single value to check
        for finite-ness.

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()


class TestAsPLS(WhittakerTester):
    """Class for testing aspls baseline."""

    func_name = 'aspls'
    checked_keys = ('weights', 'alpha', 'tol_history')
    weight_keys = ('weights', 'alpha')

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_wrong_alpha_shape(self):
        """Ensures that an exception is raised if input alpha and data are different shapes."""
        alpha = np.ones(self.y.shape[0] + 1)
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha=alpha)

    def test_avoid_overflow_warning(self, no_noise_data_fixture):
        """
        Ensures no warning is emitted for exponential overflow.

        The weighting is 1 / (1 + exp(values)), so if values is too high,
        exp(values) is inf, which should usually emit an overflow warning.
        However, the resulting weight is 0, which is fine, so the warning is
        not needed and should be avoided. This test ensures the overflow warning
        is not emitted, and also ensures that the output is all finite, just in
        case the weighting was not actually stable.

        """
        y, _ = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline = self.class_func(y, tol=-1, max_iter=1000)[0]

        assert np.isfinite(baseline.dot(baseline))

    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_alpha_multiplication(self, diff_order):
        """Ensures multiplication of the alpha array and banded penalty is handled correctly."""
        lam = 5.
        num_points = len(self.y)
        alpha = np.arange(num_points, dtype=float)
        penalized_system = _banded_utils.PenalizedSystem(
            num_points, lam=lam, diff_order=diff_order, allow_lower=False, reverse_diags=True
        )
        penalty_matrix = lam * _banded_utils.diff_penalty_matrix(num_points, diff_order=diff_order)

        expected_result = _banded_utils._sparse_to_banded(diags(alpha) @ penalty_matrix)[0]

        result = alpha * penalized_system.penalty
        result = _banded_utils._shift_rows(result, diff_order, diff_order)
        assert_allclose(result, expected_result, rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize('asymmetric_coef', (0, -1))
    def test_outside_asymmetric_coef_fails(self, asymmetric_coef):
        """Ensures asymmetric_coef values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetric_coef=asymmetric_coef)

    @pytest.mark.parametrize('asymmetric_coef', (0.5, 2, 4))
    @pytest.mark.parametrize('alternate_weighting', (True, False))
    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_sparse_comparison(self, diff_order, asymmetric_coef, alternate_weighting):
        """
        Ensures the banded version of the implementation is correct.

        Since aspls uses a more involved linear equation, ensure that the banded implementation
        matches a simpler sparse implementation.
        """
        max_iter = 100
        tol = 1e-3
        lam = {2: 1e7, 3: 1e10}[diff_order]
        sparse_output = sparse_aspls(
            self.y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            asymmetric_coef=asymmetric_coef, alternate_weighting=alternate_weighting
        )
        banded_output = self.class_func(
            self.y, lam=lam, diff_order=diff_order, max_iter=max_iter, tol=tol,
            asymmetric_coef=asymmetric_coef, alternate_weighting=alternate_weighting
        )[0]

        rtol = {2: 1.5e-4, 3: 5e-4}[diff_order]
        assert_allclose(banded_output, sparse_output, rtol=rtol, atol=1e-8)


class TestPsalsa(WhittakerTester):
    """Class for testing psalsa baseline."""

    func_name = 'psalsa'

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestDerpsalsa(WhittakerTester):
    """Class for testing derpsalsa baseline."""

    func_name = 'derpsalsa'

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)

    @ensure_deprecation(1, 4)
    def test_kwargs_deprecation(self):
        """Ensure passing kwargs outside of the pad_kwargs keyword is deprecated."""
        with pytest.warns(DeprecationWarning):
            output, _ = self.class_func(self.y, mode='edge')
        output_2, _ = self.class_func(self.y, pad_kwargs={'mode': 'edge'})

        # ensure the outputs are still the same
        assert_allclose(output_2, output, rtol=1e-12, atol=1e-12)

        # also ensure both pad_kwargs and **kwargs are passed to pad_edges; derpsalsa does
        # the padding outside of setup_smooth, so have to do this to cover those cases
        with pytest.raises(TypeError):
            with pytest.warns(DeprecationWarning):
                self.class_func(self.y, pad_kwargs={'mode': 'extrapolate'}, mode='extrapolate')


class TestBrPLS(WhittakerTester):
    """Class for testing brpls baseline."""

    func_name = 'brpls'

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        max_iter_2 = 2
        _, params = self.class_func(
            self.y, max_iter=max_iter, max_iter_2=max_iter_2, tol=-1, tol_2=-1
        )

        assert params['tol_history'].size == (max_iter_2 + 2) * (max_iter + 1)
        assert params['tol_history'].shape == (max_iter_2 + 2, max_iter + 1)


class TestLSRPLS(WhittakerTester):
    """Class for testing lsrpls baseline."""

    func_name = 'lsrpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_avoid_nonfinite_weights(self, no_noise_data_fixture):
        """
        Ensures that the function does not create non-finite weights.

        lsrpls should not experience overflow since there is a cap on the iteration used
        within the exponential, so no warnings or errors should be emitted even when using
        a very high max_iter and low tol.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.
        Uses np.isfinite on the dot product of the baseline since the dot product is fast,
        would propogate the nan or inf, and will create only a single value to check
        for finite-ness.

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()
