# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import _banded_utils, whittaker
from pybaselines._compat import diags

from .conftest import BaseTester, InputWeightsMixin, has_pentapy


class WhittakerTester(BaseTester, InputWeightsMixin):
    """Base testing class for whittaker functions."""

    module = whittaker
    algorithm_base = whittaker._Whittaker
    checked_keys = ('weights', 'tol_history')

    @has_pentapy
    def test_pentapy_solver(self):
        """Ensure pentapy solver gives similar result to SciPy's solver."""
        with mock.patch.object(_banded_utils, '_HAS_PENTAPY', False):
            scipy_output = self.class_func(self.y)[0]
            assert not self.algorithm.whittaker_system.using_pentapy

        pentapy_output = self.class_func(self.y)[0]
        assert self.algorithm.whittaker_system.using_pentapy

        assert_allclose(pentapy_output, scipy_output, 1e-4)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1

    def test_recreation(self):
        """
        Ensures inputting weights can recreate the same baseline.

        Optimizers such as `collab_pls` require this functionality, so ensure
        it works.

        Note that if `max_iter` is set such that the function does not converge,
        then this will fail; that behavior is fine since exiting before convergence
        should not be a typical usage.
        """
        # TODO this should eventually be incorporated into InputWeightsMixin, but would
        # need to be generalized first; also most polynomial algorithms currently fail with
        # it due to their different exit criteria -> intended or not?
        first_baseline, params = self.class_func(self.y)
        kwargs = {'weights': params['weights']}
        if self.func_name in ('aspls', 'pspline_aspls'):
            kwargs['alpha'] = params['alpha']
        elif self.func_name in ('brpls', 'pspline_brpls'):
            kwargs['tol_2'] = np.inf
        second_baseline, params_2 = self.class_func(self.y, tol=np.inf, **kwargs)

        if self.func_name in ('brpls', 'pspline_brpls'):
            assert params_2['tol_history'].shape == (2, 1)
            assert params_2['tol_history'].size == 2
        else:
            assert len(params_2['tol_history']) == 1
        assert_allclose(second_baseline, first_baseline, rtol=1e-12)


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

        expected_result = _banded_utils._sparse_to_banded(
            diags(alpha) @ penalty_matrix, num_points
        )[0]

        result = alpha * penalized_system.penalty
        result = _banded_utils._shift_rows(result, diff_order, diff_order)
        assert_allclose(result, expected_result, rtol=1e-13, atol=1e-13)

    @pytest.mark.parametrize('asymmetric_coef', (0, -1))
    def test_outside_asymmetric_coef_fails(self, asymmetric_coef):
        """Ensures asymmetric_coef values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetric_coef=asymmetric_coef)


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
