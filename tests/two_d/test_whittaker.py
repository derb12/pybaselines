# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
import pytest

from pybaselines.two_d import whittaker
from pybaselines.utils import ParameterWarning

from ..conftest import BaseTester2D, InputWeightsMixin


class WhittakerTester(BaseTester2D, InputWeightsMixin):
    """Base testing class for whittaker functions."""

    module = whittaker
    algorithm_base = whittaker._Whittaker
    checked_keys = ('weights', 'tol_history')

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


class TestAirPLS(WhittakerTester):
    """Class for testing airpls baseline."""

    func_name = 'airpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e3, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    # ignore the RuntimeWarning that occurs from using +/- inf or nan
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_avoid_nonfinite_weights(self, no_noise_data_fixture2d):
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
        x, z, y = no_noise_data_fixture2d
        with pytest.warns(ParameterWarning):
            baseline = self.class_func(y, tol=-1, max_iter=3000)[0]

        assert np.isfinite(baseline.T.dot(baseline)).all()


class TestArPLS(WhittakerTester):
    """Class for testing arpls baseline."""

    func_name = 'arpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    def test_avoid_overflow_warning(self, no_noise_data_fixture2d):
        """
        Ensures no warning is emitted for exponential overflow.

        The weighting is 1 / (1 + exp(values)), so if values is too high,
        exp(values) is inf, which should usually emit an overflow warning.
        However, the resulting weight is 0, which is fine, so the warning is
        not needed and should be avoided. This test ensures the overflow warning
        is not emitted, and also ensures that the output is all finite, just in
        case the weighting was not actually stable.

        """
        x, z, y = no_noise_data_fixture2d
        with np.errstate(over='raise'):
            baseline = self.class_func(y, tol=-1, max_iter=1000)[0]

        assert np.isfinite(baseline.T.dot(baseline)).all()


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

    # ignore the RuntimeWarning that occurs from using +/- inf or nan
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_avoid_nonfinite_weights(self, no_noise_data_fixture2d):
        """
        Ensures that the function gracefully exits when non-finite weights are created.

        When there are no negative residuals or exp(iterations) / std is very high, both
        of which occur when a low tol value is used with a high max_iter value, the
        weighting function would produce non-finite values. The returned baseline should
        be the last iteration that was successful, and thus should not contain nan or +/- inf.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.
        Uses np.isfinite on the dot product of the baseline since the dot product is fast,
        would propogate the nan or inf, and will create only a single value to check
        for finite-ness.

        """
        x, z, y = no_noise_data_fixture2d
        with pytest.warns(ParameterWarning):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline.T.dot(baseline).all())
        # ensure last tolerence calculation was non-finite as a double-check that
        # this test is actually doing what it should be doing
        assert not np.isfinite(params['tol_history'][-1])


class TestIArPLS(WhittakerTester):
    """Class for testing iarpls baseline."""

    func_name = 'iarpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    # ignore the RuntimeWarning that occurs from using +/- inf or nan
    @pytest.mark.filterwarnings('ignore::RuntimeWarning')
    def test_avoid_nonfinite_weights(self, no_noise_data_fixture2d):
        """
        Ensures that the function gracefully exits when non-finite weights are created.

        When there are no negative residuals or exp(iterations) / std is very high, both
        of which occur when a low tol value is used with a high max_iter value, the
        weighting function would produce non-finite values. The returned baseline should
        be the last iteration that was successful, and thus should not contain nan or +/- inf.

        Use data without noise since the lack of noise makes it easier to induce failure.
        Set tol to -1 so that it is never reached, and set max_iter to a high value.
        Uses np.isfinite on the dot product of the baseline since the dot product is fast,
        would propogate the nan or inf, and will create only a single value to check
        for finite-ness.

        """
        x, z, y = no_noise_data_fixture2d
        with pytest.warns(ParameterWarning):
            baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline.T.dot(baseline)).all()
        # ensure last tolerence calculation was non-finite as a double-check that
        # this test is actually doing what it should be doing
        assert not np.isfinite(params['tol_history'][-1])


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

    @pytest.mark.parametrize('alpha_enum', (0, 1))
    def test_wrong_alpha_shape(self, alpha_enum):
        """Ensures that an exception is raised if input alpha and data are different shapes."""
        if alpha_enum == 0:
            alpha = np.ones(np.array(self.y.shape) + 1)
        else:
            alpha = np.ones(self.y.size)
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha=alpha)

    def test_avoid_overflow_warning(self, no_noise_data_fixture2d):
        """
        Ensures no warning is emitted for exponential overflow.

        The weighting is 1 / (1 + exp(values)), so if values is too high,
        exp(values) is inf, which should usually emit an overflow warning.
        However, the resulting weight is 0, which is fine, so the warning is
        not needed and should be avoided. This test ensures the overflow warning
        is not emitted, and also ensures that the output is all finite, just in
        case the weighting was not actually stable.

        """
        x, z, y = no_noise_data_fixture2d
        with np.errstate(over='raise'):
            baseline = self.class_func(y, tol=-1, max_iter=1000)[0]

        assert np.isfinite(baseline.T.dot(baseline)).all()


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
