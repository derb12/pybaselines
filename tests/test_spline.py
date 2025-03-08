# -*- coding: utf-8 -*-
"""Tests for pybaselines.splines.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import _banded_utils, _spline_utils, morphological, spline, whittaker

from .conftest import BaseTester, InputWeightsMixin, ensure_deprecation


def compare_pspline_whittaker(pspline_class, whittaker_func, data, lam=1e5,
                              test_rtol=1e-6, test_atol=1e-12, **kwargs):
    """
    Compares the output of the penalized spline (P-spline) versions of Whittaker functions.

    The number of knots for the P-splines are set to ``len(data) + 1`` and the spline
    degree is set to 0; the result is that the spline basis becomes the identity matrix,
    and the P-spline version should give the same output as the Whittaker version if
    the weighting and linear systems were correctly set up.

    """
    # ensure the Whittaker functions use Scipy since that is what P-splines use
    with mock.patch.object(_banded_utils, '_HAS_PENTAPY', False):
        whittaker_output = whittaker_func(data, lam=lam, **kwargs)[0]

    if hasattr(pspline_class, 'class_func'):
        spline_output = pspline_class.class_func(
            data, lam=lam, num_knots=len(data) + 1, spline_degree=0, **kwargs
        )[0]
    else:
        spline_output = pspline_class._call_func(
            data, lam=lam, num_knots=len(data) + 1, spline_degree=0, **kwargs
        )[0]

    assert_allclose(spline_output, whittaker_output, rtol=test_rtol, atol=test_atol)


class SplineTester(BaseTester):
    """Base testing class for spline functions."""

    module = spline
    algorithm_base = spline._Spline

    def test_numba_implementation(self):
        """
        Ensures the output is consistent between the two separate pspline solvers.

        Some testing subclasses do not use pspline solvers, but this is the easiest way
        to cover all affect algorithms.

        """
        with mock.patch.object(_spline_utils, '_HAS_NUMBA', False):
            normal_output = self.class_func(self.y)[0]
        with mock.patch.object(_spline_utils, '_HAS_NUMBA', True):
            numba_output = self.class_func(self.y)[0]

        assert_allclose(numba_output, normal_output, rtol=1e-9)


class IterativeSplineTester(SplineTester, InputWeightsMixin):
    """Base testing class for iterative spline functions."""

    checked_keys = ('weights', 'tol_history')

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
        # TODO this should eventually be incorporated into InputWeightsMixin
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


class TestMixtureModel(IterativeSplineTester):
    """Class for testing mixture_model baseline."""

    func_name = 'mixture_model'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('weight_bool', (True, False))
    def test_unchanged_data(self, use_class, weight_bool):
        """Ensures that input data is unchanged by the function."""
        if weight_bool:
            weights = np.ones_like(self.y)
        else:
            weights = None
        super().test_unchanged_data(use_class, weights=weights)

    @pytest.mark.parametrize('symmetric', (False, True))
    def test_output(self, symmetric):
        """Ensures that the output has the desired format."""
        initial_y = self.y
        try:
            if symmetric:
                # make data with both positive and negative peaks; roll so peaks are not overlapping
                self.y = np.roll(self.y, -50) - np.roll(self.y, 50)
                p = 0.5
            else:
                p = 0.01
            super().test_output(p=p, symmetric=symmetric)
        finally:
            self.y = initial_y

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @ensure_deprecation(1, 3)
    def test_num_bins_deprecation(self):
        """Ensures a DeprecationWarning is given when num_bins is input."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, num_bins=20)


class TestIRSQR(IterativeSplineTester):
    """Class for testing irsqr baseline."""

    func_name = 'irsqr'

    @pytest.mark.parametrize('quantile', (-1, 2))
    def test_outside_p_fails(self, quantile):
        """Ensures quantile values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 2: 1e5, 3: 1e8}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)


class TestCornerCutting(SplineTester):
    """
    Class for testing corner_cutting baseline.

    Has lower tolerance values for some tests since it is not currently perfectly repeatable.

    """

    func_name = 'corner_cutting'

    def test_no_x(self):
        """Ensures that function output is similar when no x is input."""
        super().test_no_x(rtol=1e-3)

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        super().test_list_input(rtol=1e-5)


class TestPsplineAsLS(IterativeSplineTester):
    """Class for testing pspline_asls baseline."""

    func_name = 'pspline_asls'

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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('p', (0.01, 0.1))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, p, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(
            self, whittaker.asls, self.y, lam=lam, p=p, diff_order=diff_order
        )


class TestPsplineIAsLS(IterativeSplineTester):
    """Class for testing pspline_iasls baseline."""

    func_name = 'pspline_iasls'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('weight_bool', (True, False))
    def test_unchanged_data(self, use_class, weight_bool):
        """Ensures that input data is unchanged by the function."""
        if weight_bool:
            weights = np.ones_like(self.y)
        else:
            weights = None
        super().test_unchanged_data(use_class, weights=weights)

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, diff_order=1)

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('p', (0.01, 0.1))
    @pytest.mark.parametrize('diff_order', (2, 3))
    @pytest.mark.parametrize('lam_1', (1e1, 1e3))
    def test_whittaker_comparison(self, lam, lam_1, p, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(
            self, whittaker.iasls, self.y, lam=lam, lam_1=lam_1, p=p, diff_order=diff_order
        )


class TestPsplineAirPLS(IterativeSplineTester):
    """Class for testing pspline_airpls baseline."""

    func_name = 'pspline_airpls'

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

        """
        x, y = no_noise_data_fixture
        with np.errstate(over='raise'):
            baseline, params = self.class_func(y, tol=-1, max_iter=7000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.airpls, self.y, lam=lam, diff_order=diff_order)


class TestPsplineArPLS(IterativeSplineTester):
    """Class for testing pspline_arpls baseline."""

    func_name = 'pspline_arpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    # ignore the ParameterWarning that can occur from the fit not being good at high iterations
    # only relevant for earlier SciPy versions, so maybe due to change within scipy.special.expit
    @pytest.mark.filterwarnings('ignore::UserWarning')
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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.arpls, self.y, lam=lam, diff_order=diff_order)


class TestPsplineDrPLS(IterativeSplineTester):
    """Class for testing pspline_drpls baseline."""

    func_name = 'pspline_drpls'

    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {2: 1e6, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

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
        baseline, params = self.class_func(y, tol=-1, max_iter=1000)

        assert np.isfinite(baseline).all()
        # ensure last tolerence calculation was finite as a double-check that
        # this test is actually doing what it should be doing
        assert np.isfinite(params['tol_history'][-1])
        assert np.isfinite(params['weights']).all()

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('eta', (0.2, 0.8))
    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_whittaker_comparison(self, lam, eta, diff_order):
        """
        Ensures the P-spline version is the same as the Whittaker version.

        Have to use a larger tolerance since pspline_drpls uses interpolation to
        get the weight at the coefficients' x-values.
        """
        compare_pspline_whittaker(
            self, whittaker.drpls, self.y, lam=lam, eta=eta, diff_order=diff_order, test_rtol=5e-3
        )

    @pytest.mark.parametrize('eta', (-1, 2))
    def test_outside_eta_fails(self, eta):
        """Ensures eta values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, eta=eta)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, diff_order=1)


class TestPsplineIArPLS(IterativeSplineTester):
    """Class for testing pspline_iarpls baseline."""

    func_name = 'pspline_iarpls'

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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.iarpls, self.y, lam=lam, diff_order=diff_order)


class TestPsplineAsPLS(IterativeSplineTester):
    """Class for testing pspline_aspls baseline."""

    func_name = 'pspline_aspls'
    checked_keys = ('weights', 'tol_history', 'alpha')
    weight_keys = ('weights', 'alpha')

    def test_wrong_alpha_shape(self):
        """Ensures that an exception is raised if input alpha and data are different shapes."""
        alpha = np.ones(self.y.shape[0] + 1)
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha=alpha)

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
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
            baseline = self.class_func(y, tol=-1, max_iter=1000)[0]

        assert np.isfinite(baseline.dot(baseline))

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """
        Ensures the P-spline version is the same as the Whittaker version.

        Have to use a larger tolerance since pspline_aspls uses interpolation to
        get the alpha values at the coefficients' x-values.
        """
        if diff_order == 2:
            rtol = 2e-3
        else:
            rtol = 5e-2
        compare_pspline_whittaker(
            self, whittaker.aspls, self.y, lam=lam, diff_order=diff_order, test_rtol=rtol
        )

    @pytest.mark.parametrize('asymmetric_coef', (0, -1))
    def test_outside_asymmetric_coef_fails(self, asymmetric_coef):
        """Ensures asymmetric_coef values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetric_coef=asymmetric_coef)


class TestPsplinePsalsa(IterativeSplineTester):
    """Class for testing pspline_psalsa baseline."""

    func_name = 'pspline_psalsa'

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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('p', (0.01, 0.1))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, p, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(
            self, whittaker.psalsa, self.y, lam=lam, p=p, diff_order=diff_order
        )

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestPsplineDerpsalsa(IterativeSplineTester):
    """Class for testing pspline_derpsalsa baseline."""

    func_name = 'pspline_derpsalsa'

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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('p', (0.01, 0.1))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, p, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(
            self, whittaker.derpsalsa, self.y, lam=lam, p=p, diff_order=diff_order
        )

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestPsplineMPLS(SplineTester, InputWeightsMixin):
    """Class for testing pspline_mpls baseline."""

    func_name = 'pspline_mpls'
    checked_keys = ('half_window', 'weights')

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e4, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('half_window', (4, 15, 30))
    def test_mpls_weights(self, half_window):
        """
        Ensure that the assigned weights are the same as the MPLS method.

        The assigned weights are not dependent on the least-squared fitting parameters,
        only on the half window.
        """
        _, params = self.class_func(self.y, half_window=half_window)
        _, mpls_params = morphological.mpls(self.y, half_window=half_window)

        assert_allclose(params['weights'], mpls_params['weights'], rtol=1e-9)

    @ensure_deprecation(1, 4)
    def test_tol_deprecation(self):
        """Ensures a DeprecationWarning is given when tol is input."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, tol=1e-3)

    @ensure_deprecation(1, 4)
    def test_max_iter_deprecation(self):
        """Ensures a DeprecationWarning is given when max_iter is input."""
        with pytest.warns(DeprecationWarning):
            self.class_func(self.y, max_iter=20)


class TestPsplineBrPLS(IterativeSplineTester):
    """Class for testing pspline_brpls baseline."""

    func_name = 'pspline_brpls'

    @pytest.mark.parametrize('diff_order', (1, 3))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        lam = {1: 1e2, 3: 1e10}[diff_order]
        self.class_func(self.y, lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.brpls, self.y, lam=lam, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        max_iter_2 = 2
        _, params = self.class_func(
            self.y, max_iter=max_iter, max_iter_2=max_iter_2, tol=-1, tol_2=-1
        )

        assert params['tol_history'].size == (max_iter_2 + 2) * (max_iter + 1)
        assert params['tol_history'].shape == (max_iter_2 + 2, max_iter + 1)


class TestPsplineLSRPLS(IterativeSplineTester):
    """Class for testing pspline_lsrpls baseline."""

    func_name = 'pspline_lsrpls'

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

    @pytest.mark.parametrize('lam', (1e1, 1e5))
    @pytest.mark.parametrize('diff_order', (1, 2, 3))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        compare_pspline_whittaker(self, whittaker.lsrpls, self.y, lam=lam, diff_order=diff_order)
