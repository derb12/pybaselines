# -*- coding: utf-8 -*-
"""Tests for pybaselines.splines.

@author: Donald Erb
Created on March 20, 2021

"""

from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines import _spline_utils, morphological, spline, Baseline

from .base_tests import BaseTester, InputWeightsMixin, RecreationMixin, ensure_deprecation


class WhittakerComparisonMixin:
    """Mixin for comparing penalized spline versions of Whittaker-smoothing algorithms."""

    def test_whittaker_comparison(self, lam=1e5, test_rtol=1e-6, test_atol=1e-12, **kwargs):
        """
        Compares the output of the penalized spline (P-spline) versions of Whittaker functions.

        The number of knots for the P-splines are set to ``len(data) + 1`` and the spline
        degree is set to 0; the result is that the spline basis becomes the identity matrix,
        and the P-spline version should give the same output as the Whittaker version if
        the weighting and linear systems were correctly set up.

        """
        fitter = Baseline(self.x, check_finite=False, assume_sorted=True)
        # ensure the Whittaker functions use Scipy since that is what P-splines use
        fitter.banded_solver = 3
        whittaker_func = getattr(fitter, self.func_name.split('pspline_')[-1])

        whittaker_output = whittaker_func(self.y, lam=lam, **kwargs)[0]
        spline_output = self.class_func(
            self.y, lam=lam, num_knots=len(self.y) + 1, spline_degree=0, **kwargs
        )[0]

        assert_allclose(spline_output, whittaker_output, rtol=test_rtol, atol=test_atol)


class SplineTester(BaseTester):
    """Base testing class for spline functions."""

    module = spline

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

        assert_allclose(numba_output, normal_output, rtol=1e-10, atol=1e-10)


class IterativeSplineTester(SplineTester, InputWeightsMixin, RecreationMixin):
    """Base testing class for iterative spline functions."""

    checked_keys = ('weights', 'tol_history')

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1

    @pytest.mark.parametrize('spline_degree', (1, 2, 3))
    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_numba_implementation(self, diff_order, spline_degree, **kwargs):
        """
        Ensures the output is consistent between the two separate P-Spline setup pathways.

        The calculation of ``B.T @ W @ B`` and ``B.T @ W @ y`` is done using sparse matrices or
        arrays if Numba is not installed; otherwise, a faster method is used. While simple
        cases are tested within test_spline_utils, this test is a more useful smoke test for
        any complex issues that could arise.

        """
        with mock.patch.object(_spline_utils, '_HAS_NUMBA', False):
            sparse_output = self.class_func(
                self.y, diff_order=diff_order, spline_degree=spline_degree, **kwargs
            )[0]
        with mock.patch.object(_spline_utils, '_HAS_NUMBA', True):
            numba_output = self.class_func(
                self.y, diff_order=diff_order, spline_degree=spline_degree, **kwargs
            )[0]

        assert_allclose(numba_output, sparse_output, rtol=1e-10, atol=1e-10)


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
    requires_unique_x = True

    def test_no_x(self):
        """Ensures that function output is similar when no x is input."""
        super().test_no_x(rtol=1e-3)

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        super().test_list_input(rtol=1e-5)


class TestPsplineAsLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, p=p, diff_order=diff_order)


class TestPsplineIAsLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, lam_1=lam_1, p=p, diff_order=diff_order)


class TestPsplineAirPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplineArPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplineDrPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, eta=eta, diff_order=diff_order, test_rtol=5e-3)

    @pytest.mark.parametrize('eta', (-1, 2))
    def test_outside_eta_fails(self, eta):
        """Ensures eta values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, eta=eta)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, diff_order=1)


class TestPsplineIArPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplineAsPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
    @pytest.mark.parametrize('alternate_weighting', (True, False))
    def test_whittaker_comparison(self, lam, diff_order, alternate_weighting):
        """
        Ensures the P-spline version is the same as the Whittaker version.

        Have to use a larger tolerance since pspline_aspls uses interpolation to
        get the alpha values at the coefficients' x-values.
        """
        if diff_order == 2:
            rtol = 3e-3
        else:
            rtol = 5e-2
        if alternate_weighting:
            asymmetric_coef = 2.
        else:
            asymmetric_coef = 0.5
        super().test_whittaker_comparison(
            lam=lam, diff_order=diff_order, alternate_weighting=alternate_weighting,
            asymmetric_coef=asymmetric_coef, test_rtol=rtol
        )

    @pytest.mark.parametrize('asymmetric_coef', (0, -1))
    def test_outside_asymmetric_coef_fails(self, asymmetric_coef):
        """Ensures asymmetric_coef values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetric_coef=asymmetric_coef)

    @ensure_deprecation(1, 3)  # revisit this once the aspls weighting situation is looked at
    @pytest.mark.parametrize('spline_degree', (1, 2, 3))
    @pytest.mark.parametrize('diff_order', (2, 3))
    def test_numba_implementation(self, diff_order, spline_degree):
        """
        Runs the numba vs sparse comparison using a non-default asymmetric_coef value.

        The weighting for aspls when asymmetric_coef=2 seems to be a bit sensitive, so likely
        any small floating point differences in the B.T @ W @ B and B.T @ W @ y calculation after
        several iterations leads to slightly different results (the test needs an rtol of ~1e-5 to
        pass). Increasing asymmetric_coef or setting max_iter to ~20 both fix this, so the
        divergence arises from the aspls weighting and not within the pspline solver. To avoid
        this, the tolerance for the method is set to 1e-2 to exit early enough that these floating
        point issues do not influence the output.
        """
        super().test_numba_implementation(diff_order, spline_degree, tol=1e-2)


class TestPsplinePsalsa(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, p=p, diff_order=diff_order)

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestPsplineDerpsalsa(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, p=p, diff_order=diff_order)

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

        # also ensure both pad_kwargs and **kwargs are passed to pad_edges; pspline_derpsalsa does
        # the padding outside of setup_smooth, so have to do this to cover those cases
        with pytest.raises(TypeError):
            with pytest.warns(DeprecationWarning):
                self.class_func(self.y, pad_kwargs={'mode': 'extrapolate'}, mode='extrapolate')


@pytest.mark.filterwarnings('ignore:"pspline_mpls" is deprecated')
class TestPsplineMPLS(SplineTester, InputWeightsMixin, WhittakerComparisonMixin):
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

    @ensure_deprecation(1, 5)
    def test_method_deprecation(self):
        """Ensures the deprecation warning is emitted if this method is used."""
        with pytest.warns(DeprecationWarning):
            self.class_func(data=self.y)


class TestPsplineBrPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        max_iter_2 = 2
        _, params = self.class_func(
            self.y, max_iter=max_iter, max_iter_2=max_iter_2, tol=-1, tol_2=-1
        )

        assert params['tol_history'].size == (max_iter_2 + 2) * (max_iter + 1)
        assert params['tol_history'].shape == (max_iter_2 + 2, max_iter + 1)


class TestPsplineLSRPLS(IterativeSplineTester, WhittakerComparisonMixin):
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
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)
