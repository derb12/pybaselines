# -*- coding: utf-8 -*-
"""Tests for pybaselines.splines.

@author: Donald Erb
Created on March 20, 2021

"""

import inspect

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines.two_d import Baseline2D, spline, _spline_utils
from pybaselines._spline_utils import _basis_midpoints

from ..base_tests import (
    BaseTester2D, ConvergenceMixin, InputWeightsMixin, PSplineResult2DMixin, RecreationMixin
)


class WhittakerComparisonMixin:
    """Mixin for comparing penalized spline versions of Whittaker-smoothing algorithms."""

    def test_whittaker_comparison(self, lam=1e5, test_rtol=1e-8, **kwargs):
        """
        Compares the output of the penalized spline (P-spline) versions of Whittaker functions.

        The number of knots for the P-splines are set to ``data.shape`` and the
        spline degree is set to 1; the result is that the spline basis becomes the identity matrix
        with basis midpoints centered on each value, and the P-spline version should give the same
        output as the Whittaker version if the weighting and linear systems were correctly set up.

        """
        fitter = Baseline2D(self.x, self.z, check_finite=False, assume_sorted=True)
        whittaker_func = getattr(fitter, self.func_name.split('pspline_')[-1])
        if 'num_eigens' in inspect.signature(whittaker_func).parameters:
            added_kwargs = {'num_eigens': None}
        else:
            added_kwargs = {}

        whittaker_output, whittaker_params = whittaker_func(
            self.y, lam=lam, **kwargs, **added_kwargs
        )
        spline_output, spline_params = self.class_func(
            self.y, lam=lam, num_knots=self.y.shape, spline_degree=1, **kwargs
        )

        assert_allclose(spline_output, whittaker_output, rtol=test_rtol, atol=1e-10)
        assert_allclose(
            spline_params['weights'], whittaker_params['weights'], rtol=1e-10, atol=1e-6
        )
        if 'tol_history' in whittaker_params and whittaker_params['tol_history'].ndim == 1:
            assert whittaker_params['tol_history'].shape == spline_params['tol_history'].shape


def test_ensure_whittaker_comparison_setup():
    """Ensures the configuration within test_whittaker_comparison actually does what is expected.

    Checks that the spline basis is the identity matrix, and the midpoint of each spline
    basis is centered on the x- and z-values.

    """
    # simple case first as a sanity check
    x = np.array([1, 2, 3, 4], dtype=float)
    z = np.array([1, 2, 3, 4, 5], dtype=float)
    basis = _spline_utils.SplineBasis2D(x, z, (x.size, z.size), spline_degree=1)

    assert_allclose(basis.basis.toarray(), np.eye(x.size * z.size), rtol=1e-16, atol=1e-16)
    assert_allclose(basis.basis_r.toarray(), np.eye(x.size), rtol=1e-16, atol=1e-16)
    assert_allclose(basis.basis_c.toarray(), np.eye(z.size), rtol=1e-16, atol=1e-16)
    assert_allclose(basis.knots_r, [0, 1, 2, 3, 4, 5], rtol=1e-16, atol=1e-16)
    assert_allclose(basis.knots_c, [0, 1, 2, 3, 4, 5, 6], rtol=1e-16, atol=1e-16)
    assert_allclose(
        _basis_midpoints(basis.knots_r, spline_degree=1), x, rtol=1e-16, atol=1e-16
    )
    assert_allclose(
        _basis_midpoints(basis.knots_c, spline_degree=1), z, rtol=1e-16, atol=1e-16
    )

    x = np.linspace(1, 1500, 30)
    z = np.linspace(1, 1231, 21)
    dx = np.mean(np.diff(x))
    dz = np.mean(np.diff(z))
    basis = _spline_utils.SplineBasis2D(x, z, (x.size, z.size), spline_degree=1)

    assert_allclose(basis.basis.toarray(), np.eye(x.size * z.size), rtol=1e-16, atol=1e-16)
    assert_allclose(basis.basis_r.toarray(), np.eye(x.size), rtol=1e-16, atol=1e-16)
    assert_allclose(basis.basis_c.toarray(), np.eye(z.size), rtol=1e-16, atol=1e-16)
    expected_knots_r = np.concatenate(([x[0] - dx], x, [x[-1] + dx]))
    expected_knots_c = np.concatenate(([z[0] - dz], z, [z[-1] + dz]))
    assert_allclose(basis.knots_r, expected_knots_r, rtol=1e-16, atol=1e-16)
    assert_allclose(basis.knots_c, expected_knots_c, rtol=1e-16, atol=1e-16)
    assert_allclose(
        _basis_midpoints(basis.knots_r, spline_degree=1), x, rtol=1e-16, atol=1e-16
    )
    assert_allclose(
        _basis_midpoints(basis.knots_c, spline_degree=1), z, rtol=1e-16, atol=1e-16
    )


class SplineTester(BaseTester2D, PSplineResult2DMixin):
    """Base testing class for spline functions."""

    module = spline
    supports_mask = True

    @pytest.mark.parametrize('spline_degree', (None, (None, 1), (3, None), (None, None)))
    def test_check_spline_degree(self, spline_degree):
        """
        Ensures an exception is raised if the input spline_degree is None.

        For methods that are implemented as ND penalized least square mixins, the same
        code path is used for both penalized spline and Whittaker smoothing, whose logic
        is controlled by the input `spline_degree`, so need to ensure that the input for
        penalized spline methods is not None.

        For methods that are not PLS mixins, a TypeError should still occur during basis
        creation.

        """
        with pytest.raises(TypeError):
            self.class_func(self.y, spline_degree=spline_degree)


class IterativeSplineTester(SplineTester, InputWeightsMixin, RecreationMixin, ConvergenceMixin):
    """Base testing class for iterative spline functions."""

    checked_keys = ('weights', 'tol_history', 'result', 'success')

    @classmethod
    def setup_class(cls):
        """Increases default `tol` to reduce computation time."""
        super().setup_class()
        if 'tol' not in cls.repeated_kwargs:
            cls.repeated_kwargs['tol'] = 1e-1
        if 'tol' not in cls.kwargs:
            cls.kwargs['tol'] = 1e-1

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


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

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)


class TestIRSQR(IterativeSplineTester):
    """Class for testing irsqr baseline."""

    func_name = 'irsqr'
    required_repeated_kwargs = {'lam': 1e2}

    @pytest.mark.parametrize('quantile', (-1, 2))
    def test_outside_quantile_fails(self, quantile):
        """Ensures quantile values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('has_x', (True, False))
    @pytest.mark.parametrize('has_z', (True, False))
    def test_no_xz(self, has_x, has_z):
        """Ensures the output is not affected by not having x or z values."""
        super().test_no_xz(has_x, has_z, rtol=1e-5, atol=1e-4)


class TestPsplineAsLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_asls baseline."""

    func_name = 'pspline_asls'
    required_repeated_kwargs = {'lam': 1e0}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('p', (0.01, 0.2))
    def test_output_binary_weights(self, p):
        """Ensures all weights are either ``p`` or ``1 - p``."""
        _, params = self.class_func(self.y, p=p)
        weights = params['weights']
        assert (
            np.isclose(weights, p, atol=1e-15, rtol=0)
            | np.isclose(weights, 1 - p, atol=1e-15, rtol=0)
        ).all()


class TestPsplineIAsLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_iasls baseline."""

    func_name = 'pspline_iasls'
    required_repeated_kwargs = {'lam': 1e-2}

    @pytest.mark.parametrize('use_instance', (True, False))
    @pytest.mark.parametrize('weight_bool', (True, False))
    def test_unchanged_data(self, use_instance, weight_bool):
        """Ensures that input data is unchanged by the function."""
        if weight_bool:
            weights = np.ones_like(self.y)
        else:
            weights = None
        super().test_unchanged_data(use_instance, weights=weights)

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('p', (0.01, 0.2))
    def test_output_binary_weights(self, p):
        """Ensures all weights are either ``p**2`` or ``(1 - p)**2``."""
        _, params = self.class_func(self.y, p=p)
        weights = params['weights']
        assert (
            np.isclose(weights, p**2, atol=1e-15, rtol=0)
            | np.isclose(weights, (1 - p)**2, atol=1e-15, rtol=0)
        ).all()


class TestPsplineAirPLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_airpls baseline."""

    func_name = 'pspline_airpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplineArPLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_arpls baseline."""

    func_name = 'pspline_arpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplineIArPLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_iarpls baseline."""

    func_name = 'pspline_iarpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)


class TestPsplinePsalsa(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_psalsa baseline."""

    func_name = 'pspline_psalsa'
    required_repeated_kwargs = {'lam': 1e0}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestPsplineBrPLS(IterativeSplineTester, WhittakerComparisonMixin):
    """Class for testing pspline_brpls baseline."""

    func_name = 'pspline_brpls'
    # increase tol_2 to speed up tests
    required_kwargs = {'tol_2': 1e-1}
    required_repeated_kwargs = {'lam': 1e1, 'tol_2': 1e-1}

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
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
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order, **self.kwargs)

    @pytest.mark.parametrize('lam', (1e1, [1e1, 1e3]))
    @pytest.mark.parametrize('diff_order', (1, [2, 3]))
    def test_whittaker_comparison(self, lam, diff_order):
        """Ensures the P-spline version is the same as the Whittaker version."""
        super().test_whittaker_comparison(lam=lam, diff_order=diff_order)
