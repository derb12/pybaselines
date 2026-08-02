# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
import pytest

from pybaselines.two_d import whittaker

from ..base_tests import (
    BaseTester2D, ConvergenceMixin, InputWeightsMixin, RecreationMixin, WhittakerResult2DMixin,
    ensure_deprecation
)


class WhittakerTester(BaseTester2D, InputWeightsMixin, RecreationMixin, WhittakerResult2DMixin,
                      ConvergenceMixin):
    """Base testing class for whittaker functions."""

    module = whittaker
    checked_keys = ('weights', 'tol_history', 'result', 'success')
    supports_mask = True

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter + 1


class EigenvalueMixin:
    """BaseTester2D mixin for testing the Whittaker methods that can use eigendecomposition."""

    @pytest.mark.parametrize('return_dof', (True, False))
    def test_dof_output(self, return_dof):
        """Ensures the degrees of freedom are output if `return_dof` is True."""
        if return_dof:
            additional_keys = ['dof']
        else:
            additional_keys = None
        self.test_output(additional_keys=additional_keys, return_dof=return_dof)

    def test_dof_shape(self):
        """Ensures the returned degrees of freedom are correct."""
        num_eigens = (5, 10)
        baseline, params = self.class_func(
            data=self.y, num_eigens=num_eigens, **self.kwargs, return_dof=True
        )

        assert 'dof' in params
        assert params['dof'].shape == num_eigens

    @pytest.mark.threaded_test
    @pytest.mark.parametrize('num_eigens', (10, None))
    def test_threading(self, num_eigens):
        """Tests thread safety using SVD solver and analytical solution."""
        # set tol to higher values to reduce overall computation time
        super().test_threading(num_eigens=num_eigens, tol=1e-1)


class TestAsLS(EigenvalueMixin, WhittakerTester):
    """Class for testing asls baseline."""

    func_name = 'asls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    @pytest.mark.parametrize('p', (0.01, 0.2))
    def test_output_binary_weights(self, p):
        """Ensures all weights are either ``p`` or ``1 - p``."""
        _, params = self.class_func(self.y, p=p)
        weights = params['weights']
        assert (
            np.isclose(weights, p, atol=1e-15, rtol=0)
            | np.isclose(weights, 1 - p, atol=1e-15, rtol=0)
        ).all()


class TestIAsLS(WhittakerTester):
    """Class for testing iasls baseline."""

    func_name = 'iasls'
    required_repeated_kwargs = {'lam': 1e-1, 'tol': 1e-1}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    @pytest.mark.parametrize('p', (0.01, 0.2))
    def test_output_binary_weights(self, p):
        """Ensures all weights are either ``p**2`` or ``(1 - p)**2``."""
        _, params = self.class_func(self.y, p=p)
        weights = params['weights']
        assert (
            np.isclose(weights, p**2, atol=1e-15, rtol=0)
            | np.isclose(weights, (1 - p)**2, atol=1e-15, rtol=0)
        ).all()


class TestAirPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing airpls baseline."""

    func_name = 'airpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    @ensure_deprecation(1, 5)
    @pytest.mark.parametrize('normalize_weights', (True, False))
    def test_normalize_weights_deprecation(self, normalize_weights):
        """Ensures warning is emitted if normalize_weights is input."""
        with pytest.warns(DeprecationWarning, match='normalize_weights is deprecated'):
            self.class_func(self.y, normalize_weights=normalize_weights)


class TestArPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing arpls baseline."""

    func_name = 'arpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestDrPLS(WhittakerTester):
    """Class for testing drpls baseline."""

    func_name = 'drpls'
    required_repeated_kwargs = {'lam': 1e1, 'tol': 1e-1}

    @pytest.mark.parametrize('eta', (-1, 2))
    def test_outside_eta_fails(self, eta):
        """Ensures eta values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, eta=eta)

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestIArPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing iarpls baseline."""

    func_name = 'iarpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestAsPLS(WhittakerTester):
    """Class for testing aspls baseline."""

    func_name = 'aspls'
    checked_keys = ('weights', 'alpha', 'tol_history', 'result', 'success')
    weight_keys = ('weights', 'alpha')
    required_repeated_kwargs = {'lam': 1e2, 'tol': 1e-1}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    @pytest.mark.parametrize('alpha_enum', (0, 1))
    def test_wrong_alpha_shape(self, alpha_enum):
        """Ensures that an exception is raised if input alpha and data are different shapes."""
        if alpha_enum == 0:
            alpha = np.ones(np.array(self.y.shape) + 1)
        else:
            alpha = np.ones(self.y.size)
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha=alpha)

    @pytest.mark.parametrize('asymmetric_coef', (0, -1))
    def test_outside_asymmetric_coef_fails(self, asymmetric_coef):
        """Ensures asymmetric_coef values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, asymmetric_coef=asymmetric_coef)


class TestPsalsa(EigenvalueMixin, WhittakerTester):
    """Class for testing psalsa baseline."""

    func_name = 'psalsa'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    @pytest.mark.parametrize('k', (0, -1))
    def test_outside_k_fails(self, k):
        """Ensures k values not greater than 0 raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, k=k)


class TestBrPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing brpls baseline."""

    func_name = 'brpls'
    required_repeated_kwargs = {'lam': 1e2, 'tol_2': 1e-1}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        max_iter_2 = 2
        _, params = self.class_func(
            self.y, max_iter=max_iter, max_iter_2=max_iter_2, tol=-1, tol_2=-1
        )

        assert params['tol_history'].size == (max_iter_2 + 2) * (max_iter + 1)
        assert params['tol_history'].shape == (max_iter_2 + 2, max_iter + 1)


class TestLSRPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing lsrpls baseline."""

    func_name = 'lsrpls'
    required_repeated_kwargs = {'lam': 1e2}

    @pytest.mark.parametrize('diff_order', (1, [1, 3]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)
