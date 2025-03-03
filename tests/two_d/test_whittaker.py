# -*- coding: utf-8 -*-
"""Tests for pybaselines.whittaker.

@author: Donald Erb
Created on March 20, 2021

"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from pybaselines.two_d import whittaker

from ..conftest import BaseTester2D, InputWeightsMixin, skipping_threading_tests


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

    @skipping_threading_tests
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

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestIAsLS(WhittakerTester):
    """Class for testing iasls baseline."""

    func_name = 'iasls'
    required_repeated_kwargs = {'lam': 1e-1, 'tol': 1e-1}

    @pytest.mark.parametrize('p', (-1, 2))
    def test_outside_p_fails(self, p):
        """Ensures p values outside of [0, 1] raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, p=p)

    @pytest.mark.parametrize('diff_order', (2, [3, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=1)
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[1, 1])
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[1, 2])
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[2, 1])


class TestAirPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing airpls baseline."""

    func_name = 'airpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestArPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing arpls baseline."""

    func_name = 'arpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
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

    @pytest.mark.parametrize('diff_order', (2, [3, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)

    def test_diff_order_one_fails(self):
        """Ensure that a difference order of 1 raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=1)
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[1, 1])
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[1, 2])
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=1e2, diff_order=[2, 1])


class TestIArPLS(EigenvalueMixin, WhittakerTester):
    """Class for testing iarpls baseline."""

    func_name = 'iarpls'
    required_repeated_kwargs = {'lam': 1e1}

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)


class TestAsPLS(WhittakerTester):
    """Class for testing aspls baseline."""

    func_name = 'aspls'
    checked_keys = ('weights', 'alpha', 'tol_history')
    weight_keys = ('weights', 'alpha')
    required_repeated_kwargs = {'lam': 1e2, 'tol': 1e-1}

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
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

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
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

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
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

    @pytest.mark.parametrize('diff_order', (1, [1, 2]))
    def test_diff_orders(self, diff_order):
        """Ensure that other difference orders work."""
        self.class_func(self.y, diff_order=diff_order)
