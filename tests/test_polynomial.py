# -*- coding: utf-8 -*-
"""Tests for pybaselines.polynomial.

@author: Donald Erb
Created on March 20, 2021

"""

from math import ceil
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy import stats

from pybaselines import polynomial
from pybaselines.utils import ParameterWarning

from .base_tests import (
    BasePolyTester, InputWeightsMixin, WeightMaskingMixin, RecreationMixin, ensure_deprecation
)
from .data import (
    LOESS_X, LOESS_Y, QUANTILE_Y, STATSMODELS_LOESS_DELTA, STATSMODELS_LOESS_ITER,
    STATSMODELS_QUANTILES
)


class PolynomialTester(BasePolyTester, InputWeightsMixin):
    """Base testing class for polynomial functions."""

    module = polynomial
    checked_keys = ('weights',)
    supports_mask = True


class IterativePolynomialTester(PolynomialTester):
    """Base testing class for iterative polynomial functions."""

    checked_keys = ('weights', 'tol_history')
    allows_zero_iteration = True  # whether max_iter=0 will return an initial baseline

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        if self.allows_zero_iteration:
            assert params['tol_history'].size == max_iter
        else:
            assert params['tol_history'].size == max_iter + 1


@pytest.mark.filterwarnings('ignore:"poly" is deprecated and will be removed in version 1.5.')
class TestPoly(PolynomialTester, WeightMaskingMixin):
    """Class for testing regular polynomial baseline."""

    func_name = 'poly'

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    @pytest.mark.parametrize('use_weights', (True, False))
    def test_implementation(self, poly_order, use_weights):
        """Ensures expected functionality by comparing to a more basic implementation.

        Note that Numpy polynomials are defined as minimizing ``(w * (y - y_fit))^2`` whereas
        pybaselines uses ``w * (y - y_fit))^2``.
        """
        if use_weights is None:
            weights = np.ones_like(self.y)
        else:
            weights = np.random.default_rng(123).uniform(0, 1, self.y.shape)
        fit, params = self.class_func(self.y, poly_order=poly_order, weights=weights)
        numpy_fit = np.polynomial.Polynomial.fit(
            self.x, self.y, deg=poly_order, w=np.sqrt(weights)
        )(self.x)

        assert_allclose(fit, numpy_fit, rtol=1e-10, atol=1e-12)

    @ensure_deprecation(1, 5)
    def test_method_deprecation(self):
        """Ensures the deprecation warning is emitted if this method is used."""
        with pytest.warns(DeprecationWarning, match='"poly" is deprecated'):
            self.class_func(data=self.y)


def thresholding_polynomial(x, y, poly_order, max_iter, weights=None, use_original=False,
                            num_std=0.):
    """
    A simple implementation of an iteratively thresholding polynomial fit.

    Parameters
    ----------
    x : array-like, shape (N,)
        The x-values for fitting.
    y : array-like, shape (N,)
        The y-values for fitting.
    poly_order : int
        The degree of the polynomial to fit.
    max_iter : int
        The number of iterations to perform thresholding.
    weights : array-like, shape (N,), optional
        If supplied, will use the square root of the input weights for fitting, as described
        in the Notes section below. Default is None, which weighs all points equally.
    use_original : bool, optional
        If True, will use the originally input ``y`` during thresholding. If False (default),
        will use the current iteration's ``y`` value.
    num_std : float, optional
        The number of standard deviations of the residual to include during thresholding.
        Default is 0.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The fit baseline after iterative thresholding.

    Notes
    -----
    Does not use an exit criteria so that it can be used for both ``modpoly`` and ``imodpoly``,
    which use different exit criteria, so must instead be supplied with the number of iterations
    used by the underlying method to validate.

    Applies a square root to the input weights since Numpy polynomials are defined as minimizing
    ``(w * (y - y_fit))^2`` whereas pybaselines uses ``w * (y - y_fit)^2``.

    References
    ----------
    Gan, F., et al. Baseline correction by improved iterative polynomial
    fitting with automatic threshold. Chemometrics and Intelligent
    Laboratory Systems, 2006, 82, 59-65.

    Lieber, C., et al. Automated method for subtraction of fluorescence
    from biological raman spectra. Applied Spectroscopy, 2003, 57(11),
    1363-1367.

    Zhao, J., et al. Automated Autofluorescence Background Subtraction
    Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy,
    2007, 61(11), 1225-1232.

    """
    if weights is None:
        sqrt_weights = np.ones_like(y)
    else:
        sqrt_weights = np.sqrt(weights)
    y_original = y.copy()
    baseline = np.polynomial.Polynomial.fit(x=x, y=y, deg=poly_order, w=sqrt_weights)(x)
    for _ in range(max_iter):
        y = np.minimum(
            baseline + num_std * np.std((y - baseline)[sqrt_weights > 0]),
            y_original if use_original else y
        )
        baseline = np.polynomial.Polynomial.fit(x=x, y=y, deg=poly_order, w=sqrt_weights)(x)

    return baseline


class TestModPoly(IterativePolynomialTester, WeightMaskingMixin):
    """Class for testing modpoly baseline."""

    func_name = 'modpoly'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, use_class, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_original=use_original, mask_initial_peaks=mask_initial_peaks
        )

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('max_iter', (3, None))
    @pytest.mark.parametrize('use_weights', (True, False))
    def test_implementation(self, poly_order, use_original, max_iter, use_weights):
        """Ensures expected functionality by comparing to a more basic implementation."""
        max_iter = max_iter if max_iter is not None else 500
        weights = np.random.default_rng(123).uniform(0, 1, self.x.shape) if use_weights else None
        fit, params = self.class_func(
            self.y, poly_order=poly_order, use_original=use_original, max_iter=max_iter,
            weights=weights
        )
        simple_fit = thresholding_polynomial(
            self.x, self.y, poly_order=poly_order, max_iter=len(params['tol_history']),
            use_original=use_original, weights=weights
        )
        assert_allclose(fit, simple_fit, rtol=1e-12, atol=1e-12)


class TestIModPoly(IterativePolynomialTester, WeightMaskingMixin):
    """Class for testing imodpoly baseline."""

    func_name = 'imodpoly'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('mask_initial_peaks', (True, False))
    def test_unchanged_data(self, use_class, use_original, mask_initial_peaks):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_original=use_original, mask_initial_peaks=mask_initial_peaks
        )

    @pytest.mark.parametrize('num_std', (-1, -0.01, 0, 1))
    def test_negative_num_std_fails(self, num_std):
        """Ensures `num_std` values less than 0 raise an exception."""
        if num_std < 0:
            with pytest.raises(ValueError):
                self.class_func(self.y, num_std=num_std)
        else:
            self.class_func(self.y, num_std=num_std)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    @pytest.mark.parametrize('use_original', (True, False))
    @pytest.mark.parametrize('num_std', (0, 0.7, 1))
    @pytest.mark.parametrize('max_iter', (3, None))
    @pytest.mark.parametrize('use_weights', (True, False))
    def test_implementation(self, poly_order, use_original, num_std, max_iter, use_weights):
        """Ensures expected functionality by comparing to a more basic implementation."""
        max_iter = max_iter if max_iter is not None else 500
        weights = np.random.default_rng(123).uniform(0, 1, self.x.shape) if use_weights else None
        fit, params = self.class_func(
            self.y, poly_order=poly_order, use_original=use_original, num_std=num_std,
            max_iter=max_iter, mask_initial_peaks=False, weights=weights
        )
        simple_fit = thresholding_polynomial(
            self.x, self.y, poly_order=poly_order, max_iter=len(params['tol_history']),
            use_original=use_original, num_std=num_std, weights=weights
        )
        assert_allclose(fit, simple_fit, rtol=1e-12, atol=1e-12)


class TestPenalizedPoly(IterativePolynomialTester, WeightMaskingMixin):
    """Class for testing penalized_poly baseline."""

    func_name = 'penalized_poly'

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'symmetric_truncated_quadratic',
            'a_truncated_quadratic',  # test that 'a' and 's' work as well
            's_truncated_quadratic',
            'asymmetric_huber',
            'symmetric_huber',
            'asymmetric_indec',
            'symmetric_indec'
        )
    )
    def test_unchanged_data(self, use_class, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('huber', 'p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function='a_hub')

    @pytest.mark.parametrize('weight_enum', (0, 1, 2, 3, 4))
    def test_weighting(self, weight_enum):
        """
        Tests that weighting is correctly applied by comparing to other algorithms.

        Weights were not included in the original penalized_poly method developed
        in [1]_, so need to ensure that their usage in pybaselines is correct.

        According to [1]_ (and independently verified), the penalized_poly function
        with the asymmetric truncated quadratic cost function, a threshold of 0, and
        an alpha_factor of 1 should be the same as the output of the ModPoly algorithm.

        Furthermore, the penalized_poly with any symmetric cost function and a threshold
        of infinity should equal to the output of a regular polynomial fit.

        Therefore, to ensure that weighting is correct for the penalized_poly, check
        both conditions.

        References
        ----------
        .. [1] Mazet, V., et al. Background removal from spectra by designing and
               minimising a non-quadratic cost function. Chemometrics and Intelligent
               Laboratory Systems, 2005, 76(2), 121-133.

        """
        if weight_enum == 0:
            # all weights = 1
            weights = None
        elif weight_enum == 1:
            # same as all weights = 1, but would cause issues if weights were
            # incorrectly multiplied
            weights = 2 * np.ones_like(self.y)
        elif weight_enum == 2:
            # binary mask, only fitting the first half of the data
            weights = np.ones_like(self.y)
            weights[self.x < 0.5 * (np.max(self.x) + np.min(self.x))] = 0
        elif weight_enum == 3:
            # random weights
            weights = np.random.default_rng(123).uniform(0, 1, self.y.shape)
        else:
            # weight array where the two endpoints have weighting >> 1
            weights = np.ones_like(self.y)
            fraction = max(1, ceil(self.y.shape[0] * 0.1))
            weights[:fraction] = 100
            weights[-fraction:] = 100

        poly_order = 2
        tol = 1e-3

        # Numpy polynomials are defined as minimizing ``(w * (y - y_fit))^2``
        # whereas pybaselines uses ``w * (y - y_fit)^2``, so use sqrt(weights) for the NumPy call
        poly_baseline = np.polynomial.Polynomial.fit(
            self.x, self.y, poly_order,
            w=np.sqrt(weights) if weights is not None else None
        )(self.x)
        penalized_poly_1 = self.class_func(
            self.y, poly_order, cost_function='s_huber', threshold=1e10, weights=weights
        )[0]

        assert_allclose(poly_baseline, penalized_poly_1, rtol=1e-10, atol=1e-12)

        modpoly_baseline = polynomial.modpoly(
            self.y, self.x, poly_order, tol=tol, weights=weights, use_original=True
        )[0]
        penalized_poly_2 = self.class_func(
            self.y, poly_order, cost_function='a_truncated_quadratic',
            threshold=0, weights=weights, alpha_factor=1, tol=tol
        )[0]

        assert_allclose(modpoly_baseline, penalized_poly_2, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('alpha_factor', (-0.1, 0, 1.01))
    def test_wrong_alpha_factor_fails(self, alpha_factor):
        """Ensures an alpha factor outside of (0, 1] fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha_factor=alpha_factor)


class TestLoess(IterativePolynomialTester, RecreationMixin, WeightMaskingMixin):
    """Class for testing loess baseline."""

    func_name = 'loess'
    allows_zero_iteration = False
    requires_unique_x = True
    supports_mask = False

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('delta', (0, 0.01))
    @pytest.mark.parametrize('conserve_memory', (True, False))
    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_unchanged_data(self, use_class, use_threshold, conserve_memory, delta):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, use_threshold=use_threshold,
            conserve_memory=conserve_memory, delta=delta
        )

    @pytest.mark.parametrize('use_threshold', (True, False))
    @pytest.mark.parametrize('use_original', (True, False))
    def test_x_ordering(self, use_threshold, use_original):
        """Ensures arrays are correctly sorted within the function."""
        super().test_x_ordering(use_threshold=use_threshold, use_original=use_original)

    @pytest.mark.parametrize('fraction', (-0.1, 1.1, 5))
    def test_wrong_fraction_fails(self, fraction):
        """Ensures a fraction value outside of (0, 1) raises an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, fraction)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3))
    def test_too_small_window_fails(self, poly_order):
        """Ensures a window smaller than poly_order + 2 raises an exception."""
        for num_points in range(poly_order + 2):
            with pytest.raises(ValueError):
                self.class_func(self.y, total_points=num_points, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (0, 1, 2, 3, 4))
    def test_high_polynomial_order_warns(self, poly_order):
        """Ensure a warning is emitted when using a polynomial order above 2."""
        if poly_order > 2:
            with pytest.warns(ParameterWarning):
                self.class_func(self.y, poly_order=poly_order)
        else:  # no warning should be emitted
            self.class_func(self.y, poly_order=poly_order)

    @pytest.mark.parametrize('poly_order', (1, 2))
    @pytest.mark.parametrize('delta', (0, 0.01))
    def test_output_coefs(self, poly_order, delta):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self.class_func(
            self.y, return_coef=True, poly_order=poly_order, delta=delta
        )
        # have to build the polynomial using the coefficients for each x-value
        recreated_poly = np.empty_like(baseline)
        for i, coef in enumerate(params['coef']):
            recreated_poly[i] = np.polynomial.Polynomial(coef)(self.x[i])

        # ignore coefficients that are all 0 since that means no fitting was done for
        # that x-value, so there are no actual coefficients available
        if delta > 0:
            mask = np.all(params['coef'] == 0, axis=1)
            baseline[mask] = 0

        assert_allclose(baseline, recreated_poly)

    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_compare_to_statsmodels(self, conserve_memory):
        """
        Compares the output of loess to the output of statsmodels.lowess.

        The library statsmodels has a well-tested lowess implementation, so
        can compare the output of polynomial.loess to statsmodels to ensure
        that the pybaselines implementation is correct.

        Since pybaselines's loess is for calculating the baseline rather than
        smoothing, the following changes need to be made to match statsmodels:

        * statsmodels uses int(fraction * num_x) to determine the window size while
          pybaselines uses ceil(fraction * num_x), so need to specify total points
          instead of fraction.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * set tol to -1 so that it goes through all iterations.

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, iterations, delta=0.0).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # Use set values to not worry about rng generation changes causing issues.
        # Used the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x), kind='stable')
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = LOESS_X
        y = LOESS_Y

        # test several iterations to ensure weighting is correct
        for iterations in range(4):
            output = self.algorithm_base(x, check_finite=False, assume_sorted=True).loess(
                y, conserve_memory=conserve_memory, total_points=total_points,
                max_iter=iterations, tol=-1, scale=4.0469385011764905, symmetric_weights=True,
                delta=0.0
            )

            assert_allclose(
                output[0], STATSMODELS_LOESS_ITER[iterations],
                err_msg=f'failed on iteration {iterations}'
            )

    @pytest.mark.parametrize('delta', (0.01, 0.3))
    def test_compare_to_statsmodels_delta(self, delta):
        """
        Compares the output of loess to the output of statsmodels.lowess when using delta.

        The library statsmodels has a well-tested lowess implementation, so
        can compare the output of polynomial.loess to statsmodels to ensure
        that the pybaselines implementation is correct.

        Since pybaselines's loess is for calculating the baseline rather than
        smoothing, the following changes need to be made to match statsmodels:

        * statsmodels uses int(fraction * num_x) to determine the window size while
          pybaselines uses ceil(fraction * num_x), so need to specify total points
          instead of fraction.
        * statsmodels divides the residuals by 6 * median-absolute-value(residuals)
          when weighting residuals, while pybaselines divides by
          m-a-v * scale / 0.6744897501960817, so set scale to 4.0469385011764905 to
          get 6 and match statsmodels.
        * set symmetric weights to True.
        * only test the first iteration, since just want to check which points are selected
          for fitting

        The outputs from statsmodels were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, fraction, 0, delta=delta * (x.max() - x.min())).T[1]

        with statsmodels version 0.11.1.

        """
        num_x = 100
        fraction = 0.1
        total_points = int(num_x * fraction)
        # use set values since minimum numpy version is < 1.17
        # once min numpy version is >= 1.17, can use the following to create x and y:
        # random_generator = np.random.default_rng(0)
        # x = np.sort(random_generator.uniform(0, 10 * np.pi, num_x), kind='stable')
        # use a simple sine function since only smoothing the data
        # y = np.sin(x) + random_generator.normal(0, 0.3, num_x)
        x = LOESS_X
        y = LOESS_Y

        output = self.algorithm_base(x, check_finite=False, assume_sorted=True).loess(
            y, total_points=total_points, max_iter=0, scale=4.0469385011764905,
            symmetric_weights=True, delta=delta * (x.max() - x.min())
        )

        assert_allclose(output[0], STATSMODELS_LOESS_DELTA[delta])

    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_input_weights(self, use_threshold):
        """Ensures the input weights are sorted correctly."""
        super().test_input_weights(use_threshold=use_threshold)

    @pytest.mark.threaded_test
    @pytest.mark.parametrize('conserve_memory', (True, False))
    def test_threading(self, conserve_memory):
        """Tests the different possible computation routes under threading."""
        delta = 0.05 * (self.x.max() - self.x.min())  # use a larger delta to speed up method
        super().test_threading(conserve_memory=conserve_memory, delta=delta)

    def test_custom_sigma_func(self):
        """Ensures input sigma_func modifies the reweighting."""
        baseline, params = self.class_func(self.y)
        baseline2, params2 = self.class_func(self.y, sigma_func=lambda vals: np.std(vals[vals < 0]))

        # simple check that different sigma calcs produced different baselines and weights
        with pytest.raises(AssertionError):
            assert_allclose(baseline2, baseline, rtol=1e-4, atol=1e-3)
        with pytest.raises(AssertionError):
            assert_allclose(params2['weights'], params['weights'], rtol=1e-1, atol=1e-1)

    def test_incorrect_sigma_func_fails(self):
        """Ensures an exception is raised if input sigma_func does not return a float."""
        with pytest.raises(TypeError, match='"sigma_func" must return a float'):
            self.class_func(self.y, sigma_func=lambda vals: 'a')

    def test_zero_sigma_exits(self):
        """Ensures the method exits early when the calculated noise sigma is ~0.

        Replicates statsmodels issue #2108.

        """
        x = np.arange(20)
        y = np.array([0] * 10 + [1] * 10, dtype=float)
        fraction = 2 / 3
        total_points = int(len(x) * fraction)
        with pytest.warns(ParameterWarning, match='calculated noise scale is near 0'):
            output = self.algorithm_base(x).loess(
                y, total_points=total_points, max_iter=3, delta=0,
                scale=4.0469385011764905, symmetric_weights=True, tol=-1
            )[0]
        expected_output = np.array([
            0, 0, 0, 0, 0, 0, 0, 0.03796574, 0.29511209, 0.44982749, 0.55017251, 0.70488791,
            0.96203426, 1, 1, 1, 1, 1, 1, 1
        ])
        assert_allclose(output, expected_output, rtol=1e-6, atol=1e-9)

    def test_zero_sigma_exits_2(self):
        """Ensures the method exits early when the calculated noise sigma is ~0.

        Replicates the first part of statsmodels issue #1798.

        """
        x = np.arange(20)
        y = np.arange(20, dtype=float)
        fraction = 0.4
        total_points = int(len(x) * fraction)
        with pytest.warns(ParameterWarning, match='calculated noise scale is near 0'):
            output = self.algorithm_base(x).loess(
                y, total_points=total_points, max_iter=3, delta=0,
                scale=4.0469385011764905, symmetric_weights=True, tol=-1
            )[0]
        expected_output = y  # should be a perfect fit
        assert_allclose(output, expected_output, rtol=1e-14, atol=1e-13)

    @pytest.mark.parametrize('max_iter', (1, 2))
    def test_zero_weights_fill(self, max_iter):
        """Ensures a window with zero weights with fill with y instead of causing numerical issues.

        Dataset is adapted from statsmodels issue #7700. The data files for statsmodels's output
        were created using::

            from statsmodels.nonparametric.smoothers_lowess import lowess
            output = lowess(y, x, frac=11 / len(x), it=max_iter, delta=0).T[1]

        with statsmodels version 0.14.6.

        """
        y = np.array([
            29.60046, 29.70066, 29.99869, 30.18495,
            30.52497, 30.88539, 31.06073, 31.16298, 31.3087, 31.34476, 31.4047, 31.27913,
            31.29533, 31.14104, 31.033, 30.95522, 30.7452, 30.6161, 30.48558, 30.20304,
            29.94876, 29.49816, 28.99673, 28.47641, 27.75036, 26.98692, 26.22662, 25.29733,
            24.45699, 23.47883, 22.421, 21.46149, 20.50521, 19.55747, 18.71905, 17.97059,
            17.4616, 17.15413, 17.02539, 17.23645, 17.69518, 18.47265, 19.49916, 20.87392,
            22.47629, 24.34076, 26.46264, 28.66842, 31.13522, 33.57669, 35.95129, 38.50984,
            40.9788, 43.45954, 45.54811, 47.72132, 49.50215, 51.28018, 52.67683, 53.87601,
            54.98996, 55.89579, 56.45095, 56.88656, 57.15155, 57.16919, 57.04115, 56.87761,
            56.42096, 55.93649, 55.2568, 54.47306, 53.79956, 52.8701, 51.84985, 50.93586,
            49.95632, 48.73087, 47.77627, 46.75819, 45.54977, 44.36957, 43.32188, 42.29313,
            41.24385, 40.14291, 39.15614, 38.17805, 37.27126, 36.13561, 35.32942, 34.35569,
            33.69126, 32.67565, 31.91131, 31.0636, 30.32011, 29.60982, 28.88217, 28.10989,
            27.56996, 27.03619, 26.36284, 25.82758, 25.27555, 24.80477, 24.25029, 23.74979,
            23.31028, 22.95834, 22.56406, 22.13128, 21.81209, 21.42739, 21.12386, 20.8205,
            20.52693, 20.26264, 19.94682, 19.74871, 19.47004, 19.28826, 19.09282, 18.8813,
            18.69543, 18.51512, 18.37025, 18.21213, 18.09597, 18.00692, 17.84771, 17.7365,
            17.70439, 17.54311, 17.50521, 17.42641, 17.32607, 17.29374, 17.17156, 17.14076,
            17.18559, 17.12909, 17.11519, 17.06809, 17.05098, 17.06691, 17.02511, 17.01555,
            17.07787, 17.05032, 17.05407, 17.06751, 17.12841, 17.12312, 17.16593, 17.21924,
            17.19979, 17.25681, 17.31144, 17.36246, 17.43259, 17.43767, 17.5086, 17.58345,
            17.62989, 17.70608, 17.70383, 17.81441, 17.82661, 17.8836, 18.00816, 18.05311,
            18.16044, 18.19468, 18.24426, 18.32978, 18.41256, 18.47817, 18.57559, 18.6523,
            18.71417, 18.79602, 18.89392, 18.96791, 19.0598, 19.17692, 19.25897, 19.33334,
            19.45276, 19.56273, 19.63092, 19.71592, 19.83377, 19.91831, 19.97547, 20.07111,
            20.15791, 20.23325, 20.38081, 20.49393, 20.54687, 20.62749, 20.70332, 20.81285,
            20.87916, 21.01356, 21.07556, 21.19642, 21.26882, 21.35373, 21.45083, 21.55625,
            21.66463, 21.75115, 21.8033, 21.9497, 22.06961, 22.1253, 22.20523, 22.32333,
            22.41526, 22.50364, 22.62715, 22.70702, 22.80392, 22.89037, 23.02072, 23.12152,
            23.18633, 23.29179, 23.39558, 23.4171, 23.56042, 23.59962, 23.76348, 23.7985,
            23.93591, 23.97028, 24.04745, 24.12475
        ])
        x = np.linspace(2160, 2559, len(y)) / 60
        output = self.algorithm_base(x).loess(
            y, poly_order=1, total_points=11, max_iter=max_iter, delta=0,
            scale=4.0469385011764905, symmetric_weights=True, tol=-1
        )[0]
        expected_output = np.loadtxt(
            Path(__file__).parent.joinpath(f'data/lowess_zero_weights_iter{max_iter}.csv')
        )
        assert_allclose(output, expected_output, rtol=1e-11, atol=1e-11)

    def test_zero_weights(self):
        """Simpler version of test_zero_weights_fill, using input all-zeros weights.

        Allows testing for which indices should fail.

        """
        output, params = self.class_func(
            self.y, delta=0, weights=np.zeros_like(self.y), max_iter=0, return_coef=True
        )

        assert_allclose(output, self.y, rtol=1e-14, atol=1e-14)
        assert np.isnan(params['coef']).all()

    @pytest.mark.parametrize('use_threshold', (True, False))
    def test_weight_masking(self, use_threshold):
        """Masking only works if `use_threshold` is True."""
        if use_threshold:
            super().test_weight_masking(use_threshold=use_threshold)
        else:
            with pytest.raises(AssertionError):
                super().test_weight_masking(use_threshold=use_threshold)


class TestQuantReg(IterativePolynomialTester, RecreationMixin):
    """Class for testing quant_reg baseline."""

    func_name = 'quant_reg'
    required_kwargs = {'tol': 1e-9}
    required_repeated_kwargs = {'tol': 1e-3}
    allows_zero_iteration = False

    @pytest.mark.parametrize('quantile', (0, 1, -0.1, 1.1))
    def test_outside_quantile_fails(self, quantile):
        """Ensures quantile values outside of (0, 1) raise an exception."""
        with pytest.raises(ValueError):
            self.class_func(self.y, quantile=quantile)

    @pytest.mark.parametrize('quantile', tuple(STATSMODELS_QUANTILES.keys()))
    def test_compare_to_statsmodels(self, quantile):
        """
        Compares the output of quant_reg to statsmodels's quantile regression implementation.

        The library statsmodels has a well-tested quantile regression implementation,
        so can compare the output of polynomial.quant_reg to statsmodels to ensure
        that the pybaselines implementation is correct.

        The outputs from statsmodels were created using::

            from statsmodels.regression.quantile_regression import QuantReg
            vander = np.polynomial.polynomial.polyvander(x, 1)
            fitter = QuantReg(y, vander).fit(quantile, max_iter=1000, p_tol=1e-6).predict()

        with statsmodels version 0.11.1.

        Could also compare with the "true" quantile regression result using linear
        programming such as detailed in:

        https://stats.stackexchange.com/questions/384909/formulating-quantile-regression-as-
        linear-programming-problem

        but the comparison to statsmodels is good enough since it uses an iteratively
        reweighted least squares calculation for the quantile regression similar to the
        pybaselines implementation, and the linear programming requires a scipy version
        of at least 1.0 or 1.6 to get a fast, reliable result due to the older solvers not
        working as well.

        """
        x = np.linspace(-1000, 1000, 200)
        # Use set values to not worry about rng generation changes causing issues.
        # Used the following to create y:
        # y = x + np.random.default_rng(0).normal(0, 200, x.size)
        y = QUANTILE_Y

        output = self.algorithm_base(x, check_finite=False, assume_sorted=True).quant_reg(
            y, poly_order=1, quantile=quantile, tol=1e-9, eps=1e-12
        )

        assert_allclose(output[0], STATSMODELS_QUANTILES[quantile], rtol=1e-6)


class TestGoldindec(PolynomialTester, WeightMaskingMixin):
    """Class for testing goldindec baseline."""

    func_name = 'goldindec'
    checked_keys = ('weights', 'tol_history', 'threshold')

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize(
        'cost_function',
        (
            'asymmetric_truncated_quadratic',
            'a_truncated_quadratic',
            'asymmetric_huber',
            'asymmetric_indec',
            'indec',
            'huber',
            'truncated_quadratic'
        )
    )
    def test_unchanged_data(self, use_class, cost_function):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(use_class, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('p_huber', ''))
    def test_unknown_cost_function_prefix_fails(self, cost_function):
        """Ensures cost function with no prefix or a wrong prefix fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function=cost_function)

    @pytest.mark.parametrize('cost_function', ('s_huber', 's_indec', 'symmetric_indec'))
    def test_symmetric_cost_function_fails(self, cost_function):
        """Ensures a symmetric cost function fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, cost_function=cost_function)

    def test_unknown_cost_function_fails(self):
        """Ensures than an unknown cost function fails."""
        with pytest.raises(KeyError):
            self.class_func(self.y, cost_function='a_hub')

    @pytest.mark.parametrize('weight_enum', (0, 1, 2, 3))
    def test_weighting(self, weight_enum):
        """
        Tests that weighting is correctly applied by comparing to other algorithms.

        Weights were not included in the original goldindec method, so need to ensure
        that their usage in pybaselines is correct.

        For uniform weights, the reference baseline is simply the unweighted calculation,
        since they should be equal. For non-uniform weights, compare to the output of
        penalized_poly, whose weighting is correctly tested, using the output optimal
        threshold.

        """
        if weight_enum == 0:
            # all weights = 1
            weights = None
            uniform_weights = True
        elif weight_enum == 1:
            # same as all weights = 1, but would cause issues if weights were
            # incorrectly multiplied
            weights = np.full_like(self.y, 2)
            uniform_weights = True
        elif weight_enum == 2:
            # binary mask, only fitting the first half of the data
            weights = np.ones_like(self.y)
            weights[self.x < 0.5 * (np.max(self.x) + np.min(self.x))] = 0
            uniform_weights = False
        else:
            # weight array where the two endpoints have weighting >> 1
            weights = np.ones_like(self.y)
            fraction = max(1, ceil(self.y.shape[0] * 0.1))
            weights[:fraction] = 100
            weights[-fraction:] = 100
            uniform_weights = False

        poly_order = 2
        fit_baseline, params = self.class_func(self.y, poly_order=poly_order, weights=weights)
        if uniform_weights:
            reference_baseline = self.class_func(self.y, poly_order=poly_order)[0]
        else:
            reference_baseline = polynomial.penalized_poly(
                self.y, self.x, poly_order=poly_order, weights=weights,
                threshold=params['threshold'], cost_function='a_indec'
            )[0]

        assert_allclose(fit_baseline, reference_baseline)

    @pytest.mark.parametrize('exit_enum', (0, 1, 2, 3))
    def test_tol_history(self, exit_enum):
        """
        Ensures the 'tol_history' item in the parameter output is correct.

        Since the shape of 'tol_history' is dictated by the number of iterations
        completed for fitting each threshold value and for iterating between
        threshold values, need to ensure each exit criteria works independently.

        """
        if exit_enum == 0:
            # inner fitting does more iterations
            max_iter = 15
            tol = -1
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = -1

            expected_shape_0 = max_iter_2 + 2
            expected_shape_1 = max_iter

        if exit_enum == 1:
            # outer fitting does more iterations
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = -1

            expected_shape_0 = max_iter_2 + 2
            expected_shape_1 = max_iter_2

        if exit_enum == 2:
            # only one iteration completed; exits due to tol_2
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 1e6
            tol_3 = -1

            expected_shape_0 = 3
            expected_shape_1 = 1

        if exit_enum == 3:
            # only one iteration completed; exits due to tol_3
            max_iter = 15
            tol = 1e6
            max_iter_2 = 10
            tol_2 = 0
            tol_3 = 1e6

            expected_shape_0 = 3
            expected_shape_1 = 1

        _, params = self.class_func(
            self.y, max_iter=max_iter, tol=tol, max_iter_2=max_iter_2,
            tol_2=tol_2, tol_3=tol_3
        )

        assert params['tol_history'].shape[0] == expected_shape_0
        assert params['tol_history'].shape[1] == expected_shape_1

    @pytest.mark.parametrize('alpha_factor', (-0.1, 0, 1.01))
    def test_wrong_alpha_factor_fails(self, alpha_factor):
        """Ensures an alpha factor outside of (0, 1] fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, alpha_factor=alpha_factor)

    @pytest.mark.parametrize('peak_ratio', (-0.1, 0, 1, 1.01))
    def test_wrong_peak_ratio_fails(self, peak_ratio):
        """Ensures a peak ratio outside of (0, 1) fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, peak_ratio=peak_ratio)


@pytest.mark.parametrize(
    'residual', (np.arange(100), -np.arange(100), np.linspace(-100, 100, 100))
)
@pytest.mark.parametrize('scale', (-3, 0.01, 1, 3, 50))
@pytest.mark.parametrize('symmetric', (True, False))
def test_tukey_square(residual, scale, symmetric):
    """
    Tests the Tukey square (sqrt of Tukey's bisquare) weighting for loess.

    Note for future, a negative scale is included to ensure it has no effect since it is
    squared in the weighting.

    """
    weights = polynomial._tukey_square(residual, scale, symmetric)

    assert np.all(weights >= 0)
    assert np.all(weights <= 1)

    if not symmetric:
        assert np.all(weights[residual < 0] == 1)

    # ensure that skipping the second squaring part of Tukey's bisquare does not change
    # the weights
    assert_allclose(weights, np.sqrt(weights**2))


@pytest.mark.parametrize(
    'values', (np.arange(10), np.linspace(-10, 10), np.full(10, 1))
)
def test_median_absolute_value(values):
    """Tests the median absolute values function."""
    mav_calc = polynomial._median_absolute_value(values)
    mav_actual = np.median(np.abs(values)) / 0.6744897501960817

    assert_allclose(mav_calc, mav_actual, rtol=1e-14, atol=1e-14)

    # also compare against scipy, set center to 0 to make the MAD into MAV
    mav_scipy = stats.median_abs_deviation(
        values, scale='normal', center=lambda *args, **kwargs: 0
    )
    assert_allclose(mav_calc, mav_scipy, rtol=1e-14, atol=1e-14)


def test_loess_solver():
    """Tests that the loess solver solves `Ax=b` given `A.T` and `b`."""
    x = np.linspace(-1.0, 1.0, 50)
    coefs = np.array([2.0, -1.0, 0.2])
    y = coefs[0] + coefs[1] * x + coefs[2] * x**2

    vander = np.polynomial.polynomial.polyvander(x, coefs.size - 1)

    solved_coefs = polynomial._loess_solver(vander.T, y)

    assert_allclose(solved_coefs, coefs, rtol=1e-12, atol=1e-14)


def test_determine_fits_simple():
    """A simple test to ensure the inner workings of _determine_fits work."""
    x = np.arange(22, dtype=float)
    num_x = x.shape[0]
    total_points = 5
    delta = 2.1  # should skip every other x-value, excluding the endpoints

    windows, fits, skips = polynomial._determine_fits(x, num_x, total_points, delta)

    # always fit first point
    desired_windows = [[0, total_points]]
    desired_fits = [0]
    desired_skips = []
    left = 0
    right = total_points
    for i, x_val in enumerate(x[1:-1], 1):
        if i % 2:  # all odd indices are skipped in this test setup
            # should be a slice that includes the last fit index and next fit index
            desired_skips.append([i - 1, i + 2])
        else:
            desired_fits.append(i)
            while right < num_x and x_val - x[left] > x[right] - x_val:
                left += 1
                right += 1
            desired_windows.append([left, right])
    # always fit last point
    desired_fits.append(num_x - 1)
    desired_windows.append([num_x - total_points, num_x])

    assert_array_equal(windows, desired_windows)
    assert_array_equal(fits, desired_fits)
    assert_array_equal(skips, desired_skips)


@pytest.mark.parametrize('delta', (0.0, 0.01, 0.5, -1.0, 3.0, np.nan, np.inf, -np.inf))
@pytest.mark.parametrize('total_points', (2, 10, 25, 50))
def test_determine_fits(delta, total_points):
    """Tests various inputs for _determine_fits to ensure any float delta works."""
    x = np.linspace(-1, 1, 50)
    num_x = x.shape[0]

    windows, fits, skips = polynomial._determine_fits(x, num_x, total_points, delta)

    assert windows.shape[0] == fits.shape[0]
    assert windows.shape[1] == 2

    # always fit first and last x-values
    assert fits[0] == 0
    assert fits[-1] == num_x - 1
    assert_array_equal(windows[0], (0, total_points))
    assert_array_equal(windows[-1], (num_x - total_points, num_x))

    # each window should be separated by total_points indices
    windows_transpose = windows.T
    assert_array_equal(
        windows_transpose[1] - windows_transpose[0],
        np.full(windows.shape[0], total_points)
    )

    # ensure no repeated fit indices
    assert not (np.diff(fits) == 0).sum()

    if delta <= 0:  # no points should be skipped
        assert skips.shape[0] == 0
        assert windows.shape[0] == num_x
        assert fits.shape[0] == num_x

        assert_array_equal(fits, np.arange(num_x))


def test_fill_skips():
    """Tests the linear interpolation performed by _fill_skips."""
    x = np.arange(20)
    y_actual = 2 + 5 * x
    y_calc = y_actual.copy()
    # `skips` slices y[left:right] where y_slice[0] and y_slice[-1] are actual values
    # and in-between will be calculated using interpolation; fill in the sections
    # [left+1:right-1] in y_calc with zeros, and then check that they are returned to
    # the correct value by _fill_skips
    skips = np.array([[0, 5], [8, 14], [16, x.shape[0]]], dtype=np.intp)
    for (left, right) in skips:
        y_calc[left + 1:right - 1] = 0

    output = polynomial._fill_skips(x, y_calc, skips)

    # should not output anything from the function
    assert output is None
    # y_calc should be same as y_actual after interpolating each section
    assert_allclose(y_calc, y_actual)


def test_fill_skips_no_skips():
    """Ensures _fill_skips does not affect the input array when there are no skipped points."""
    skips = np.array([], np.intp).reshape(0, 2)

    x = np.arange(10)
    y_calc = np.empty(x.shape[0])
    y_calc[0] = 5
    y_calc[-1] = 10

    y_calc_before = y_calc.copy()

    polynomial._fill_skips(x, y_calc, skips)

    # y_calc should be unchanged since skips is an empty array
    assert_array_equal(y_calc, y_calc_before)
