# -*- coding: utf-8 -*-
"""Tests for pybaselines.classification.

@author: Donald Erb
Created on July 3, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy

from pybaselines import classification
from pybaselines.utils import ParameterWarning, gaussian, whittaker_smooth

from .conftest import BaseTester, InputWeightsMixin
from .data import PYWAVELETS_HAAR


def _nieve_rolling_std(data, half_window, ddof=0):
    """
    A nieve approach for a rolling standard deviation.

    Used for ensuring faster, more complex approaches are correct.

    Parameters
    ----------
    data : numpy.ndarray
        The array for the calculation. Should be padded on the left and right
        edges by `half_window`.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int, optional
        The degrees of freedom for the calculation. Default is 0.

    Returns
    -------
    rolling_std : numpy.ndarray
        The array of the rolling standard deviation for each window.

    """
    num_y = data.shape[0]
    rolling_std = np.array([
        np.std(data[max(0, i - half_window):min(i + half_window + 1, num_y)], ddof=ddof)
        for i in range(num_y)
    ])

    return rolling_std


@pytest.mark.parametrize('y_scale', (1, 1e-9, 1e9))
@pytest.mark.parametrize('half_window', (1, 3, 10, 30))
@pytest.mark.parametrize('ddof', (0, 1))
def test_rolling_std(y_scale, half_window, ddof):
    """
    Test the rolling standard deviation calculation against a nieve implementation.

    Also tests different y-scales while using the same noise level, since some
    implementations have numerical instability when values are small/large compared
    to the standard deviation.

    """
    x = np.arange(100)
    y = y_scale * np.sin(x) + np.random.default_rng(0).normal(0, 0.2, x.size)
    # only compare within [half_window:-half_window] since the calculation
    # can have slightly different values at the edges
    compare_slice = slice(half_window, -half_window)

    actual_rolled_std = _nieve_rolling_std(y, half_window, ddof)
    calc_rolled_std = classification._rolling_std(
        np.pad(y, half_window, 'reflect'), half_window, ddof
    )[half_window:-half_window]

    assert_allclose(calc_rolled_std[compare_slice], actual_rolled_std[compare_slice])


@pytest.mark.parametrize('y_scale', (1, 1e-9, 1e9))
@pytest.mark.parametrize('half_window', (1, 3, 10, 30))
@pytest.mark.parametrize('ddof', (0, 1))
def test_padded_rolling_std(y_scale, half_window, ddof):
    """
    Test the padded rolling standard deviation calculation against a nieve implementation.

    Also tests different y-scales while using the same noise level, since some
    implementations have numerical instability when values are small/large compared
    to the standard deviation.

    """
    x = np.arange(100)
    y = y_scale * np.sin(x) + np.random.default_rng(0).normal(0, 0.2, x.size)
    # only compare within [half_window:-half_window] since the calculation
    # can have slightly different values at the edges
    compare_slice = slice(half_window, -half_window)

    actual_rolled_std = _nieve_rolling_std(y, half_window, ddof)
    calc_rolled_std = classification._padded_rolling_std(y, half_window, ddof)

    assert_allclose(calc_rolled_std[compare_slice], actual_rolled_std[compare_slice])


@pytest.mark.parametrize(
    'inputs_and_expected',
    (
        [2, [0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0]],
        [2, [0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
        [3, [1, 0, 1, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0, 0, 1, 1]],
        [2, [0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1]],
        [
            5, [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]
    )
)
def test_refine_mask(inputs_and_expected):
    """Test that _refine_mask fills holes in binary mask."""
    min_length, mask, expected_mask = inputs_and_expected
    output_mask = classification._refine_mask(mask, min_length)

    assert_array_equal(np.asarray(expected_mask, bool), output_mask)


@pytest.mark.parametrize(
    'mask_and_expected',
    (   # mask, peak-starts, peak-ends
        ([0, 0, 1, 0, 0, 0, 1, 1, 0], [0, 2, 7], [2, 6, 8]),
        ([1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1], [2, 5], [4, 9]),
        ([0, 0, 0, 0, 0, 0, 0], [0], [6]),  # all peak points, will assign first and last indices
        ([1, 1, 1, 1, 1, 1, 1], [], [])  # all baseline points, will not assign any starts or ends
    )
)
def test_find_peak_segments(mask_and_expected):
    """Ensures peak starts and ends are correct for boolean and binary masks."""
    mask, expected_starts, expected_ends = mask_and_expected
    expected_starts = np.array(expected_starts)
    expected_ends = np.array(expected_ends)

    calc_starts, calc_ends = classification._find_peak_segments(np.array(mask, dtype=bool))

    assert_array_equal(expected_starts, calc_starts)
    assert_array_equal(expected_ends, calc_ends)

    # test that it also works with a binary array with 0s and 1s
    calc_starts, calc_ends = classification._find_peak_segments(np.array(mask, dtype=int))

    assert_array_equal(expected_starts, calc_starts)
    assert_array_equal(expected_ends, calc_ends)


@pytest.mark.parametrize('interp_half_window', (0, 1, 3, 1000))
def test_averaged_interp(interp_half_window):
    """Ensures the averaged interpolated works for different interpolation windows."""
    mask = np.array([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], bool)
    peak_starts = [3, 9]
    peak_ends = [6, 13]

    x = np.arange(mask.shape[0])
    y = np.sin(x)
    num_y = y.shape[0]
    expected_output = y.copy()
    for start, end in zip(peak_starts, peak_ends):
        left_mean = np.mean(
            y[max(0, start - interp_half_window):min(start + interp_half_window + 1, num_y)]
        )
        right_mean = np.mean(
            y[max(0, end - interp_half_window):min(end + interp_half_window + 1, num_y)]
        )
        expected_output[start + 1:end] = np.linspace(left_mean, right_mean, end - start + 1)[1:-1]

    calc_output = classification._averaged_interp(x, y, mask, interp_half_window)

    assert_allclose(calc_output, expected_output)


def test_averaged_interp_warns():
    """Ensure warning is issued when mask is all 0s or all 1s."""
    num_points = 50
    x = np.arange(num_points)
    y = np.sin(x)

    # all ones indicate all baseline points; output should be the same as y
    mask = np.ones(num_points, dtype=bool)
    expected_output = np.linspace(y[0], y[-1], num_points)
    with pytest.warns(ParameterWarning):
        output = classification._averaged_interp(x, y, mask)
    assert_array_equal(output, y)

    # all zeros indicate all peak points; output should interpolate between first and last points
    mask = np.zeros(num_points, dtype=bool)
    expected_output = np.linspace(y[0], y[-1], num_points)
    with pytest.warns(ParameterWarning):
        output = classification._averaged_interp(x, y, mask)
    assert_allclose(output, expected_output)


def test_averaged_interp_negative_interp_half_window():
    """Ensure exception is raised when input interp_half_window is negative."""
    x = np.arange(5)
    y = 1 * x
    mask = np.ones_like(y, dtype=bool)
    mask[1] = False  # so no warning is emitted about all points belonging to baseline
    with pytest.raises(ValueError):
        classification._averaged_interp(x, y, mask, interp_half_window=-1)


@pytest.mark.parametrize('window_size', [20, 21])
@pytest.mark.parametrize('scale', [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_haar(scale, window_size):
    """Ensures the Haar wavelet implementation is correct."""
    haar_wavelet = classification._haar(window_size, scale)
    actual_window_size = len(haar_wavelet)

    assert isinstance(haar_wavelet, np.ndarray)

    # odd scales should produce odd-length wavelets; even scale produces even-length
    assert scale % 2 == actual_window_size % 2

    half_window = actual_window_size // 2
    if scale % 2:
        # wavelet for odd scales should be 0 at mid-point
        assert_allclose(haar_wavelet[half_window], 0., 0, 1e-14)

    # the wavelet should be reflected around the mid-point; total area should
    # be 0, and the area for [:mid_point] and [-mid_moint:] should be equivalent
    # and equal to (scale // 2) / sqrt(scale), where sqrt(scale) is due to
    # normalization.
    assert_allclose(haar_wavelet.sum(), 0., 0, 1e-14)
    # re-normalize the wavelet to make further calculations easier; all values
    # should be -1, 0, or 1 after re-normilazation
    haar_wavelet *= np.sqrt(scale)

    left_side = haar_wavelet[:half_window]
    right_side = haar_wavelet[-half_window:]

    assert_allclose(left_side, -right_side[::-1], 1e-14)
    assert_allclose(left_side.sum(), scale // 2, 1e-14)
    assert_allclose(-right_side.sum(), scale // 2, 1e-14)


@pytest.mark.parametrize('scale', [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_haar_cwt_comparison_to_pywavelets(scale):
    """
    Compares the Haar wavelet cwt with pywavelet's implementation.

    pywavelets's cwt does not naturally work with their Haar wavelet, so had to apply
    a patch mentioned in pywavelets issue #365 to make their cwt work with their Haar.
    Additionally, had to apply the patches in pywavelets pull request #580 to correct an
    issue with pywavelets's cwt interpolation so that the output looks correct.

    The outputs from pywavelets were created using::

        import pywt
        output = pywt.cwt(y, [scale], 'haar')[0][0]

    with pywavelets version 1.1.1.

    The idea for the input array was adapted from a MATLAB example at
    https://www.mathworks.com/help/wavelet/gs/interpreting-continuous-wavelet-coefficients.html.

    The squares of the two cwt arrays are compared since until scipy version 1.4, the
    convolution was incorrectly done on the wavelet rather than the reversed wavelet,
    and since the Haar wavelet is not symmetric, the output will be reversed of what
    it should be and creates negative values instead of positive and vice versa. That
    does not affect any calculations within pybaselines, so it is not a concern.

    """
    y = np.zeros(100)
    y[50] = 1

    haar_cwt = classification._cwt(y, classification._haar, [scale])[0]
    # test absolute tolerance rather than relative tolerance since
    # some values are very close to 0
    assert_allclose(haar_cwt**2, PYWAVELETS_HAAR[scale]**2, 0, 1e-14)
    try:
        scipy_version = scipy.__version__.split('.')[:2]
        major = int(scipy_version[0])
        minor = int(scipy_version[1])
        if major > 1 or (major == 1 and minor >= 4):
            test_values = True
        else:
            test_values = False
    except Exception:  # in case the version checking is wrong, then just ignore
        test_values = False

    if test_values:
        assert_allclose(haar_cwt, PYWAVELETS_HAAR[scale], 0, 1e-14)


class ClassificationTester(BaseTester, InputWeightsMixin):
    """Base testing class for classification functions."""

    module = classification
    algorithm_base = classification._Classification
    checked_keys = ('mask',)
    weight_keys = ('mask',)


class TestGolotvin(ClassificationTester):
    """Class for testing golotvin baseline."""

    func_name = 'golotvin'
    required_kwargs = {'half_window': 15, 'num_std': 6}
    required_repeated_kwargs = {'half_window': 15, 'num_std': 6}


class TestDietrich(ClassificationTester):
    """Class for testing dietrich baseline."""

    func_name = 'dietrich'

    @pytest.mark.parametrize('return_coef', (True, False))
    @pytest.mark.parametrize('max_iter', (0, 1, 2))
    def test_output(self, return_coef, max_iter):
        """Ensures that the output has the desired format."""
        additional_keys = []
        if return_coef and max_iter > 0:
            additional_keys.append('coef')
        if max_iter > 1:
            additional_keys.append('tol_history')
        super().test_output(
            additional_keys=additional_keys, return_coef=return_coef, max_iter=max_iter
        )

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self.class_func(data=self.y, **self.kwargs, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self.class_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter - 1


class TestStdDistribution(ClassificationTester):
    """Class for testing std_distribution baseline."""

    func_name = 'std_distribution'


class TestFastChrom(ClassificationTester):
    """Class for testing fastchrom baseline."""

    func_name = 'fastchrom'

    @pytest.mark.parametrize('threshold', (None, 1, lambda std: np.mean(std)))
    def test_threshold_inputs(self, threshold):
        """Ensures a callable threshold value works."""
        self.class_func(self.y, half_window=20, threshold=threshold)


class TestCwtBR(ClassificationTester):
    """Class for testing cwt_br baseline."""

    func_name = 'cwt_br'
    checked_keys = ('mask', 'tol_history', 'best_scale')

    @pytest.mark.parametrize('scales', (None, np.arange(3, 20)))
    def test_output(self, scales):
        """Ensures that the output has the desired format."""
        super().test_output(scales=scales)


class TestFabc(ClassificationTester):
    """Class for testing fabc baseline."""

    func_name = 'fabc'
    checked_keys = ('mask', 'weights')
    weight_keys = ('mask', 'weights')

    @pytest.mark.parametrize('weights_as_mask', (True, False))
    def test_input_weights(self, weights_as_mask):
        """Tests input weights as both a mask and as weights."""
        super().test_input_weights(weights_as_mask=weights_as_mask)


def rubberband_data(x_data):
    """
    Creates y-data for testing indexing for the rubberband baseline.

    Parameters
    ----------
    x_data : numpy.ndarray
        The x-values; should be the same x-values used as the default testing values.
        (ie. np.linspace(1, 100, 1000))

    Returns
    -------
    x_data : numpy.ndarray
        The x-values.
    y_data : numpy.ndarray
        The y-values.

    Notes
    -----
    Produces a baseline such that the convex hull produces a sitution where the
    minimum and maximum index occur on the same value, mirroring issue 29.

    """
    signal = (
        500  # constant baseline
        + np.exp(-(x_data - 500) / 60)  # severe exponential baseline
        + gaussian(x_data, 100, 25)
        + gaussian(x_data, 200, 50)
        + gaussian(x_data, 100, 75)
    )
    noise = np.random.default_rng(0).normal(0, 0.5, x_data.size)
    y_data = signal + noise

    return y_data


class TestRubberband(ClassificationTester):
    """Class for testing rubberband baseline."""

    func_name = 'rubberband'

    def test_segments_scalar_vs_array(self):
        """Compares scalar and array-like segments in the case they should have the same value."""
        segments = 5
        manual_segments = np.arange(segments + 1) * len(self.x) // segments

        output_1 = self.class_func(self.y, segments=segments)[0]
        output_2 = self.class_func(self.y, segments=manual_segments)[0]

        assert_allclose(output_1, output_2, rtol=1e-12)

    def test_segment_repeats(self):
        """Ensures repeated segments are only counted once."""
        segments_1 = [250, 500, 750]
        segments_2 = [250, 500, 750, 250, 500, 750]
        segments_3 = [0, 250, 500, 750, len(self.x)]

        output_1 = self.class_func(self.y, segments=segments_1)[0]
        output_2 = self.class_func(self.y, segments=segments_2)[0]
        output_3 = self.class_func(self.y, segments=segments_3)[0]

        assert_allclose(output_1, output_2, rtol=1e-12)
        assert_allclose(output_1, output_3, rtol=1e-12)

    def test_incorrect_scalar_segments_fail(self):
        """Ensures scalar values less than 1 or greater than len(x) // 3 fail."""
        max_segments = len(self.x) // 3
        self.class_func(self.y, segments=max_segments)
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=max_segments + 1)
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=0)

    def test_incorrect_array_segments_fail(self):
        """Ensures array values less than 0, greater than len(x), or too small spacing fail."""
        max_segments = len(self.x) // 3
        segments = np.arange(max_segments + 1) * len(self.x) // max_segments
        self.class_func(self.y, segments=segments)
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=segments + 1)
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=segments - 1)

        segments = [10, 15]  # ensure index 15 works before trying failing cases
        self.class_func(self.y, segments=segments)

        segments = [-1, 15]
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=segments)

        segments = [15, len(self.x) + 1]
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=segments)

        segments = [15, 18]
        self.class_func(self.y, segments=segments)

        segments = [15, 17]
        with pytest.raises(ValueError):
            self.class_func(self.y, segments=segments)

    def test_non_sorted_x_fails(self):
        """Ensures that non-monotonically increasing x-values fails."""
        reverse_fitter = self.algorithm_base(self.x[::-1], assume_sorted=True)
        with pytest.raises(ValueError):
            getattr(reverse_fitter, self.func_name)(self.y)

    @pytest.mark.parametrize('lam', [0, None])
    def test_zero_lam_interp(self, lam):
        """Ensures that a None or zero-valued lam gives a linear interpolation."""
        output, params = self.class_func(self.y, lam=lam)
        interp_output = np.interp(self.x, self.x[params['mask']], self.y[params['mask']])
        assert_allclose(output, interp_output, rtol=1e-12, atol=0)

    @pytest.mark.parametrize('lam', [-1, -10])
    def test_negative_lam_fails(self, lam):
        """Ensures that a negative lam value fails."""
        with pytest.raises(ValueError):
            self.class_func(self.y, lam=lam)

    @pytest.mark.parametrize('diff_order', [1, 2])
    @pytest.mark.parametrize('segments', (1, 5, [10, 50]))
    @pytest.mark.parametrize('lam', [0.1, 1])
    def test_smoothing(self, segments, lam, diff_order):
        """Ensures the whittaker smoothing is correct."""
        output, params = self.class_func(
            self.y, segments=segments, lam=lam, diff_order=diff_order,
            smooth_half_window=0
        )

        spline = whittaker_smooth(
            self.y, weights=params['mask'], lam=lam, diff_order=diff_order, check_finite=False
        )
        assert_allclose(output, spline, rtol=1e-10)

    @pytest.mark.parametrize('use_class', (True, False))
    @pytest.mark.parametrize('lam', (0, 1))
    @pytest.mark.parametrize('smooth_half_window', (None, 0, 1))
    def test_unchanged_data(self, use_class, lam, smooth_half_window):
        """Ensures that input data is unchanged by the function."""
        super().test_unchanged_data(
            use_class, lam=lam, smooth_half_window=smooth_half_window
        )

    def test_indexing(self):
        """
        Ensures indexing is handled correctly by the rubberband baseline.

        Addresses issue 29 where the indexing had failed due to the min and max index
        values occuring on the same value.

        """
        data = rubberband_data(self.x)
        self.class_func(data)  # just ensure it runs without issue
