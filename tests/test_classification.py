# -*- coding: utf-8 -*-
"""Tests for pybaselines.classification.

@author: Donald Erb
Created on July 3, 2021

"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import scipy
from scipy.signal import cwt

from pybaselines import classification
from pybaselines.utils import ParameterWarning

from .conftest import AlgorithmTester, get_data
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
    y = y_scale * np.sin(x) + np.random.normal(0, 0.2, x.size)
    # only compare within [half_window:-half_window] since the calculation
    # can have slightly different values at the edges
    compare_slice = slice(half_window, -half_window)

    actual_rolled_std = _nieve_rolling_std(y, half_window, ddof)
    calc_rolled_std = classification._rolling_std(
        np.pad(y, half_window, 'reflect'), half_window, ddof
    )[half_window:-half_window]

    assert_allclose(calc_rolled_std[compare_slice], actual_rolled_std[compare_slice])


@pytest.mark.parametrize(
    'mask_and_expected',
    (
        [[0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0]],
        [[0, 1, 0, 1, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0]],
        [[0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1]]
    )
)
def test_remove_single_points(mask_and_expected):
    """
    Test that _remove_single_points fills holes in binary mask.

    Lone True values should be removed before lone False values, and
    the edges should convert to False unless there are two True values.

    """
    mask, expected_mask = np.asarray(mask_and_expected, bool)
    output_mask = classification._remove_single_points(mask)

    assert_array_equal(expected_mask, output_mask)


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

    haar_cwt = cwt(y, classification._haar, [scale])[0]
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


class TestGolotvin(AlgorithmTester):
    """Class for testing golotvin baseline."""

    func = classification.golotvin

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x, 15, 6)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, self.x, 15, 6, checked_keys=('mask',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        x_list = self.x.tolist()
        self._test_algorithm_list(
            array_args=(self.y, self.x, 15, 6), list_args=(y_list, x_list, 15, 6)
        )


class TestDietrich(AlgorithmTester):
    """Class for testing dietrich baseline."""

    func = classification.dietrich

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    @pytest.mark.parametrize('return_coef', (True, False))
    @pytest.mark.parametrize('max_iter', (0, 1, 2))
    def test_output(self, return_coef, max_iter):
        """Ensures that the output has the desired format."""
        param_keys = ['mask']
        if return_coef and max_iter > 0:
            param_keys.append('coef')
        if max_iter > 1:
            param_keys.append('tol_history')
        self._test_output(
            self.y, self.y, checked_keys=param_keys, return_coef=return_coef, max_iter=max_iter
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))

    def test_output_coefs(self):
        """Ensures the output coefficients can correctly reproduce the baseline."""
        baseline, params = self._call_func(self.y, self.x, return_coef=True)
        recreated_poly = np.polynomial.Polynomial(params['coef'])(self.x)

        assert_allclose(baseline, recreated_poly)

    def test_tol_history(self):
        """Ensures the 'tol_history' item in the parameter output is correct."""
        max_iter = 5
        _, params = self._call_func(self.y, max_iter=max_iter, tol=-1)

        assert params['tol_history'].size == max_iter - 1


class TestStdDistribution(AlgorithmTester):
    """Class for testing std_distribution baseline."""

    func = classification.std_distribution

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('mask',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestFastChrom(AlgorithmTester):
    """Class for testing fastchrom baseline."""

    func = classification.fastchrom

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('mask',))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestCwtBR(AlgorithmTester):
    """Class for testing cwt_br baseline."""

    func = classification.cwt_br

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    @pytest.mark.parametrize('scales', (None, np.arange(3, 20)))
    def test_output(self, scales):
        """Ensures that the output has the desired format."""
        self._test_output(
            self.y, self.y, scales=scales, checked_keys=('mask', 'tol_history', 'best_scale')
        )

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))


class TestFabc(AlgorithmTester):
    """Class for testing fabc baseline."""

    func = classification.fabc

    def test_unchanged_data(self, data_fixture):
        """Ensures that input data is unchanged by the function."""
        x, y = get_data()
        self._test_unchanged_data(data_fixture, y, x, y, x)

    def test_output(self):
        """Ensures that the output has the desired format."""
        self._test_output(self.y, self.y, checked_keys=('mask', 'weights'))

    def test_list_input(self):
        """Ensures that function works the same for both array and list inputs."""
        y_list = self.y.tolist()
        self._test_algorithm_list(array_args=(self.y,), list_args=(y_list,))
