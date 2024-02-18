# -*- coding: utf-8 -*-
"""Techniques that rely on classifying peak and/or baseline segments for fitting baselines.

Created on July 3, 2021
@author: Donald Erb


Several functions were adapted from SciPy
(https://github.com/scipy/scipy, accessed December 28, 2023), which was
licensed under the BSD-3-Clause below.

Copyright (c) 2001-2002 Enthought, Inc.  2003-2023, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from math import ceil
import warnings

import numpy as np
from scipy.ndimage import (
    binary_dilation, binary_erosion, binary_opening, grey_dilation, grey_erosion, uniform_filter1d
)
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.spatial import ConvexHull

from ._algorithm_setup import _Algorithm, _class_wrapper
from ._compat import jit, trapezoid
from ._validation import _check_scalar
from .utils import (
    _MIN_FLOAT, ParameterWarning, _convert_coef, _interp_inplace, gaussian, optimize_window,
    pad_edges, relative_difference
)


class _Classification(_Algorithm):
    """A base class for all classification algorithms."""

    @_Algorithm._register(sort_keys=('mask',))
    def golotvin(self, data, half_window=None, num_std=2.0, sections=32, smooth_half_window=None,
                 interp_half_window=5, weights=None, min_length=2, **pad_kwargs):
        """
        Golotvin's method for identifying baseline regions.

        Divides the data into sections and takes the minimum standard deviation of all
        sections as the noise standard deviation for the entire data. Then classifies any point
        where the rolling max minus min is less than ``num_std * noise standard deviation``
        as belonging to the baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window to use for the rolling maximum and rolling minimum calculations.
            Should be approximately equal to the full-width-at-half-maximum of the peaks or
            features in the data. Default is None, which will use half of the value from
            :func:`.optimize_window`, which is not always a good value, but at least scales
            with the number of data points and gives a starting point for tuning the parameter.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Higher values
            will assign more points as baseline. Default is 3.0.
        sections : int, optional
            The number of sections to divide the input data into for finding the minimum
            standard deviation.
        smooth_half_window : int, optional
            The half window to use for smoothing the interpolated baseline with a moving average.
            Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
        interp_half_window : int, optional
            When interpolating between baseline segments, will use the average of
            ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
            the index of the peak start or end, to fit the linear segment. Default is 5.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
            Default is 2, which only removes lone baseline points.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from the moving average smoothing.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.

        References
        ----------
        Golotvin, S., et al. Improved Baseline Recognition and Modeling of
        FT NMR Spectra. Journal of Magnetic Resonance. 2000, 146, 122-125.

        """
        y, weight_array = self._setup_classification(data, weights)
        if half_window is None:
            # optimize_window(y) / 2 gives an "okay" estimate that at least scales
            # with data size
            half_window = ceil(optimize_window(y) / 2)
        if smooth_half_window is None:
            smooth_half_window = half_window
        min_sigma = np.inf
        for i in range(sections):
            # use ddof=1 since sampling subsets of the data
            min_sigma = min(
                min_sigma,
                np.std(y[i * self._len // sections:((i + 1) * self._len) // sections], ddof=1)
            )

        mask = (
            grey_dilation(y, 2 * half_window + 1) - grey_erosion(y, 2 * half_window + 1)
        ) < num_std * min_sigma
        mask = _refine_mask(mask, min_length)
        np.logical_and(mask, weight_array, out=mask)

        rough_baseline = _averaged_interp(self.x, y, mask, interp_half_window)
        baseline = uniform_filter1d(
            pad_edges(rough_baseline, smooth_half_window, **pad_kwargs),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]

        return baseline, {'mask': mask}

    @_Algorithm._register(sort_keys=('mask',))
    def dietrich(self, data, smooth_half_window=None, num_std=3.0, interp_half_window=5,
                 poly_order=5, max_iter=50, tol=1e-3, weights=None, return_coef=False,
                 min_length=2, **pad_kwargs):
        """
        Dietrich's method for identifying baseline regions.

        Calculates the power spectrum of the data as the squared derivative of the data.
        Then baseline points are identified by iteratively removing points where the mean
        of the power spectrum is less than `num_std` times the standard deviation of the
        power spectrum.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        smooth_half_window : int, optional
            The half window to use for smoothing the input data with a moving average.
            Default is None, which will use N / 256. Set to 0 to not smooth the data.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Higher values
            will assign more points as baseline. Default is 3.0.
        interp_half_window : int, optional
            When interpolating between baseline segments, will use the average of
            ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
            the index of the peak start or end, to fit the linear segment. Default is 5.
        poly_order : int, optional
            The polynomial order for fitting the identified baseline. Default is 5.
        max_iter : int, optional
            The maximum number of iterations for fitting a polynomial to the identified
            baseline. If `max_iter` is 0, the returned baseline will be just the linear
            interpolation of the baseline segments. Default is 50.
        tol : float, optional
            The exit criteria for fitting a polynomial to the identified baseline points.
            Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        return_coef : bool, optional
            If True, will convert the polynomial coefficients for the fit baseline to
            a form that fits the input `x_data` and return them in the params dictionary.
            Default is False, since the conversion takes time.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
            Default is 2, which only removes lone baseline points.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from smoothing.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.
            * 'coef': numpy.ndarray, shape (poly_order,)
                Only if `return_coef` is True and `max_iter` is greater than 0. The array
                of polynomial coefficients for the baseline, in increasing order. Can be
                used to create a polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.
            * 'tol_history': numpy.ndarray
                Only if `max_iter` is greater than 1. An array containing the calculated
                tolerance values for each iteration. The length of the array is the number
                of iterations completed. If the last value in the array is greater than
                the input `tol` value, then the function did not converge.

        Notes
        -----
        When choosing parameters, first choose a `smooth_half_window` that appropriately
        smooths the data, and then reduce `num_std` until no peak regions are included in
        the baseline. If no value of `num_std` works, change `smooth_half_window` and repeat.

        If `max_iter` is 0, the baseline is simply a linear interpolation of the identified
        baseline points. Otherwise, a polynomial is iteratively fit through the baseline
        points, and the interpolated sections are replaced each iteration with the polynomial
        fit.

        References
        ----------
        Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
        Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

        """
        y, weight_array = self._setup_classification(data, weights)
        if smooth_half_window is None:
            smooth_half_window = ceil(self._len / 256)
        smooth_y = uniform_filter1d(
            pad_edges(y, smooth_half_window, **pad_kwargs),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]
        power = np.gradient(smooth_y)**2
        mask = _refine_mask(_iter_threshold(power, num_std), min_length)
        np.logical_and(mask, weight_array, out=mask)
        rough_baseline = _averaged_interp(self.x, y, mask, interp_half_window)

        params = {'mask': mask}
        baseline = rough_baseline
        if max_iter > 0:
            *_, pseudo_inverse = self._setup_polynomial(
                y, poly_order=poly_order, calc_vander=True, calc_pinv=True
            )
            old_coef = coef = pseudo_inverse @ rough_baseline
            baseline = self.vandermonde @ coef
            if max_iter > 1:
                tol_history = np.empty(max_iter - 1)
                for i in range(max_iter - 1):
                    rough_baseline[mask] = baseline[mask]
                    coef = pseudo_inverse @ rough_baseline
                    baseline = self.vandermonde @ coef
                    calc_difference = relative_difference(old_coef, coef)
                    tol_history[i] = calc_difference
                    if calc_difference < tol:
                        break
                    old_coef = coef
                params['tol_history'] = tol_history[:i + 1]

            if return_coef:
                params['coef'] = _convert_coef(coef, self.x_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('mask',))
    def std_distribution(self, data, half_window=None, interp_half_window=5,
                         fill_half_window=3, num_std=1.1, smooth_half_window=None,
                         weights=None, **pad_kwargs):
        """
        Identifies baseline segments by analyzing the rolling standard deviation distribution.

        The rolling standard deviations are split into two distributions, with the smaller
        distribution assigned to noise. Baseline points are then identified as any point
        where the rolling standard deviation is less than a multiple of the median of the
        noise's standard deviation distribution.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window to use for the rolling standard deviation calculation. Should
            be approximately equal to the full-width-at-half-maximum of the peaks or features
            in the data. Default is None, which will use half of the value from
            :func:`.optimize_window`, which is not always a good value, but at least scales
            with the number of data points and gives a starting point for tuning the parameter.
        interp_half_window : int, optional
            When interpolating between baseline segments, will use the average of
            ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
            the index of the peak start or end, to fit the linear segment. Default is 5.
        fill_half_window : int, optional
            When a point is identified as a peak point, all points +- `fill_half_window`
            are likewise set as peak points. Default is 3.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Higher values
            will assign more points as baseline. Default is 1.1.
        smooth_half_window : int, optional
            The half window to use for smoothing the interpolated baseline with a moving average.
            Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from the moving average smoothing.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.

        References
        ----------
        Wang, K.C., et al. Distribution-Based Classification Method for Baseline
        Correction of Metabolomic 1D Proton Nuclear Magnetic Resonance Spectra.
        Analytical Chemistry. 2013, 85, 1231-1239.

        """
        y, weight_array = self._setup_classification(data, weights)
        if half_window is None:
            # optimize_window(y) / 2 gives an "okay" estimate that at least scales
            # with data size
            half_window = ceil(optimize_window(y) / 2)
        if smooth_half_window is None:
            smooth_half_window = half_window

        # use dof=1 since sampling a subset of the data
        std = _padded_rolling_std(y, half_window, 1)
        median = np.median(std)
        median_2 = np.median(std[std < 2 * median])  # TODO make the 2 an input?
        while median_2 / median < 0.999:  # TODO make the 0.999 an input?
            median = median_2
            median_2 = np.median(std[std < 2 * median])
        noise_std = median_2

        # use ~ to convert from peak==1, baseline==0 to peak==0, baseline==1; if done before,
        # would have to do ~binary_dilation(~mask) or binary_erosion(np.hstack((1, mask, 1))[1:-1]
        mask = np.logical_and(
            ~binary_dilation(std > num_std * noise_std, np.ones(2 * fill_half_window + 1)),
            weight_array
        )

        rough_baseline = _averaged_interp(self.x, y, mask, interp_half_window)

        baseline = uniform_filter1d(
            pad_edges(rough_baseline, smooth_half_window, **pad_kwargs),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]

        return baseline, {'mask': mask}

    @_Algorithm._register(sort_keys=('mask',))
    def fastchrom(self, data, half_window=None, threshold=None, min_fwhm=None,
                  interp_half_window=5, smooth_half_window=None, weights=None,
                  max_iter=100, min_length=2, **pad_kwargs):
        """
        Identifies baseline segments by thresholding the rolling standard deviation distribution.

        Baseline points are identified as any point where the rolling standard deviation
        is less than the specified threshold. Peak regions are iteratively interpolated
        until the baseline is below the data.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window to use for the rolling standard deviation calculation. Should
            be approximately equal to the full-width-at-half-maximum of the peaks or features
            in the data. Default is None, which will use half of the value from
            :func:`.optimize_window`, which is not always a good value, but at least scales
            with the number of data points and gives a starting point for tuning the parameter.
        threshold : float of Callable, optional
            All points in the rolling standard deviation below `threshold` will be considered
            as baseline. Higher values will assign more points as baseline. Default is None,
            which will set the threshold as the 15th percentile of the rolling standard
            deviation. If `threshold` is Callable, it should take the rolling standard deviation
            as the only argument and output a float.
        min_fwhm : int, optional
            After creating the interpolated baseline, any region where the baseline
            is greater than the data for `min_fwhm` consecutive points will have an additional
            baseline point added and reinterpolated. Should be set to approximately the
            index-based full-width-at-half-maximum of the smallest peak. Default is None,
            which uses 2 * `half_window`.
        interp_half_window : int, optional
            When interpolating between baseline segments, will use the average of
            ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
            the index of the peak start or end, to fit the linear segment. Default is 5.
        smooth_half_window : int, optional
            The half window to use for smoothing the interpolated baseline with a moving average.
            Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        max_iter : int, optional
            The maximum number of iterations to attempt to fill in regions where the baseline
            is greater than the input data. Default is 100.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
            Default is 2, which only removes lone baseline points.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from the moving average smoothing.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.

        Notes
        -----
        Only covers the baseline correction from FastChrom, not its peak finding and peak
        grouping capabilities.

        References
        ----------
        Johnsen, L., et al. An automated method for baseline correction, peak finding
        and peak grouping in chromatographic data. Analyst. 2013, 138, 3502-3511.

        """
        y, weight_array = self._setup_classification(data, weights)
        if half_window is None:
            # optimize_window(y) / 2 gives an "okay" estimate that at least scales
            # with data size
            half_window = ceil(optimize_window(y) / 2)
        if smooth_half_window is None:
            smooth_half_window = half_window
        if min_fwhm is None:
            min_fwhm = 2 * half_window

        # use dof=1 since sampling a subset of the data
        std = _padded_rolling_std(y, half_window, 1)
        if threshold is None:
            # scales fairly well with y and gaurantees baseline segments are created;
            # picked 15% since it seems to work better than 10%
            threshold_val = np.percentile(std, 15)
        elif callable(threshold):
            threshold_val = threshold(std)
        else:
            threshold_val = threshold

        # reference did not mention removing single points, but do so anyway to
        # be more thorough
        mask = _refine_mask(std < threshold_val, min_length)
        np.logical_and(mask, weight_array, out=mask)
        rough_baseline = _averaged_interp(self.x, y, mask, interp_half_window)

        mask_sum = mask.sum()
        # only try to fix peak regions if there actually are peak and baseline regions
        if mask_sum and mask_sum != self._len:
            peak_starts, peak_ends = _find_peak_segments(mask)
            for _ in range(max_iter):
                modified_baseline = False
                for start, end in zip(peak_starts, peak_ends):
                    baseline_section = rough_baseline[start:end + 1]
                    data_section = y[start:end + 1]
                    # mask should be baseline_section > data_section, but use the
                    # inverse since _find_peak_segments looks for 0s, not 1s
                    section_mask = baseline_section < data_section
                    seg_starts, seg_ends = _find_peak_segments(section_mask)
                    if np.any(seg_ends - seg_starts > min_fwhm):
                        modified_baseline = True
                        # designate lowest point as baseline
                        # TODO should surrounding points also be classified as baseline?
                        mask[np.argmin(data_section - baseline_section) + start] = 1

                if modified_baseline:
                    # TODO probably faster to just re-interpolate changed sections
                    rough_baseline = _averaged_interp(self.x, y, mask, interp_half_window)
                else:
                    break

        # reference did not discuss smoothing, but include to be consistent with
        # other classification functions
        baseline = uniform_filter1d(
            pad_edges(rough_baseline, smooth_half_window, **pad_kwargs),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]

        return baseline, {'mask': mask}

    @_Algorithm._register(sort_keys=('mask',))
    def cwt_br(self, data, poly_order=5, scales=None, num_std=1.0, min_length=2,
               max_iter=50, tol=1e-3, symmetric=False, weights=None, **pad_kwargs):
        """
        Continuous wavelet transform baseline recognition (CWT-BR) algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int, optional
            The polynomial order for fitting the baseline. Default is 5.
        scales : array-like, optional
            The scales at which to perform the continuous wavelet transform. Default
            is None,
        num_std : float, optional
            The number of standard deviations to include when thresholding. Default
            is 1.0.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
            Default is 2, which only removes lone baseline points.
        max_iter : int, optional
            The maximum number of iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        symmetric : bool, optional
            When fitting the identified baseline points with a polynomial, if `symmetric`
            is False (default), will add any point `i` as a baseline point where the fit
            polynomial is greater than the input data for ``N/100`` consecutive points on both
            sides of point `i`. If `symmetric` is True, then it means that both positive and
            negative peaks exist and baseline points are not modified during the polynomial fitting.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from convolution for the
            continuous wavelet transform.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'best_scale' : scalar
                The scale at which the Shannon entropy of the continuous wavelet transform
                of the data is at a minimum.

        Notes
        -----
        Uses the standard deviation for determining outliers during polynomial fitting rather
        than the standard error as used in the reference since the number of standard errors
        to include when thresholding varies with data size while the number of standard
        deviations is independent of data size.

        References
        ----------
        Bertinetto, C., et al. Automatic Baseline Recognition for the Correction of Large
        Sets of Spectra Using Continuous Wavelet Transform and Iterative Fitting. Applied
        Spectroscopy, 2014, 68(2), 155-164.

        """
        y, weight_array = self._setup_classification(data, weights)
        self._setup_polynomial(y, weight_array, poly_order=poly_order, calc_vander=True)
        # scale y between -1 and 1 so that the residual fit is more numerically stable
        y_domain = np.polynomial.polyutils.getdomain(y)
        y = np.polynomial.polyutils.mapdomain(y, y_domain, np.array([-1., 1.]))
        if scales is None:
            # avoid low scales since their cwt is fairly noisy
            min_scale = max(2, self._len // 500)
            max_scale = self._len // 4
            scales = range(min_scale, max_scale)
        else:
            scales = np.atleast_1d(scales).reshape(-1)
            max_scale = scales.max()

        shannon_old = -np.inf
        shannon_current = -np.inf
        half_window = max_scale * 2  # TODO is x2 enough padding to prevent edge effects from cwt?
        padded_y = pad_edges(y, half_window, **pad_kwargs)
        for scale in scales:
            wavelet_cwt = _cwt(padded_y, _ricker, [scale])[0, half_window:-half_window]
            abs_wavelet = np.abs(wavelet_cwt)
            inner = abs_wavelet / abs_wavelet.sum(axis=0)
            # was not stated in the reference to use abs(wavelet) for the Shannon entropy,
            # but otherwise the Shannon entropy vs wavelet scale curve does not look like
            # Figure 2 in the reference; masking out non-positive values also gives an
            # incorrect entropy curve
            shannon_entropy = -np.sum(inner * np.log(inner + _MIN_FLOAT), 0)
            if shannon_current < shannon_old and shannon_entropy > shannon_current:
                break
            shannon_old = shannon_current
            shannon_current = shannon_entropy

        best_scale_ptp_multiple = 8 * abs(wavelet_cwt.max() - wavelet_cwt.min())
        num_bins = 200
        histogram, bin_edges = np.histogram(wavelet_cwt, num_bins)
        bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        fit_params = [histogram.max(), np.log10(0.2 * np.std(wavelet_cwt))]
        # use 10**sigma so that sigma is not actually bounded
        gaussian_fit = lambda x, height, sigma: gaussian(x, height, 0, 10**sigma)
        # TODO should the number of iterations, the height cutoff for the histogram,
        # and the exit tol be parameters? The number of iterations is never greater than
        # 2 or 3, matching the reference. The height maybe should be since the masking
        # depends on the histogram scale
        dilation_structure = np.ones(5, bool)
        for _ in range(10):
            # dilate the mask to ensure at least five points are included for the fitting
            fit_mask = binary_dilation(histogram > histogram.max() / 5, dilation_structure)
            # histogram[~fit_mask] = 0  TODO use this instead? does it help fitting?
            fit_params = curve_fit(
                gaussian_fit, bins[fit_mask], histogram[fit_mask], fit_params,
                check_finite=False
            )[0]
            sigma_opt = 10**fit_params[1]

            new_num_bins = ceil(best_scale_ptp_multiple / sigma_opt)
            if relative_difference(num_bins, new_num_bins) < 0.05:
                break
            num_bins = new_num_bins
            histogram, bin_edges = np.histogram(wavelet_cwt, num_bins)
            bins = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        gaussian_mask = np.abs(bins) < 3 * sigma_opt
        gaus_area = trapezoid(histogram[gaussian_mask], bins[gaussian_mask])
        num_sigma = 0.6 + 10 * ((trapezoid(histogram, bins) - gaus_area) / gaus_area)

        wavelet_mask = _refine_mask(abs_wavelet < num_sigma * sigma_opt, min_length)
        np.logical_and(wavelet_mask, weight_array, out=wavelet_mask)

        check_window = np.ones(2 * (self._len // 200) + 1, bool)  # TODO make window size a param?
        baseline_old = y
        mask = wavelet_mask.copy()
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            coef = np.linalg.lstsq(self.vandermonde[mask], y[mask], None)[0]
            baseline = self.vandermonde @ coef
            residual = y - baseline
            mask[residual > num_std * np.std(residual)] = False

            # TODO is this necessary? It improves fits where the initial fit didn't
            # include enough points, but ensures that negative peaks are not allowed;
            # maybe make it a param called symmetric, like for mixture_model, and only
            # do if not symmetric; also probably only need to do it the first iteration
            # since after that the masking above will not remove negative residuals
            coef = np.linalg.lstsq(self.vandermonde[mask], y[mask], None)[0]
            baseline = self.vandermonde @ coef

            calc_difference = relative_difference(baseline_old, baseline)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            baseline_old = baseline
            if not symmetric:
                np.logical_or(mask, binary_erosion(y < baseline, check_window), out=mask)

        # TODO should include wavelet_mask in params; maybe called 'initial_mask'?
        params = {
            'mask': mask, 'tol_history': tol_history[:i + 1], 'best_scale': scale
        }

        baseline = np.polynomial.polyutils.mapdomain(baseline, np.array([-1., 1.]), y_domain)

        return baseline, params

    @_Algorithm._register(sort_keys=('mask', 'weights'))
    def fabc(self, data, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2,
             weights=None, weights_as_mask=False, **pad_kwargs):
        """
        Fully automatic baseline correction (fabc).

        Similar to Dietrich's method, except that the derivative is estimated using a
        continuous wavelet transform and the baseline is calculated using Whittaker
        smoothing through the identified baseline points.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
        scale : int, optional
            The scale at which to calculate the continuous wavelet transform. Should be
            approximately equal to the index-based full-width-at-half-maximum of the peaks
            or features in the data. Default is None, which will use half of the value from
            :func:`.optimize_window`, which is not always a good value, but at least scales
            with the number of data points and gives a starting point for tuning the parameter.
        num_std : float, optional
            The number of standard deviations to include when thresholding. Higher values
            will assign more points as baseline. Default is 3.0.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        min_length : int, optional
            Any region of consecutive baseline points less than `min_length` is considered
            to be a false positive and all points in the region are converted to peak points.
            A higher `min_length` ensures less points are falsely assigned as baseline points.
            Default is 2, which only removes lone baseline points.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        weights_as_mask : bool, optional
            If True, signifies that the input `weights` is the mask to use for fitting,
            which skips the continuous wavelet calculation and just smooths the input data.
            Default is False.
        **pad_kwargs
            Additional keyword arguments to pass to :func:`.pad_edges` for padding
            the edges of the data to prevent edge effects from convolution for the
            continuous wavelet transform.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.
            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.

        Notes
        -----
        The classification of baseline points is similar to :meth:`~Baseline.dietrich`, except that
        this method approximates the first derivative using a continous wavelet transform
        with the Haar wavelet, which is more robust than the numerical derivative in
        Dietrich's method.

        References
        ----------
        Cobas, J., et al. A new general-purpose fully automatic baseline-correction
        procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1),
        145-151.

        """
        if weights_as_mask:
            y, whittaker_weights = self._setup_whittaker(data, lam, diff_order, weights)
            mask = whittaker_weights.astype(bool)
        else:
            y, weight_array = self._setup_classification(data, weights)
            if scale is None:
                # optimize_window(y) / 2 gives an "okay" estimate that at least scales
                # with data size
                scale = ceil(optimize_window(y) / 2)
            # TODO is 2*scale enough padding to prevent edge effects from cwt?
            half_window = scale * 2
            wavelet_cwt = _cwt(pad_edges(y, half_window, **pad_kwargs), _haar, [scale])
            power = wavelet_cwt[0, half_window:-half_window]**2

            mask = _refine_mask(_iter_threshold(power, num_std), min_length)
            np.logical_and(mask, weight_array, out=mask)

            _, whittaker_weights = self._setup_whittaker(y, lam, diff_order, mask)
            if self._sort_order is not None:
                whittaker_weights = whittaker_weights[self._inverted_order]

        baseline = self.whittaker_system.solve(
            self.whittaker_system.add_diagonal(whittaker_weights), whittaker_weights * y,
            overwrite_b=True, overwrite_ab=True
        )
        params = {'mask': mask, 'weights': whittaker_weights}

        return baseline, params

    @_Algorithm._register(sort_keys=('mask',))
    def rubberband(self, data, segments=1, lam=None, diff_order=2, weights=None,
                   smooth_half_window=None, **pad_kwargs):
        """
        Identifies baseline points by fitting a convex hull to the bottom of the data.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        segments : int or array-like[int], optional
            Used to fit multiple convex hulls to the data to negate the effects of
            concave data. If the input is an integer, it sets the number of equally sized
            segments the data will be split into. If the input is an array-like, each integer
            in the array will be the index that splits two segments, which allows
            constructing unequally sized segments. Default is 1, which fits a single convex
            hull to the data.
        lam : float or None, optional
            The smoothing parameter for interpolating the baseline points using
            Whittaker smoothing. Set to 0 or None to use linear interpolation instead.
            Default is None, which does not smooth.
        diff_order : int, optional
            The order of the differential matrix if using Whittaker smoothing. Must
            be greater than 0. Default is 2 (second order differential matrix).
            Typical values are 2 or 1.
        weights : array-like, shape (N,), optional
            The weighting array, used to override the function's baseline identification
            to designate peak points. Only elements with 0 or False values will have
            an effect; all non-zero values are considered potential baseline points. If None
            (default), then will be an array with size equal to N and all values set to 1.
        smooth_half_window : int or None, optional
            The half window to use for smoothing the input data with a moving average
            before calculating the convex hull, which gives much better results for
            noisy data. Set to None (default) or 0 to not smooth the data.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'mask': numpy.ndarray, shape (N,)
                The boolean array designating baseline points as True and peak points
                as False.

        Raises
        ------
        ValueError
            Raised if the number of segments per window for the fitting is less than
            `poly_order` + 1 or greater than the total number of points, or if the
            values in `self.x` are not strictly increasing.

        """
        sections, scalar_sections = _check_scalar(segments, None, coerce_0d=False, dtype=np.intp)
        if scalar_sections and (sections < 1 or self._len / sections < 3):
            raise ValueError(
                f'There must be between 1 and {self._len // 3} segments for the rubberband fit'
            )
        elif not scalar_sections and (np.any(sections < 0) or np.any(sections > self._len)):
            raise ValueError(
                f'Segment indices must be between 0 and {self._len} for the rubberband fit'
            )

        if np.any(self.x[1:] < self.x[:-1]):
            raise ValueError('x must be strictly increasing')

        y, weight_array = self._setup_classification(data, weights)
        if smooth_half_window is not None and smooth_half_window != 0:
            y = self._setup_smooth(y, smooth_half_window, allow_zero=False, **pad_kwargs)
            y = uniform_filter1d(
                y, 2 * smooth_half_window + 1
            )[smooth_half_window:-smooth_half_window]

        if scalar_sections:
            total_sections = np.arange(sections + 1, dtype=np.intp) * self._len // sections
        else:
            total_sections = np.concatenate(([0], sections, [self._len]))
            # np.unique already sorts so do not need to check order
            total_sections = np.unique(total_sections)
            for i, section in enumerate(total_sections[:-1]):
                if total_sections[i + 1] - section < 3:
                    raise ValueError('Each segment must have at least 3 points.')

        hull_data = np.vstack((self.x, y)).T
        total_vertices = []
        for i, left_idx in enumerate(total_sections[:-1]):
            vertices = ConvexHull(hull_data[left_idx:total_sections[i + 1]]).vertices
            min_idx = vertices.argmin()
            max_idx = vertices.argmax() + 1
            if max_idx < min_idx:
                vertices = np.concatenate((vertices[min_idx:], vertices[:max_idx]))
            else:
                vertices = vertices[min_idx:max_idx]
            total_vertices.extend(vertices + left_idx)

        mask = np.zeros(self._len, dtype=bool)
        mask[np.unique(total_vertices)] = True
        np.logical_and(mask, weight_array, out=mask)
        if lam is not None and lam != 0:
            self._setup_whittaker(y, lam, diff_order, mask)
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(mask), mask * y,
                overwrite_b=True, overwrite_ab=True
            )
        else:
            baseline = np.interp(self.x, self.x[mask], y[mask])

        return baseline, {'mask': mask}


_classification_wrapper = _class_wrapper(_Classification)


def _refine_mask(mask, min_length=2):
    """
    Removes small consecutive True values and lone False values from a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The boolean array designating baseline points as True and peak points as False.
    min_length : int, optional
        The minimum consecutive length of True values needed for a section to remain True.
        Lengths of True values less than `min_length` are converted to False. Default is
        2, which removes all lone True values.

    Returns
    -------
    numpy.ndarray
        The input mask after removing lone True and False values.

    Notes
    -----
    Removes the lone True values first since True values designate the baseline.
    That way, the approach is more conservative with assigning baseline points.

    Examples
    --------
    >>> mask = np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1])
    >>> _remove_single_points(mask, 3).astype(int)
    array([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> _remove_single_points(mask, 5).astype(int)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    """
    min_length = max(min_length, 1)  # has to be at least 1 for the binary opening
    half_length = min_length // 2
    # do not use border_value=1 since that automatically makes the borders True and
    # extends the True section by half_window on each side
    output = binary_opening(
        np.pad(mask, half_length, 'constant', constant_values=True), np.ones(min_length, bool)
    )[half_length:len(mask) + half_length]

    # convert lone False values to True
    np.logical_or(output, binary_erosion(output, [1, 0, 1]), out=output)
    # TODO should there be an erosion step here, using another parameter (erode_hw)?
    # that way, can control both the minimum length and then remove edges of baselines
    # independently, allowing more control over the output mask
    return output


def _find_peak_segments(mask):
    """
    Identifies the peak starts and ends from a boolean mask.

    Parameters
    ----------
    mask : numpy.ndarray
        The boolean mask with peak points as 0 or False and baseline points
        as 1 or True.

    Returns
    -------
    peak_starts : numpy.ndarray
        The array identifying the indices where each peak begins.
    peak_ends : numpy.ndarray
        The array identifying the indices where each peak ends.

    """
    extended_mask = np.concatenate(([True], mask, [True]))
    peak_starts = extended_mask[1:-1] < extended_mask[:-2]
    peak_starts = np.flatnonzero(peak_starts)
    if len(peak_starts):
        peak_starts[1 if peak_starts[0] == 0 else 0:] -= 1

    peak_ends = extended_mask[1:-1] < extended_mask[2:]
    peak_ends = np.flatnonzero(peak_ends)
    if len(peak_ends):
        peak_ends[:-1 if peak_ends[-1] == mask.shape[0] - 1 else None] += 1

    return peak_starts, peak_ends


def _averaged_interp(x, y, mask, interp_half_window=0):
    """
    Averages each anchor point and then interpolates between segments.

    Parameters
    ----------
    x : numpy.ndarray
        The x-values.
    y : numpy.ndarray
        The y-values.
    mask : numpy.ndarray
        A boolean array with 0 or False designating peak points and 1 or True
        designating baseline points.
    interp_half_window : int, optional
        The half-window to use for averaging around the anchor points before interpolating.
        Default is 0, which uses just the anchor point value.

    Returns
    -------
    output : numpy.ndarray
        A copy of the input `y` array with peak values in `mask` calculcated using linear
        interpolation.

    """
    output = y.copy()
    mask_sum = mask.sum()
    if not mask_sum:  # all points belong to peaks
        # will just interpolate between first and last points
        warnings.warn('there were no baseline points found', ParameterWarning, stacklevel=2)
    elif mask_sum == mask.shape[0]:  # all points belong to baseline
        warnings.warn('there were no peak points found', ParameterWarning, stacklevel=2)
        return output

    peak_starts, peak_ends = _find_peak_segments(mask)
    num_y = y.shape[0]
    for start, end in zip(peak_starts, peak_ends):
        left_mean = np.mean(
            y[max(0, start - interp_half_window):min(start + interp_half_window + 1, num_y)]
        )
        right_mean = np.mean(
            y[max(0, end - interp_half_window):min(end + interp_half_window + 1, num_y)]
        )
        _interp_inplace(x[start:end + 1], output[start:end + 1], left_mean, right_mean)

    return output


@_classification_wrapper
def golotvin(data, x_data=None, half_window=None, num_std=2.0, sections=32,
             smooth_half_window=None, interp_half_window=5, weights=None, min_length=2,
             **pad_kwargs):
    """
    Golotvin's method for identifying baseline regions.

    Divides the data into sections and takes the minimum standard deviation of all
    sections as the noise standard deviation for the entire data. Then classifies any point
    where the rolling max minus min is less than ``num_std * noise standard deviation``
    as belonging to the baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window to use for the rolling maximum and rolling minimum calculations.
        Should be approximately equal to the full-width-at-half-maximum of the peaks or
        features in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    sections : int, optional
        The number of sections to divide the input data into for finding the minimum
        standard deviation.
    smooth_half_window : int, optional
        The half window to use for smoothing the interpolated baseline with a moving average.
        Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak points.
        A higher `min_length` ensures less points are falsely assigned as baseline points.
        Default is 2, which only removes lone baseline points.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from the moving average smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    References
    ----------
    Golotvin, S., et al. Improved Baseline Recognition and Modeling of
    FT NMR Spectra. Journal of Magnetic Resonance. 2000, 146, 122-125.

    """


def _iter_threshold(power, num_std=3.0):
    """
    Iteratively thresholds a power spectrum based on the mean and standard deviation.

    Any values greater than the mean of the power spectrum plus a multiple of the
    standard deviation are masked out to create a new power spectrum. The process
    is performed iteratively until no further points are masked out.

    Parameters
    ----------
    power : numpy.ndarray, shape (N,)
        The power spectrum to threshold.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Default is 3.0.

    Returns
    -------
    mask : numpy.ndarray, shape (N,)
        The boolean mask with True values where any point in the input power spectrum
        was less than

    References
    ----------
    Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
    Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

    """
    mask = power < np.mean(power) + num_std * np.std(power, ddof=1)
    old_mask = np.ones_like(mask)
    while not np.array_equal(mask, old_mask):
        old_mask = mask
        masked_power = power[mask]
        if masked_power.size < 2:  # need at least 2 points for std calculation
            warnings.warn(
                'not enough baseline points found; "num_std" is likely too low',
                ParameterWarning, stacklevel=2
            )
            break
        mask = power < np.mean(masked_power) + num_std * np.std(masked_power, ddof=1)

    return mask


@_classification_wrapper
def dietrich(data, x_data=None, smooth_half_window=None, num_std=3.0,
             interp_half_window=5, poly_order=5, max_iter=50, tol=1e-3, weights=None,
             return_coef=False, min_length=2, **pad_kwargs):
    """
    Dietrich's method for identifying baseline regions.

    Calculates the power spectrum of the data as the squared derivative of the data.
    Then baseline points are identified by iteratively removing points where the mean
    of the power spectrum is less than `num_std` times the standard deviation of the
    power spectrum.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    smooth_half_window : int, optional
        The half window to use for smoothing the input data with a moving average.
        Default is None, which will use N / 256. Set to 0 to not smooth the data.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    poly_order : int, optional
        The polynomial order for fitting the identified baseline. Default is 5.
    max_iter : int, optional
        The maximum number of iterations for fitting a polynomial to the identified
        baseline. If `max_iter` is 0, the returned baseline will be just the linear
        interpolation of the baseline segments. Default is 50.
    tol : float, optional
        The exit criteria for fitting a polynomial to the identified baseline points.
        Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    return_coef : bool, optional
        If True, will convert the polynomial coefficients for the fit baseline to
        a form that fits the input `x_data` and return them in the params dictionary.
        Default is False, since the conversion takes time.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak points.
        A higher `min_length` ensures less points are falsely assigned as baseline points.
        Default is 2, which only removes lone baseline points.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.
        * 'coef': numpy.ndarray, shape (poly_order,)
            Only if `return_coef` is True and `max_iter` is greater than 0. The array
            of polynomial coefficients for the baseline, in increasing order. Can be
            used to create a polynomial using :class:`numpy.polynomial.polynomial.Polynomial`.
        * 'tol_history': numpy.ndarray
            Only if `max_iter` is greater than 1. An array containing the calculated
            tolerance values for each iteration. The length of the array is the number
            of iterations completed. If the last value in the array is greater than
            the input `tol` value, then the function did not converge.

    Notes
    -----
    When choosing parameters, first choose a `smooth_half_window` that appropriately
    smooths the data, and then reduce `num_std` until no peak regions are included in
    the baseline. If no value of `num_std` works, change `smooth_half_window` and repeat.

    If `max_iter` is 0, the baseline is simply a linear interpolation of the identified
    baseline points. Otherwise, a polynomial is iteratively fit through the baseline
    points, and the interpolated sections are replaced each iteration with the polynomial
    fit.

    References
    ----------
    Dietrich, W., et al. Fast and Precise Automatic Baseline Correction of One- and
    Two-Dimensional NMR Spectra. Journal of Magnetic Resonance. 1991, 91, 1-11.

    """


@jit(nopython=True, cache=True)
def _rolling_std(data, half_window, ddof):
    """
    Computes the rolling standard deviation of an array.

    Parameters
    ----------
    data : numpy.ndarray
        The array for the calculation. Should be padded on the left and right
        edges by `half_window`.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int
        The delta degrees of freedom for the calculation. Usually 0 (numpy default) or 1.

    Returns
    -------
    numpy.ndarray
        The array of the rolling standard deviation for each window.

    Notes
    -----
    This implementation is a version of Welford's method [1]_, modified for a
    fixed-length window [2]_. It is slightly modified from the version in [2]_
    in that it assumes the data is padded on the left and right. Other deviations
    from [2]_ are noted within the function.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    .. [2] Chmielowiec, A. Algorithm for error-free determination of the variance of all
           contiguous subsequences and fixed-length contiguous subsequences for a sequence
           of industrial measurement data. Computational Statistics. 2021, 1-28.

    """
    window_size = half_window * 2 + 1
    num_y = data.shape[0]
    squared_diff = np.zeros(num_y)
    mean = data[0]
    # fill the first window
    for i in range(1, window_size):
        val = data[i]
        size_factor = i / (i + 1)
        squared_diff[i] = squared_diff[i - 1] + 2 * size_factor * (val - mean)**2
        mean = mean * size_factor + val / (i + 1)
    # at this point, mean == np.mean(data[:window_size])

    # update squared_diff[half_window] with squared_diff[window_size - 1] / 2; if
    # this isn't done, all values within [half_window:-half_window] in the output are
    # off; no idea why... but it works
    squared_diff[half_window] = squared_diff[window_size - 1] / 2
    for j in range(half_window + 1, num_y - half_window):
        old_val = data[j - half_window - 1]
        new_val = data[j + half_window]
        val_diff = new_val - old_val  # reference divided by window_size here

        new_mean = mean + val_diff / window_size
        squared_diff[j] = squared_diff[j - 1] + val_diff * (old_val + new_val - mean - new_mean)
        mean = new_mean

    # empty the last half-window
    # TODO need to double-check this; not high priority since last half-window
    # is discarded currently
    size = window_size
    for k in range(num_y - half_window + 1, num_y):
        val = data[k]
        size_factor = size / (size - 1)
        squared_diff[k] = squared_diff[k - 1] + 2 * size_factor * (val - mean)**2
        mean = mean * size_factor + val / (size - 1)
        size -= 1

    return np.sqrt(squared_diff / (window_size - ddof))


def _padded_rolling_std(data, half_window, ddof=0):
    """
    Convenience function that pads data before performing rolling standard deviation calculation.

    Parameters
    ----------
    data : array-like
        The array for the calculation.
    half_window : int
        The half-window the rolling calculation. The full number of points for each
        window is ``half_window * 2 + 1``.
    ddof : int, optional
        The delta degrees of freedom for the calculation. Default is 0.

    Returns
    -------
    numpy.ndarray
        The array of the rolling standard deviation for each window.

    Notes
    -----
    Reflect the data since the standard deviation calculation requires noisy data to work.

    """
    padded_data = np.pad(data, half_window, 'reflect')
    rolling_std = _rolling_std(padded_data, half_window, ddof)[half_window:-half_window]
    return rolling_std


@_classification_wrapper
def std_distribution(data, x_data=None, half_window=None, interp_half_window=5,
                     fill_half_window=3, num_std=1.1, smooth_half_window=None,
                     weights=None, **pad_kwargs):
    """
    Identifies baseline segments by analyzing the rolling standard deviation distribution.

    The rolling standard deviations are split into two distributions, with the smaller
    distribution assigned to noise. Baseline points are then identified as any point
    where the rolling standard deviation is less than a multiple of the median of the
    noise's standard deviation distribution.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window to use for the rolling standard deviation calculation. Should
        be approximately equal to the full-width-at-half-maximum of the peaks or features
        in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    fill_half_window : int, optional
        When a point is identified as a peak point, all points +- `fill_half_window`
        are likewise set as peak points. Default is 3.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 1.1.
    smooth_half_window : int, optional
        The half window to use for smoothing the interpolated baseline with a moving average.
        Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from the moving average smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    References
    ----------
    Wang, K.C., et al. Distribution-Based Classification Method for Baseline
    Correction of Metabolomic 1D Proton Nuclear Magnetic Resonance Spectra.
    Analytical Chemistry. 2013, 85, 1231-1239.

    """


@_classification_wrapper
def fastchrom(data, x_data=None, half_window=None, threshold=None, min_fwhm=None,
              interp_half_window=5, smooth_half_window=None, weights=None,
              max_iter=100, min_length=2, **pad_kwargs):
    """
    Identifies baseline segments by thresholding the rolling standard deviation distribution.

    Baseline points are identified as any point where the rolling standard deviation
    is less than the specified threshold. Peak regions are iteratively interpolated
    until the baseline is below the data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    half_window : int, optional
        The half-window to use for the rolling standard deviation calculation. Should
        be approximately equal to the full-width-at-half-maximum of the peaks or features
        in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    threshold : float of Callable, optional
        All points in the rolling standard deviation below `threshold` will be considered
        as baseline. Higher values will assign more points as baseline. Default is None,
        which will set the threshold as the 15th percentile of the rolling standard
        deviation. If `threshold` is Callable, it should take the rolling standard deviation
        as the only argument and output a float.
    min_fwhm : int, optional
        After creating the interpolated baseline, any region where the baseline
        is greater than the data for `min_fwhm` consecutive points will have an additional
        baseline point added and reinterpolated. Should be set to approximately the
        index-based full-width-at-half-maximum of the smallest peak. Default is None,
        which uses 2 * `half_window`.
    interp_half_window : int, optional
        When interpolating between baseline segments, will use the average of
        ``data[i-interp_half_window:i+interp_half_window+1]``, where `i` is
        the index of the peak start or end, to fit the linear segment. Default is 5.
    smooth_half_window : int, optional
        The half window to use for smoothing the interpolated baseline with a moving average.
        Default is None, which will use `half_window`. Set to 0 to not smooth the baseline.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    max_iter : int, optional
        The maximum number of iterations to attempt to fill in regions where the baseline
        is greater than the input data. Default is 100.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak points.
        A higher `min_length` ensures less points are falsely assigned as baseline points.
        Default is 2, which only removes lone baseline points.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from the moving average smoothing.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    Notes
    -----
    Only covers the baseline correction from FastChrom, not its peak finding and peak
    grouping capabilities.

    References
    ----------
    Johnsen, L., et al. An automated method for baseline correction, peak finding
    and peak grouping in chromatographic data. Analyst. 2013, 138, 3502-3511.

    """


@_classification_wrapper
def cwt_br(data, x_data=None, poly_order=5, scales=None, num_std=1.0, min_length=2,
           max_iter=50, tol=1e-3, symmetric=False, weights=None, **pad_kwargs):
    """
    Continuous wavelet transform baseline recognition (CWT-BR) algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int, optional
        The polynomial order for fitting the baseline. Default is 5.
    scales : array-like, optional
        The scales at which to perform the continuous wavelet transform. Default
        is None,
    num_std : float, optional
        The number of standard deviations to include when thresholding. Default
        is 1.0.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak points.
        A higher `min_length` ensures less points are falsely assigned as baseline points.
        Default is 2, which only removes lone baseline points.
    max_iter : int, optional
        The maximum number of iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    symmetric : bool, optional
        When fitting the identified baseline points with a polynomial, if `symmetric`
        is False (default), will add any point `i` as a baseline point where the fit
        polynomial is greater than the input data for ``N/100`` consecutive points on both
        sides of point `i`. If `symmetric` is True, then it means that both positive and
        negative peaks exist and baseline points are not modified during the polynomial fitting.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution for the
        continuous wavelet transform.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.
        * 'best_scale' : scalar
            The scale at which the Shannon entropy of the continuous wavelet transform
            of the data is at a minimum.

    Notes
    -----
    Uses the standard deviation for determining outliers during polynomial fitting rather
    than the standard error as used in the reference since the number of standard errors
    to include when thresholding varies with data size while the number of standard
    deviations is independent of data size.

    References
    ----------
    Bertinetto, C., et al. Automatic Baseline Recognition for the Correction of Large
    Sets of Spectra Using Continuous Wavelet Transform and Iterative Fitting. Applied
    Spectroscopy, 2014, 68(2), 155-164.

    """


def _haar(num_points, scale=2):
    """
    Creates a Haar wavelet.

    Parameters
    ----------
    num_points : int
        The number of points for the wavelet. Note that if `num_points` is odd
        and `scale` is even, or if `num_points` is even and `scale` is odd, then
        the length of the output wavelet will be `num_points` + 1 to ensure the
        symmetry of the wavelet.
    scale : int, optional
        The scale at which the wavelet is evaluated. Default is 2.

    Returns
    -------
    wavelet : numpy.ndarray
        The Haar wavelet.

    Notes
    -----
    This implementation is only designed to work for integer scales.

    Matches pywavelets's Haar implementation after applying patches from pywavelets
    issue #365 and pywavelets pull request #580.

    References
    ----------
    https://wikipedia.org/wiki/Haar_wavelet

    """
    # to maintain symmetry, even scales should have even windows and odd
    # scales have odd windows
    odd_scale = scale % 2
    odd_window = num_points % 2
    if (odd_scale and not odd_window) or (not odd_scale and odd_window):
        num_points += 1
    # center at 0 rather than 1/2 to make calculation easier
    # from [-scale/2 to 0), wavelet = 1; [0, scale/2), wavelet = -1
    x_vals = np.arange(num_points) - (num_points - 1) / 2
    wavelet = np.zeros(num_points)
    if not odd_scale:
        wavelet[(x_vals >= -scale / 2) & (x_vals < 0)] = 1
        wavelet[(x_vals < scale / 2) & (x_vals >= 0)] = -1
    else:
        # set such that wavelet[x_vals == 0] = 0
        wavelet[(x_vals > -scale / 2) & (x_vals < 0)] = 1
        wavelet[(x_vals < scale / 2) & (x_vals > 0)] = -1

    # the 1/sqrt(scale) is a normalization
    return wavelet / (np.sqrt(scale))


# adapted from scipy (scipy/signal/_wavelets.py/ricker); see license above
def _ricker(points, a):
    """
    Return a Ricker wavelet, also known as the "Mexican hat wavelet".

    It models the function:

        ``A * (1 - (x/a)**2) * exp(-0.5*(x/a)**2)``,

    where ``A = 2/(sqrt(3*a)*(pi**0.25))``.

    Parameters
    ----------
    points : int
        Number of points in `vector`.
        Will be centered around 0.
    a : scalar
        Width parameter of the wavelet.

    Returns
    -------
    vector : (N,) ndarray
        Array of length `points` in shape of ricker curve.

    Notes
    -----
    This function was deprecated from scipy.signal in version 1.12.

    """
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total


# adapted from scipy (scipy/signal/_wavelets.py/cwt); see license above
def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter. The `wavelet` function
    is allowed to be complex.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(length,width)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.
    dtype : data-type, optional
        The desired data type of output. Defaults to ``float64`` if the
        output of `wavelet` is real and ``complex128`` if it is complex.
    kwargs
        Keyword arguments passed to wavelet function.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(widths), len(data)).

    Notes
    -----
    This function was deprecated from scipy.signal in version 1.12.

    References
    ----------
    S. Mallat, "A Wavelet Tour of Signal Processing (3rd Edition)", Academic Press, 2009.

    """
    # Determine output type
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in 'FDG':
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = convolve(data, wavelet_data, mode='same')
    return output


@_classification_wrapper
def fabc(data, lam=1e6, scale=None, num_std=3.0, diff_order=2, min_length=2, weights=None,
         weights_as_mask=False, x_data=None, **pad_kwargs):
    """
    Fully automatic baseline correction (fabc).

    Similar to Dietrich's method, except that the derivative is estimated using a
    continuous wavelet transform and the baseline is calculated using Whittaker
    smoothing through the identified baseline points.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    scale : int, optional
        The scale at which to calculate the continuous wavelet transform. Should be
        approximately equal to the index-based full-width-at-half-maximum of the peaks
        or features in the data. Default is None, which will use half of the value from
        :func:`.optimize_window`, which is not always a good value, but at least scales
        with the number of data points and gives a starting point for tuning the parameter.
    num_std : float, optional
        The number of standard deviations to include when thresholding. Higher values
        will assign more points as baseline. Default is 3.0.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    min_length : int, optional
        Any region of consecutive baseline points less than `min_length` is considered
        to be a false positive and all points in the region are converted to peak points.
        A higher `min_length` ensures less points are falsely assigned as baseline points.
        Default is 2, which only removes lone baseline points.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    weights_as_mask : bool, optional
        If True, signifies that the input `weights` is the mask to use for fitting,
        which skips the continuous wavelet calculation and just smooths the input data.
        Default is False.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **pad_kwargs
        Additional keyword arguments to pass to :func:`.pad_edges` for padding
        the edges of the data to prevent edge effects from convolution for the
        continuous wavelet transform.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.
        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.

    Notes
    -----
    The classification of baseline points is similar to :meth:`~Baseline.dietrich`, except that
    this method approximates the first derivative using a continous wavelet transform
    with the Haar wavelet, which is more robust than the numerical derivative in
    Dietrich's method.

    References
    ----------
    Cobas, J., et al. A new general-purpose fully automatic baseline-correction
    procedure for 1D and 2D NMR data. Journal of Magnetic Resonance, 2006, 183(1),
    145-151.

    """


@_classification_wrapper
def rubberband(data, x_data=None, segments=1, lam=None, diff_order=2, weights=None,
               smooth_half_window=None, **pad_kwargs):
    """
    Identifies baseline points by fitting a convex hull to the bottom of the data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    segments : int or array-like[int], optional
        Used to fit multiple convex hulls to the data to negate the effects of
        concave data. If the input is an integer, it sets the number of equally sized
        segments the data will be split into. If the input is an array-like, each integer
        in the array will be the index that splits two segments, which allows
        constructing unequally sized segments. Default is 1, which fits a single convex
        hull to the data.
    lam : float or None, optional
        The smoothing parameter for interpolating the baseline points using
        Whittaker smoothing. Set to 0 or None to use linear interpolation instead.
        Default is None, which does not smooth.
    diff_order : int, optional
        The order of the differential matrix if using Whittaker smoothing. Must
        be greater than 0. Default is 2 (second order differential matrix).
        Typical values are 2 or 1.
    weights : array-like, shape (N,), optional
        The weighting array, used to override the function's baseline identification
        to designate peak points. Only elements with 0 or False values will have
        an effect; all non-zero values are considered potential baseline points. If None
        (default), then will be an array with size equal to N and all values set to 1.
    smooth_half_window : int or None, optional
        The half window to use for smoothing the input data with a moving average
        before calculating the convex hull, which gives much better results for
        noisy data. Set to None (default) or 0 to not smooth the data.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'mask': numpy.ndarray, shape (N,)
            The boolean array designating baseline points as True and peak points
            as False.

    Raises
    ------
    ValueError
        Raised if the number of segments per window for the fitting is less than
        `poly_order` + 1 or greater than the total number of points, or if the
        values in `self.x` are not strictly increasing.

    """
