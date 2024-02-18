# -*- coding: utf-8 -*-
"""Morphological techniques for fitting baselines to experimental data.

Created on March 5, 2021
@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import grey_closing, grey_dilation, grey_erosion, grey_opening, uniform_filter1d

from ._algorithm_setup import _Algorithm, _class_wrapper
from ._validation import _check_lam
from .utils import _mollifier_kernel, _sort_array, pad_edges, padded_convolve, relative_difference


class _Morphological(_Algorithm):
    """A base class for all morphological algorithms."""

    @_Algorithm._register(sort_keys=('weights',))
    def mpls(self, data, half_window=None, lam=1e6, p=0.0, diff_order=2, tol=1e-3, max_iter=50,
             weights=None, **window_kwargs):
        """
        The Morphological penalized least squares (MPLS) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Anchor points
            identified by the procedure in [4]_ are given a weight of `1 - p`, and all
            other points have a weight of `p`. Default is 0.0.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the weights will be
            calculated following the procedure in [4]_.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'half_window': int
                The half window used for the morphological calculations.

        Raises
        ------
        ValueError
            Raised if p is not between 0 and 1.

        References
        ----------
        .. [4] Li, Zhong, et al. Morphological weighted penalized least squares for
            background correction. Analyst, 2013, 138, 4483-4492.

        """
        if not 0 <= p <= 1:
            raise ValueError('p must be between 0 and 1')

        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        if weights is not None:
            w = weights
        else:
            rough_baseline = grey_opening(y, [2 * half_wind + 1])
            diff = np.diff(
                np.concatenate([rough_baseline[:1], rough_baseline, rough_baseline[-1:]])
            )
            # diff == 0 means the point is on a flat segment, and diff != 0 means the
            # adjacent point is not the same flat segment. The union of the two finds
            # the endpoints of each segment, and np.flatnonzero converts the mask to
            # indices; indices will always be even-sized.
            indices = np.flatnonzero(
                ((diff[1:] == 0) | (diff[:-1] == 0)) & ((diff[1:] != 0) | (diff[:-1] != 0))
            )
            w = np.full(self._len, p)
            # find the index of min(y) in the region between flat regions
            for previous_segment, next_segment in zip(indices[1::2], indices[2::2]):
                index = np.argmin(y[previous_segment:next_segment + 1]) + previous_segment
                w[index] = 1 - p

            # have to invert the weight ordering to match the original input y ordering
            # since it will be sorted within _setup_whittaker
            w = _sort_array(w, self._inverted_order)

        _, weight_array = self._setup_whittaker(y, lam, diff_order, w)
        baseline = self.whittaker_system.solve(
            self.whittaker_system.add_diagonal(weight_array), weight_array * y,
            overwrite_ab=True, overwrite_b=True
        )

        params = {'weights': weight_array, 'half_window': half_wind}
        return baseline, params

    @_Algorithm._register
    def mor(self, data, half_window=None, **window_kwargs):
        """
        A Morphological based (Mor) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.

        References
        ----------
        Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
        Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        opening = grey_opening(y, [2 * half_wind + 1])
        baseline = np.minimum(opening, _avg_opening(y, half_wind, opening))

        return baseline, {'half_window': half_wind}

    @_Algorithm._register
    def imor(self, data, half_window=None, tol=1e-3, max_iter=200, **window_kwargs):
        """
        An Improved Morphological based (IMor) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 200.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        References
        ----------
        Dai, L., et al. An Automated Baseline Correction Method Based on Iterative
        Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        baseline = y
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline_new = np.minimum(y, _avg_opening(baseline, half_wind))
            calc_difference = relative_difference(baseline, baseline_new)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break
            baseline = baseline_new

        params = {'half_window': half_wind, 'tol_history': tol_history[:i + 1]}
        return baseline, params

    @_Algorithm._register
    def amormol(self, data, half_window=None, tol=1e-3, max_iter=200, pad_kwargs=None,
                **window_kwargs):
        """
        Iteratively averaging morphological and mollified (aMorMol) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 200.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for
            padding the edges of the data to prevent edge effects from convolution.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        References
        ----------
        Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
        Method for Raman Spectroscopy Based on Morphological Operations and
        Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        window_size = 2 * half_wind + 1
        kernel = _mollifier_kernel(window_size)
        data_bounds = slice(window_size, -window_size)

        pad_kws = pad_kwargs if pad_kwargs is not None else {}
        y = pad_edges(y, window_size, **pad_kws)
        baseline = y
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline_old = baseline
            baseline = padded_convolve(
                np.minimum(
                    y,
                    0.5 * (
                        grey_closing(baseline, [window_size])
                        + grey_opening(baseline, [window_size])
                    )
                ),
                kernel
            )
            calc_difference = relative_difference(baseline_old[data_bounds], baseline[data_bounds])
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

        params = {'half_window': half_wind, 'tol_history': tol_history[:i + 1]}
        return baseline[data_bounds], params

    @_Algorithm._register
    def mormol(self, data, half_window=None, tol=1e-3, max_iter=250, smooth_half_window=None,
               pad_kwargs=None, **window_kwargs):
        """
        Iterative morphological and mollified (MorMol) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 200.
        smooth_half_window : int, optional
            The half-window to use for smoothing the data before performing the
            morphological operation. Default is None, which will use a value of 1,
            which gives no smoothing.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for
            padding the edges of the data to prevent edge effects from convolution.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.

        References
        ----------
        Koch, M., et al. Iterative morphological and mollifier-based baseline
        correction for Raman spectra. J Raman Spectroscopy, 2017, 48(2), 336-342.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        window_size = 2 * half_wind + 1
        kernel = _mollifier_kernel(window_size)
        if smooth_half_window is None:
            smooth_half_window = 1
        smooth_kernel = _mollifier_kernel(smooth_half_window)
        data_bounds = slice(window_size, -window_size)

        pad_kws = pad_kwargs if pad_kwargs is not None else {}
        y = pad_edges(y, window_size, **pad_kws)
        baseline = np.zeros(y.shape[0])
        tol_history = np.empty(max_iter + 1)
        for i in range(max_iter + 1):
            baseline_old = baseline
            y_smooth = padded_convolve(y - baseline, smooth_kernel)
            baseline = baseline + padded_convolve(grey_erosion(y_smooth, window_size), kernel)
            calc_difference = relative_difference(
                baseline_old[data_bounds], baseline[data_bounds]
            )
            tol_history[i] = calc_difference
            if calc_difference < tol:
                break

        params = {'half_window': half_wind, 'tol_history': tol_history[:i + 1]}
        return baseline[data_bounds], params

    @_Algorithm._register
    def rolling_ball(self, data, half_window=None, smooth_half_window=None,
                     pad_kwargs=None, **window_kwargs):
        """
        The rolling ball baseline algorithm.

        Applies a minimum and then maximum moving window, and subsequently smooths the
        result, giving a baseline that resembles rolling a ball across the data.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        smooth_half_window : int, optional
            The half-window to use for smoothing the data after performing the
            morphological operation. Default is None, which will use the same
            value as used for the morphological operation.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for
            padding the edges of the data to prevent edge effects from the moving average.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.

        References
        ----------
        Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
        backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
        109, 209-213.

        Liland, K., et al. Optimal Choice of Baseline Correction for Multivariate
        Calibration of Spectra. Applied Spectroscopy, 2010, 64(9), 1007-1016.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        if smooth_half_window is None:
            smooth_half_window = half_wind

        rough_baseline = grey_opening(y, 2 * half_wind + 1)
        pad_kws = pad_kwargs if pad_kwargs is not None else {}
        baseline = uniform_filter1d(
            pad_edges(rough_baseline, smooth_half_window, **pad_kws),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]

        return baseline, {'half_window': half_wind}

    @_Algorithm._register
    def mwmv(self, data, half_window=None, smooth_half_window=None,
             pad_kwargs=None, **window_kwargs):
        """
        Moving window minimum value (MWMV) baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        smooth_half_window : int, optional
            The half-window to use for smoothing the data after performing the
            morphological operation. Default is None, which will use the same
            value as used for the morphological operation.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for
            padding the edges of the data to prevent edge effects from the moving average.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.

        Notes
        -----
        Performs poorly when baseline is rapidly changing.

        References
        ----------
        Yaroshchyk, P., et al. Automatic correction of continuum background in Laser-induced
        Breakdown Spectroscopy using a model-free algorithm. Spectrochimica Acta Part B, 2014,
        99, 138-149.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        if smooth_half_window is None:
            smooth_half_window = half_wind

        rough_baseline = grey_erosion(y, 2 * half_wind + 1)
        pad_kws = pad_kwargs if pad_kwargs is not None else {}
        baseline = uniform_filter1d(
            pad_edges(rough_baseline, smooth_half_window, **pad_kws),
            2 * smooth_half_window + 1
        )[smooth_half_window:self._len + smooth_half_window]

        return baseline, {'half_window': half_wind}

    @_Algorithm._register
    def tophat(self, data, half_window=None, **window_kwargs):
        """
        Estimates the baseline using a top-hat transformation (morphological opening).

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphological opening. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.

        Notes
        -----
        The actual top-hat transformation is defined as `data - opening(data)`, where
        `opening` is the morphological opening operation. This function, however, returns
        `opening(data)`, since that is technically the baseline defined by the operation.

        References
        ----------
        Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
        Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        baseline = grey_opening(y, [2 * half_wind + 1])

        return baseline, {'half_window': half_wind}

    @_Algorithm._register(sort_keys=('weights',), dtype=float, order='C')
    def mpspline(self, data, half_window=None, lam=1e4, lam_smooth=1e-2, p=0.0,
                 num_knots=100, spline_degree=3, diff_order=2, weights=None,
                 pad_kwargs=None, **window_kwargs):
        """
        Morphology-based penalized spline baseline.

        Identifies baseline points using morphological operations, and then uses weighted
        least-squares to fit a penalized spline to the baseline.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        lam : float, optional
            The smoothing parameter for the penalized spline when fitting the baseline.
            Larger values will create smoother baselines. Default is 1e4. Larger values
            are needed for larger `num_knots`.
        lam_smooth : float, optional
            The smoothing parameter for the penalized spline when smoothing the input
            data. Default is 1e-2. Larger values are needed for noisy data or for larger
            `num_knots`.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Anchor points
            identified by the procedure in the reference are given a weight of `1 - p`,
            and all other points have a weight of `p`. Default is 0.0.
        num_knots : int, optional
            The number of knots for the spline. Default is 100.
        spline_degree : int, optional
            The degree of the spline. Default is 3, which is a cubic spline.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the weights will be
            calculated following the procedure in the reference.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'half_window': int
                The half window used for the morphological calculations.

        Raises
        ------
        ValueError
            Raised if `half_window` is < 1, if `lam` or `lam_smooth` is <= 0, or if
            `p` is not between 0 and 1.

        Notes
        -----
        The optimal opening is calculated as the element-wise minimum of the opening and
        the average of the erosion and dilation of the opening. The reference used the
        erosion and dilation of the smoothed data, rather than the opening, which tends to
        overestimate the baseline.

        Rather than setting knots at the intersection points of the optimal opening and the
        smoothed data as described in the reference, weights are assigned to `1 - p` at the
        intersection points and `p` elsewhere. This simplifies the penalized spline
        calculation by allowing the use of equally spaced knots, but should otherwise give
        similar results as the reference algorithm.

        References
        ----------
        Gonzalez-Vidal, J., et al. Automatic morphology-based cubic p-spline fitting
        methodology for smoothing and baseline-removal of Raman spectra. Journal of
        Raman Spectroscopy. 2017, 48(6), 878-883.

        """
        if half_window is not None and half_window < 1:
            raise ValueError('half-window must be greater than 0')
        elif not 0 <= p <= 1:
            raise ValueError('p must be between 0 and 1')

        y, weight_array = self._setup_spline(
            data, weights, spline_degree, num_knots, True, diff_order, lam_smooth
        )

        # TODO should this use np.isclose instead?
        # TODO this overestimates the data when there is a lot of noise, leading to an
        # overestimated baseline; could alternatively just fit a p-spline to
        # 0.5 * (grey_closing(y, 3) + grey_opening(y, 3)), which averages noisy data better;
        # could add it as a boolean parameter
        spline_fit = self.pspline.solve_pspline(
            y, weights=(y == grey_closing(y, 3)).astype(float, copy=False)
        )
        if weights is None:
            _, half_window = self._setup_morphology(spline_fit, half_window, **window_kwargs)
            full_window = 2 * half_window + 1

            pad_kws = pad_kwargs if pad_kwargs is not None else {}
            padded_spline = pad_edges(spline_fit, full_window, **pad_kws)
            opening = grey_opening(padded_spline, full_window)
            # using the opening rather than padded_spline is more conservative when identifying
            # baseline points and results in much better results
            optimal_opening = np.minimum(
                opening, _avg_opening(y, half_window, opening)
            )[full_window:-full_window]

            # TODO should this use np.isclose instead?
            mask = spline_fit == optimal_opening
            weight_array[mask] = 1 - p
            weight_array[~mask] = p

        self.pspline.penalty = (_check_lam(lam) / lam_smooth) * self.pspline.penalty
        baseline = self.pspline.solve_pspline(spline_fit, weight_array)

        return baseline, {'half_window': half_window, 'weights': weight_array}

    @_Algorithm._register(sort_keys=('signal',))
    def jbcd(self, data, half_window=None, alpha=0.1, beta=1e1, gamma=1., beta_mult=1.1,
             gamma_mult=0.909, diff_order=1, max_iter=20, tol=1e-2, tol_2=1e-3,
             robust_opening=True, **window_kwargs):
        """
        Joint Baseline Correction and Denoising (jbcd) Algorithm.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        half_window : int, optional
            The half-window used for the morphology functions. If a value is input,
            then that value will be used. Default is None, which will optimize the
            half-window size using :func:`.optimize_window` and `window_kwargs`.
        alpha : float, optional
            The regularization parameter that controls how close the baseline must fit the
            calculated morphological opening. Larger values make the fit more constrained to
            the opening and can make the baseline less smooth. Default is 0.1.
        beta : float, optional
            The regularization parameter that controls how smooth the baseline is. Larger
            values produce smoother baselines. Default is 1e1.
        gamma : float, optional
            The regularization parameter that controls how smooth the signal is. Larger
            values produce smoother baselines. Default is 1.
        beta_mult : float, optional
            The value that `beta` is multiplied by each iteration. Default is 1.1.
        gamma_mult : float, optional
            The value that `gamma` is multiplied by each iteration. Default is 0.909.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 1
            (first order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The maximum number of iterations. Default is 20.
        tol : float, optional
            The exit criteria for the change in the calculated signal. Default is 1e-2.
        tol_2 : float, optional
            The exit criteria for the change in the calculated baseline. Default is 1e-2.
        robust_opening : bool, optional
            If True (default), the opening used to represent the initial baseline is the
            element-wise minimum between the morphological opening and the average of the
            morphological erosion and dilation of the opening, similar to :meth:`~Baseline.mor`. If
            False, the opening is just the morphological opening, as used in the reference.
            The robust opening typically represents the baseline better.
        **window_kwargs
            Values for setting the half window used for the morphology operations.
            Items include:

                * 'increment': int
                    The step size for iterating half windows. Default is 1.
                * 'max_hits': int
                    The number of consecutive half windows that must produce the same
                    morphological opening before accepting the half window as the
                    optimum value. Default is 1.
                * 'window_tol': float
                    The tolerance value for considering two morphological openings as
                    equivalent. Default is 1e-6.
                * 'max_half_window': int
                    The maximum allowable window size. If None (default), will be set
                    to (len(data) - 1) / 2.
                * 'min_half_window': int
                    The minimum half-window size. If None (default), will be set to 1.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'half_window': int
                The half window used for the morphological calculations.
            * 'tol_history': numpy.ndarray, shape (K, 2)
                An array containing the calculated tolerance values for each
                iteration. Index 0 are the tolerence values for the relative change in
                the signal, and index 1 are the tolerance values for the relative change
                in the baseline. The length of the array is the number of iterations
                completed, K. If the last values in the array are greater than the input
                `tol` or `tol_2` values, then the function did not converge.
            * 'signal': numpy.ndarray, shape (N,)
                The pure signal portion of the input `data` without noise or the baseline.

        References
        ----------
        Liu, H., et al. Joint Baseline-Correction and Denoising for Raman Spectra.
        Applied Spectroscopy, 2015, 69(9), 1013-1022.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        self._setup_whittaker(y, lam=1, diff_order=diff_order)
        beta = _check_lam(beta)
        gamma = _check_lam(gamma, allow_zero=True)

        opening = grey_opening(y, 2 * half_wind + 1)
        if robust_opening:
            opening = np.minimum(opening, _avg_opening(y, half_wind, opening))

        baseline_old = opening
        signal_old = y
        main_diag_idx = self.whittaker_system.main_diagonal_index
        partial_rhs_2 = (2 * alpha) * opening
        tol_history = np.empty((max_iter + 1, 2))
        for i in range(max_iter + 1):
            lhs_1 = gamma * self.whittaker_system.penalty
            lhs_1[main_diag_idx] += 1
            lhs_2 = (2 * beta) * self.whittaker_system.penalty
            lhs_2[main_diag_idx] += 1 + 2 * alpha

            signal = self.whittaker_system.solve(
                lhs_1, y - baseline_old, overwrite_ab=True, overwrite_b=True
            )
            baseline = self.whittaker_system.solve(
                lhs_2, y - signal + partial_rhs_2, overwrite_ab=True, overwrite_b=True
            )

            calc_tol_1 = relative_difference(signal_old, signal)
            calc_tol_2 = relative_difference(baseline_old, baseline)
            tol_history[i] = (calc_tol_1, calc_tol_2)
            if calc_tol_1 < tol and calc_tol_2 < tol_2:
                break
            signal_old = signal
            baseline_old = baseline
            gamma *= gamma_mult
            beta *= beta_mult

        params = {'half_window': half_wind, 'tol_history': tol_history[:i + 1], 'signal': signal}

        return baseline, params


_morphological_wrapper = _class_wrapper(_Morphological)


def _avg_opening(y, half_window, opening=None):
    """
    Averages the dilation and erosion of a morphological opening on data.

    Parameters
    ----------
    y : numpy.ndarray, shape (N,)
        The array of the measured data.
    half_window : int, optional
        The half window size to use for the operations.
    opening : numpy.ndarray, optional
        The output of scipy.ndimage.grey_opening(y, window_size). Default is
        None, which will compute the value.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The average of the dilation and erosion of the opening.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64 595-600.

    """
    window_size = 2 * half_window + 1
    if opening is None:
        opening = grey_opening(y, [window_size])
    return 0.5 * (grey_dilation(opening, [window_size]) + grey_erosion(opening, [window_size]))


@_morphological_wrapper
def mpls(data, half_window=None, lam=1e6, p=0.0, diff_order=2, tol=1e-3, max_iter=50,
         weights=None, x_data=None, **window_kwargs):
    """
    The Morphological penalized least squares (MPLS) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    lam : float, optional
        The smoothing parameter. Larger values will create smoother baselines.
        Default is 1e6.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Anchor points
        identified by the procedure in [1]_ are given a weight of `1 - p`, and all
        other points have a weight of `p`. Default is 0.0.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 1.
    max_iter : int, optional
        The max number of fit iterations. Default is 50.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be
        calculated following the procedure in [1]_.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'half_window': int
            The half window used for the morphological calculations.

    Raises
    ------
    ValueError
        Raised if p is not between 0 and 1.

    References
    ----------
    .. [1] Li, Zhong, et al. Morphological weighted penalized least squares for
           background correction. Analyst, 2013, 138, 4483-4492.

    """


@_morphological_wrapper
def mor(data, half_window=None, x_data=None, **window_kwargs):
    """
    A Morphological based (Mor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

    """


@_morphological_wrapper
def imor(data, half_window=None, tol=1e-3, max_iter=200, x_data=None, **window_kwargs):
    """
    An Improved Morphological based (IMor) baseline algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    References
    ----------
    Dai, L., et al. An Automated Baseline Correction Method Based on Iterative
    Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

    """


@_morphological_wrapper
def amormol(data, half_window=None, tol=1e-3, max_iter=200, pad_kwargs=None, x_data=None,
            **window_kwargs):
    """
    Iteratively averaging morphological and mollified (aMorMol) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from convolution.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    References
    ----------
    Chen, H., et al. An Adaptive and Fully Automated Baseline Correction
    Method for Raman Spectroscopy Based on Morphological Operations and
    Mollifications. Applied Spectroscopy, 2019, 73(3), 284-293.

    """


@_morphological_wrapper
def mormol(data, half_window=None, tol=1e-3, max_iter=250, smooth_half_window=None,
           pad_kwargs=None, x_data=None, **window_kwargs):
    """
    Iterative morphological and mollified (MorMol) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    tol : float, optional
        The exit criteria. Default is 1e-3.
    max_iter : int, optional
        The maximum number of iterations. Default is 200.
    smooth_half_window : int, optional
        The half-window to use for smoothing the data before performing the
        morphological operation. Default is None, which will use a value of 1,
        which gives no smoothing.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from convolution.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.
        * 'tol_history': numpy.ndarray
            An array containing the calculated tolerance values for
            each iteration. The length of the array is the number of iterations
            completed. If the last value in the array is greater than the input
            `tol` value, then the function did not converge.

    References
    ----------
    Koch, M., et al. Iterative morphological and mollifier-based baseline
    correction for Raman spectra. J Raman Spectroscopy, 2017, 48(2), 336-342.

    """


@_morphological_wrapper
def rolling_ball(data, half_window=None, smooth_half_window=None, pad_kwargs=None,
                 x_data=None, **window_kwargs):
    """
    The rolling ball baseline algorithm.

    Applies a minimum and then maximum moving window, and subsequently smooths the
    result, giving a baseline that resembles rolling a ball across the data.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    smooth_half_window : int, optional
        The half-window to use for smoothing the data after performing the
        morphological operation. Default is None, which will use the same
        value as used for the morphological operation.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from the moving average.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    References
    ----------
    Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
    backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
    109, 209-213.

    Liland, K., et al. Optimal Choice of Baseline Correction for Multivariate
    Calibration of Spectra. Applied Spectroscopy, 2010, 64(9), 1007-1016.

    """


@_morphological_wrapper
def mwmv(data, half_window=None, smooth_half_window=None, pad_kwargs=None,
         x_data=None, **window_kwargs):
    """
    Moving window minimum value (MWMV) baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    smooth_half_window : int, optional
        The half-window to use for smoothing the data after performing the
        morphological operation. Default is None, which will use the same
        value as used for the morphological operation.
    pad_kwargs : dict, optional
        A dictionary of keyword arguments to pass to :func:`.pad_edges` for
        padding the edges of the data to prevent edge effects from the moving average.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    Notes
    -----
    Performs poorly when baseline is rapidly changing.

    References
    ----------
    Yaroshchyk, P., et al. Automatic correction of continuum background in Laser-induced
    Breakdown Spectroscopy using a model-free algorithm. Spectrochimica Acta Part B, 2014,
    99, 138-149.

    """


@_morphological_wrapper
def tophat(data, half_window=None, x_data=None, **window_kwargs):
    """
    Estimates the baseline using a top-hat transformation (morphological opening).

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphological opening. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.

    Notes
    -----
    The actual top-hat transformation is defined as `data - opening(data)`, where
    `opening` is the morphological opening operation. This function, however, returns
    `opening(data)`, since that is technically the baseline defined by the operation.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

    """


@_morphological_wrapper
def mpspline(data, half_window=None, lam=1e4, lam_smooth=1e-2, p=0.0, num_knots=100,
             spline_degree=3, diff_order=2, weights=None, pad_kwargs=None, x_data=None,
             **window_kwargs):
    """
    Morphology-based penalized spline baseline.

    Identifies baseline points using morphological operations, and then uses weighted
    least-squares to fit a penalized spline to the baseline.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    lam : float, optional
        The smoothing parameter for the penalized spline when fitting the baseline.
        Larger values will create smoother baselines. Default is 1e4. Larger values
        are needed for larger `num_knots`.
    lam_smooth : float, optional
        The smoothing parameter for the penalized spline when smoothing the input
        data. Default is 1e-2. Larger values are needed for noisy data or for larger
        `num_knots`.
    p : float, optional
        The penalizing weighting factor. Must be between 0 and 1. Anchor points
        identified by the procedure in the reference are given a weight of `1 - p`,
        and all other points have a weight of `p`. Default is 0.0.
    num_knots : int, optional
        The number of knots for the spline. Default is 100.
    spline_degree : int, optional
        The degree of the spline. Default is 3, which is a cubic spline.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 2
        (second order differential matrix). Typical values are 2 or 3.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then the weights will be
        calculated following the procedure in the reference.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'half_window': int
            The half window used for the morphological calculations.

    Raises
    ------
    ValueError
        Raised if `half_window` is < 1, if `lam` or `lam_smooth` is <= 0, or if
        `p` is not between 0 and 1.

    Notes
    -----
    The optimal opening is calculated as the element-wise minimum of the opening and
    the average of the erosion and dilation of the opening. The reference used the
    erosion and dilation of the smoothed data, rather than the opening, which tends to
    overestimate the baseline.

    Rather than setting knots at the intersection points of the optimal opening and the
    smoothed data as described in the reference, weights are assigned to `1 - p` at the
    intersection points and `p` elsewhere. This simplifies the penalized spline
    calculation by allowing the use of equally spaced knots, but should otherwise give
    similar results as the reference algorithm.

    References
    ----------
    Gonzalez-Vidal, J., et al. Automatic morphology-based cubic p-spline fitting
    methodology for smoothing and baseline-removal of Raman spectra. Journal of
    Raman Spectroscopy. 2017, 48(6), 878-883.

    """


@_morphological_wrapper
def jbcd(data, half_window=None, alpha=0.1, beta=1e1, gamma=1., beta_mult=1.1, gamma_mult=0.909,
         diff_order=1, max_iter=20, tol=1e-2, tol_2=1e-3, robust_opening=True, x_data=None,
         **window_kwargs):
    """
    Joint Baseline Correction and Denoising (jbcd) Algorithm.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    half_window : int, optional
        The half-window used for the morphology functions. If a value is input,
        then that value will be used. Default is None, which will optimize the
        half-window size using :func:`.optimize_window` and `window_kwargs`.
    alpha : float, optional
        The regularization parameter that controls how close the baseline must fit the
        calculated morphological opening. Larger values make the fit more constrained to
        the opening and can make the baseline less smooth. Default is 0.1.
    beta : float, optional
        The regularization parameter that controls how smooth the baseline is. Larger
        values produce smoother baselines. Default is 1e1.
    gamma : float, optional
        The regularization parameter that controls how smooth the signal is. Larger
        values produce smoother baselines. Default is 1.
    beta_mult : float, optional
        The value that `beta` is multiplied by each iteration. Default is 1.1.
    gamma_mult : float, optional
        The value that `gamma` is multiplied by each iteration. Default is 0.909.
    diff_order : int, optional
        The order of the differential matrix. Must be greater than 0. Default is 1
        (first order differential matrix). Typical values are 2 or 1.
    max_iter : int, optional
        The maximum number of iterations. Default is 20.
    tol : float, optional
        The exit criteria for the change in the calculated signal. Default is 1e-2.
    tol_2 : float, optional
        The exit criteria for the change in the calculated baseline. Default is 1e-2.
    robust_opening : bool, optional
        If True (default), the opening used to represent the initial baseline is the
        element-wise minimum between the morphological opening and the average of the
        morphological erosion and dilation of the opening, similar to :meth:`~Baseline.mor`. If
        False, the opening is just the morphological opening, as used in the reference.
        The robust opening typically represents the baseline better.
    x_data : array-like, optional
        The x-values. Not used by this function, but input is allowed for consistency
        with other functions.
    **window_kwargs
        Values for setting the half window used for the morphology operations.
        Items include:

            * 'increment': int
                The step size for iterating half windows. Default is 1.
            * 'max_hits': int
                The number of consecutive half windows that must produce the same
                morphological opening before accepting the half window as the
                optimum value. Default is 1.
            * 'window_tol': float
                The tolerance value for considering two morphological openings as
                equivalent. Default is 1e-6.
            * 'max_half_window': int
                The maximum allowable window size. If None (default), will be set
                to (len(data) - 1) / 2.
            * 'min_half_window': int
                The minimum half-window size. If None (default), will be set to 1.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'half_window': int
            The half window used for the morphological calculations.
        * 'tol_history': numpy.ndarray, shape (K, 2)
            An array containing the calculated tolerance values for each
            iteration. Index 0 are the tolerence values for the relative change in
            the signal, and index 1 are the tolerance values for the relative change
            in the baseline. The length of the array is the number of iterations
            completed, K. If the last values in the array are greater than the input
            `tol` or `tol_2` values, then the function did not converge.
        * 'signal': numpy.ndarray, shape (N,)
            The pure signal portion of the input `data` without noise or the baseline.

    References
    ----------
    Liu, H., et al. Joint Baseline-Correction and Denoising for Raman Spectra.
    Applied Spectroscopy, 2015, 69(9), 1013-1022.

    """
