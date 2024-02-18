# -*- coding: utf-8 -*-
"""Morphological techniques for fitting baselines to experimental data.

Created on April 8, 2023
@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import grey_dilation, grey_erosion, grey_opening, uniform_filter

from .._validation import _check_half_window
from ..utils import relative_difference
from ._algorithm_setup import _Algorithm2D


class _Morphological(_Algorithm2D):
    """A base class for all morphological algorithms."""

    @_Algorithm2D._register
    def mor(self, data, half_window=None, **window_kwargs):
        """
        A Morphological based (Mor) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        half_window : int or Sequence[int, int], optional
            The half-window used for the rows and columns, respectively, for the morphology
            functions. If a single value is given, rows and columns will use the same value.
            Default is None, which will optimize the half-window size using
            :func:`.optimize_window` and `window_kwargs`.
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
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': np.ndarray[int, int]
                The half windows used for the morphological calculations.

        References
        ----------
        Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
        Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64, 595-600.

        """
        y, half_wind = self._setup_morphology(data, half_window, **window_kwargs)
        opening = grey_opening(y, 2 * half_wind + 1)
        baseline = np.minimum(opening, _avg_opening(y, half_wind, opening))

        return baseline, {'half_window': half_wind}

    @_Algorithm2D._register
    def imor(self, data, half_window=None, tol=1e-3, max_iter=200, **window_kwargs):
        """
        An Improved Morphological based (IMor) baseline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        half_window : int or Sequence[int, int], optional
            The half-window used for the rows and columns, respectively, for the morphology
            functions. If a single value is given, rows and columns will use the same value.
            Default is None, which will optimize the half-window size using
            :func:`.optimize_window` and `window_kwargs`.
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
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': np.ndarray[int, int]
                The half windows used for the morphological calculations.
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

    @_Algorithm2D._register
    def rolling_ball(self, data, half_window=None, smooth_half_window=None,
                     pad_kwargs=None, **window_kwargs):
        """
        The rolling ball baseline algorithm.

        Applies a minimum and then maximum moving window, and subsequently smooths the
        result, giving a baseline that resembles rolling a ball across the data.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        half_window : int or Sequence[int, int], optional
            The half-window used for the rows and columns, respectively, for the morphology
            functions. If a single value is given, rows and columns will use the same value.
            Default is None, which will optimize the half-window size using
            :func:`.optimize_window` and `window_kwargs`.
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
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': np.ndarray[int, int]
                The half windows used for the morphological calculations.

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
        else:
            smooth_half_window = _check_half_window(smooth_half_window, allow_zero=True, two_d=True)

        rough_baseline = grey_opening(y, 2 * half_wind + 1)
        baseline = uniform_filter(
            rough_baseline, 2 * smooth_half_window + 1
        )

        return baseline, {'half_window': half_wind}

    @_Algorithm2D._register
    def tophat(self, data, half_window=None, **window_kwargs):
        """
        Estimates the baseline using a top-hat transformation (morphological opening).

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        half_window : int or Sequence[int, int], optional
            The half-window used for the rows and columns, respectively, for the morphology
            functions. If a single value is given, rows and columns will use the same value.
            Default is None, which will optimize the half-window size using
            :func:`.optimize_window` and `window_kwargs`.
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
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        dict
            A dictionary with the following items:

            * 'half_window': np.ndarray[int, int]
                The half windows used for the morphological calculations.

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
        baseline = grey_opening(y, 2 * half_wind + 1)

        return baseline, {'half_window': half_wind}


def _avg_opening(y, half_window, opening=None):
    """
    Averages the dilation and erosion of a morphological opening on data.

    Parameters
    ----------
    y : numpy.ndarray, shape (M, N)
        The array of the measured data.
    half_window : numpy.ndarray([int, int]), optional
        The half window size for the rows and columns, respectively, to use for the operations.
    opening : numpy.ndarray, optional
        The output of scipy.ndimage.grey_opening(y, window_size). Default is
        None, which will compute the value.

    Returns
    -------
    numpy.ndarray, shape (M, N)
        The average of the dilation and erosion of the opening.

    References
    ----------
    Perez-Pueyo, R., et al. Morphology-Based Automated Baseline Removal for
    Raman Spectra of Artistic Pigments. Applied Spectroscopy, 2010, 64 595-600.

    """
    # TODO should find a way to merge this with its 1D counterpart
    window_size = 2 * half_window + 1
    if opening is None:
        opening = grey_opening(y, window_size)
    return 0.5 * (
        grey_dilation(opening, window_size)
        + grey_erosion(opening, window_size)
    )
