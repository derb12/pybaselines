# -*- coding: utf-8 -*-
"""Morphological techniques for fitting baselines to experimental data.

Created on March 25, 2026
@author: Donald Erb

"""

import numpy as np
from scipy.ndimage import grey_opening

from ..utils import _avg_opening, _make_window, relative_difference
from ._algorithm_setup import _handle_io


class _MorphologicalNDMixin:
    """A mixin class for providing morphological methods for 1D and 2D."""

    @_handle_io
    def mor(self, data, half_window=None, window_kwargs=None, **kwargs):
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
            :func:`.estimate_window` and `window_kwargs`.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.estimate_window` for
            estimating the half window if `half_window` is None. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `window_kwargs`.

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
        y, half_wind = self._setup_morphology(data, half_window, window_kwargs, **kwargs)
        opening = grey_opening(y, _make_window(y, half_wind))
        baseline = np.minimum(opening, _avg_opening(y, half_wind, opening))

        return baseline, {'half_window': half_wind}

    @_handle_io
    def imor(self, data, half_window=None, tol=1e-3, max_iter=200, window_kwargs=None, **kwargs):
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
            :func:`.estimate_window` and `window_kwargs`.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        max_iter : int, optional
            The maximum number of iterations. Default is 200.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.estimate_window` for
            estimating the half window if `half_window` is None. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `window_kwargs`.

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'half_window': np.ndarray[int, int]
                The half windows used for the morphological calculations.
            * 'tol_history': numpy.ndarray
                An array containing the calculated tolerance values for
                each iteration. The length of the array is the number of iterations
                completed. If the last value in the array is greater than the input
                `tol` value, then the function did not converge.
            * 'success' : bool
                True if the method converged successfully, otherwise False.

        References
        ----------
        Dai, L., et al. An Automated Baseline Correction Method Based on Iterative
        Morphological Operations. Applied Spectroscopy, 2018, 72(5), 731-739.

        """
        y, half_wind = self._setup_morphology(data, half_window, window_kwargs, **kwargs)
        baseline = y
        tol_history = np.empty(max_iter + 1)
        success = False
        for i in range(max_iter + 1):
            baseline_new = np.minimum(y, _avg_opening(baseline, half_wind))
            calc_difference = relative_difference(baseline, baseline_new)
            tol_history[i] = calc_difference
            if calc_difference < tol:
                success = True
                break
            baseline = baseline_new

        params = {'half_window': half_wind, 'tol_history': tol_history[:i + 1], 'success': success}
        return baseline, params

    @_handle_io
    def tophat(self, data, half_window=None, window_kwargs=None, **kwargs):
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
            :func:`.estimate_window` and `window_kwargs`.
        window_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.estimate_window` for
            estimating the half window if `half_window` is None. Default is None.
        **kwargs

            .. deprecated:: 1.2.0
                Passing additional keyword arguments is deprecated and will be removed in version
                1.4.0. Pass keyword arguments using `window_kwargs`.

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
        y, half_wind = self._setup_morphology(data, half_window, window_kwargs, **kwargs)
        baseline = grey_opening(y, _make_window(y, half_wind))

        return baseline, {'half_window': half_wind}
