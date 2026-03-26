# -*- coding: utf-8 -*-
"""Morphological techniques for fitting baselines to experimental data.

Created on April 8, 2023
@author: Donald Erb

"""

from scipy.ndimage import grey_opening, uniform_filter

from .._nd.morphological import _MorphologicalNDMixin
from .._validation import _check_half_window
from ._algorithm_setup import _Algorithm2D


class _Morphological(_Algorithm2D, _MorphologicalNDMixin):
    """A base class for all morphological algorithms."""

    @_Algorithm2D._handle_io
    def rolling_ball(self, data, half_window=None, smooth_half_window=None,
                     pad_kwargs=None, window_kwargs=None, **kwargs):
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
            :func:`.estimate_window` and `window_kwargs`.
        smooth_half_window : int, optional
            The half-window to use for smoothing the data after performing the
            morphological operation. Default is None, which will use the same
            value as used for the morphological operation.
        pad_kwargs : dict, optional
            A dictionary of keyword arguments to pass to :func:`.pad_edges` for
            padding the edges of the data to prevent edge effects from the moving average.
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
        Kneen, M.A., et al. Algorithm for fitting XRF, SEM and PIXE X-ray spectra
        backgrounds. Nuclear Instruments and Methods in Physics Research B, 1996,
        109, 209-213.

        Liland, K., et al. Optimal Choice of Baseline Correction for Multivariate
        Calibration of Spectra. Applied Spectroscopy, 2010, 64(9), 1007-1016.

        """
        y, half_wind = self._setup_morphology(data, half_window, window_kwargs, **kwargs)
        if smooth_half_window is None:
            smooth_half_window = half_wind
        else:
            smooth_half_window = _check_half_window(smooth_half_window, allow_zero=True, two_d=True)

        rough_baseline = grey_opening(y, 2 * half_wind + 1)
        # TODO should pad the baseline here like in 1D
        baseline = uniform_filter(
            rough_baseline, 2 * smooth_half_window + 1
        )

        return baseline, {'half_window': half_wind}
