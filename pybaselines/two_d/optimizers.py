# -*- coding: utf-8 -*-
"""High level functions for making better use of baseline algorithms.

Functions in this module make use of other baseline algorithms in
pybaselines to provide better results or optimize parameters.

Created on January 14, 2024
@author: Donald Erb

"""

from collections import defaultdict
from functools import partial
from math import ceil

import numpy as np

from . import morphological, polynomial, spline, whittaker
from .._validation import _check_optional_array, _get_row_col_values
from ..api import Baseline
from ..utils import _check_scalar, _sort_array2d
from ._algorithm_setup import _Algorithm2D


class _Optimizers(_Algorithm2D):
    """A base class for all optimizer algorithms."""

    @_Algorithm2D._register(ensure_2d=False, skip_sorting=True)
    def collab_pls(self, data, average_dataset=True, method='asls', method_kwargs=None):
        """
        Collaborative Penalized Least Squares (collab-PLS).

        Averages the data or the fit weights for an entire dataset to get more
        optimal results. Uses any Whittaker-smoothing-based or weighted spline algorithm.

        Parameters
        ----------
        data : array-like, shape (L, M, N)
            An array with shape (L, M, N) where L is the number of entries in
            the dataset and (M, N) is the shape of each data entry.
        average_dataset : bool, optional
            If True (default) will average the dataset before fitting to get the
            weighting. If False, will fit each individual entry in the dataset and
            then average the weights to get the weighting for the dataset.
        method : str, optional
            A string indicating the Whittaker-smoothing-based or weighted spline method to
            use for fitting the baseline. Default is 'asls'.
        method_kwargs : dict, optional
            A dictionary of keyword arguments to pass to the selected `method` function.
            Default is None, which will use an empty dictionary.

        Returns
        -------
        baselines : np.ndarray, shape (L, M, N)
            An array of all of the baselines.
        params : dict
            A dictionary with the following items:

            * 'average_weights': numpy.ndarray, shape (M, N)
                The weight array used to fit all of the baselines.
            * 'average_alpha': numpy.ndarray, shape (M, N)
                Only returned if `method` is 'aspls'. The
                `alpha` array used to fit all of the baselines for the
                :meth:`~Baseline2D.aspls`.

            Additional items depend on the output of the selected method. Every
            other key will have a list of values, with each item corresponding to a
            fit.

        Notes
        -----
        If `method` is 'aspls', `collab_pls` will also calculate
        the `alpha` array for the entire dataset in the same manner as the weights.

        References
        ----------
        Chen, L., et al. Collaborative Penalized Least Squares for Background
        Correction of Multiple Raman Spectra. Journal of Analytical Methods
        in Chemistry, 2018, 2018.

        """
        dataset, baseline_func, _, method_kws, _ = self._setup_optimizer(
            data, method, (whittaker, morphological, spline), method_kwargs,
            True
        )
        data_shape = dataset.shape
        if len(data_shape) != 3:
            raise ValueError((
                'the input data must have a shape of (number of measurements, number of x points,'
                f' number of y points), but instead has a shape of {data_shape}'
            ))
        method = method.lower()
        # if using aspls or pspline_aspls, also need to calculate the alpha array
        # for the entire dataset
        calc_alpha = method in ('aspls', 'pspline_aspls')

        # step 1: calculate weights for the entire dataset
        if average_dataset:
            _, fit_params = baseline_func(np.mean(dataset, axis=0), **method_kws)
            method_kws['weights'] = fit_params['weights']
            if calc_alpha:
                method_kws['alpha'] = fit_params['alpha']
        else:
            weights = np.empty(data_shape)
            if calc_alpha:
                alpha = np.empty(data_shape)
            for i, entry in enumerate(dataset):
                _, fit_params = baseline_func(entry, **method_kws)
                weights[i] = fit_params['weights']
                if calc_alpha:
                    alpha[i] = fit_params['alpha']
            method_kws['weights'] = np.mean(weights, axis=0)
            if calc_alpha:
                method_kws['alpha'] = np.mean(alpha, axis=0)

        # step 2: use the dataset weights from step 1 (stored in method_kws['weights'])
        # to fit each individual data entry; set tol to infinity so that only one
        # iteration is done and new weights are not calculated
        method_kws['tol'] = np.inf
        baselines = np.empty(data_shape)
        params = {'average_weights': method_kws['weights']}
        if calc_alpha:
            params['average_alpha'] = method_kws['alpha']
        if method == 'fabc':
            # set weights as mask so it just fits the data
            method_kws['weights_as_mask'] = True

        for i, entry in enumerate(dataset):
            baselines[i], param = baseline_func(entry, **method_kws)
            for key, value in param.items():
                if key in params:
                    params[key].append(value)
                else:
                    params[key] = [value]

        return baselines, params

    @_Algorithm2D._register(skip_sorting=True)
    def adaptive_minmax(self, data, poly_order=None, method='modpoly', weights=None,
                        constrained_fraction=0.01, constrained_weight=1e5,
                        estimation_poly_order=2, method_kwargs=None):
        """
        Fits polynomials of different orders and uses the maximum values as the baseline.

        Each polynomial order fit is done both unconstrained and constrained at the
        endpoints.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        poly_order : int or Sequence[int, int] or None, optional
            The two polynomial orders to use for fitting. If a single integer is given,
            then will use the input value and one plus the input value. Default is None,
            which will do a preliminary fit using a polynomial of order `estimation_poly_order`
            and then select the appropriate polynomial orders according to [32]_.
        method : {'modpoly', 'imodpoly'}, optional
            The method to use for fitting each polynomial. Default is 'modpoly'.
        weights : array-like, shape (M, N), optional
            The weighting array. If None (default), then will be an array with
            shape equal to (M, N) and all values set to 1.
        constrained_fraction : float or Sequence[float, float], optional
            The fraction of points at the left and right edges to use for the
            constrained fit. Default is 0.01. If `constrained_fraction` is a sequence,
            the first item is the fraction for the left edge and the second is the
            fraction for the right edge.
        constrained_weight : float or Sequence[float, float], optional
            The weighting to give to the endpoints. Higher values ensure that the
            end points are fit, but can cause large fluctuations in the other sections
            of the polynomial. Default is 1e5. If `constrained_weight` is a sequence,
            the first item is the weight for the left edge and the second is the
            weight for the right edge.
        estimation_poly_order : int, optional
            The polynomial order used for estimating the baseline-to-signal ratio
            to select the appropriate polynomial orders if `poly_order` is None.
            Default is 2.
        method_kwargs : dict, optional
            Additional keyword arguments to pass to
            :meth:`~Baseline.modpoly` or :meth:`~Baseline.imodpoly`. These include
            `tol`, `max_iter`, `use_original`, `mask_initial_peaks`, and `num_std`.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (M, N)
                The weight array used for fitting the data.
            * 'constrained_weights': numpy.ndarray, shape (M, N)
                The weight array used for the endpoint-constrained fits.
            * 'poly_order': numpy.ndarray, shape (2,)
                An array of the two polynomial orders used for the fitting.

        References
        ----------
        .. [32] Cao, A., et al. A robust method for automated background subtraction
            of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
            1199-1205.

        """
        y, baseline_func, _, method_kws, _ = self._setup_optimizer(
            data, method, [polynomial], method_kwargs, False
        )
        sort_weights = weights is not None
        weight_array = _check_optional_array(
            self._len, weights, check_finite=self._check_finite, ensure_1d=False, axis=slice(None)
        )
        if poly_order is None:
            poly_orders = _determine_polyorders(
                y, estimation_poly_order, weight_array, baseline_func, **method_kws
            )
        else:
            poly_orders, scalar_poly_order = _check_scalar(poly_order, 2, True, dtype=int)
            if scalar_poly_order:
                poly_orders[1] += 1  # add 1 since they are initially equal if scalar input

        # use high weighting rather than Lagrange multipliers to constrain the points
        # to better work with noisy data
        weightings = _get_row_col_values(constrained_weight)
        constrained_fractions = _get_row_col_values(constrained_fraction)
        if np.any(constrained_fractions < 0) or np.any(constrained_fractions > 1):
            raise ValueError('constrained_fraction must be between 0 and 1')

        # have to temporarily sort weights to match x- and y-ordering so that left and right edges
        # are correct
        if sort_weights:
            weight_array = _sort_array2d(weight_array, self._sort_order)

        constrained_weights = weight_array.copy()
        constrained_weights[:ceil(self._len[0] * constrained_fractions[0])] = weightings[0]
        constrained_weights[:, :ceil(self._len[1] * constrained_fractions[2])] = weightings[2]
        constrained_weights[
            self._len[0] - ceil(self._len[0] * constrained_fractions[1]):
        ] = weightings[1]
        constrained_weights[
            :, self._len[1] - ceil(self._len[1] * constrained_fractions[3]):
        ] = weightings[3]
        # and now change back to original ordering
        if sort_weights:
            weight_array = _sort_array2d(weight_array, self._inverted_order)
            constrained_weights = _sort_array2d(constrained_weights, self._inverted_order)

        # TODO should make parameters available; a list with an item for each fit like collab_pls
        # TODO could maybe just use itertools.permutations, but would want to know the order in
        # which the parameters are used
        baselines = np.empty((4, *self._len))
        baselines[0] = baseline_func(
            data=y, poly_order=poly_orders[0], weights=weight_array, **method_kws
        )[0]
        baselines[1] = baseline_func(
            data=y, poly_order=poly_orders[0], weights=constrained_weights, **method_kws
        )[0]
        baselines[2] = baseline_func(
            data=y, poly_order=poly_orders[1], weights=weight_array, **method_kws
        )[0]
        baselines[3] = baseline_func(
            data=y, poly_order=poly_orders[1], weights=constrained_weights, **method_kws
        )[0]

        # TODO should the coefficients also be made available? Would need to get them from
        # each of the fits
        params = {
            'weights': weight_array, 'constrained_weights': constrained_weights,
            'poly_order': poly_orders
        }

        return np.maximum.reduce(baselines), params

    @_Algorithm2D._register(skip_sorting=True)
    def individual_axes(self, data, axes=(0, 1), method='asls', method_kwargs=None):
        """
        Applies a one dimensional baseline correction method along each row and/or column.

        Parameters
        ----------
        data : array-like, shape (M, N)
            The y-values of the measured data.
        axes : (0, 1) or (1, 0) or 0 or 1, optional
            The axes along which to apply baseline correction. The order dictates along which
            axis baseline correction is first applied. Default is (0, 1), which applies baseline
            correction along the rows first and then the columns.
        method : str, optional
            A string indicating the algorithm to use for fitting the baseline of each row and/or
            column; can be any one dimensional algorithm in pybaselines. Default is 'asls'.
        method_kwargs : Sequence[dict] or dict, optional
            A sequence of dictionaries of keyword arguments to pass to the selected `method`
            function for each axis in `axes`. A single dictionary designates that the same
            keyword arguments will be used for each axis. Default is None, which will use an
            empty dictionary.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'params_rows': dict[str, list]
                Only if 0 is in `axes`. A dictionary of the parameters for each fit along
                the rows. The items within the dictionary will depend on the selected method.
            * 'params_columns': dict[str, list]
                Only if 1 is in `axes`. A dictionary of the parameters for each fit along
                the columns. The items within the dictionary will depend on the selected method.
            * 'baseline_rows': numpy.ndarray, shape (M, N)
                Only if 0 is in `axes`. The fit baseline along the rows.
            * 'baseline_columns': numpy.ndarray, shape (M, N)
                Only if 1 is in `axes`. The fit baseline along the columns.

        Raises
        ------
        ValueError
            Raised if `method_kwargs` is a sequence with length greater than `axes` or if
            the values in `axes` are duplicates.

        Notes
        -----
        If using array-like inputs within `method_kwargs`, they must correspond to their
        one-dimensional counterparts. For example, `weights` must be one-dimensional and
        have a length of `M` or `N` when used for fitting the rows or columns, respectively.
        Correctness of this is NOT verified within this method.

        """
        axes, scalar_axes = _check_scalar(axes, 2, fill_scalar=False, dtype=int)
        if scalar_axes:
            axes = [axes]
            num_axes = 1
        else:
            if axes[0] == axes[1]:
                raise ValueError('Fitting the same axis twice is not allowed')
            num_axes = 2
        if (
            method_kwargs is None
            or (not isinstance(method_kwargs, dict) and len(method_kwargs) == 0)
        ):
            method_kwargs = [{}] * num_axes
        elif isinstance(method_kwargs, dict):
            method_kwargs = [method_kwargs] * num_axes
        elif len(method_kwargs) == 1:
            method_kwargs = [method_kwargs[0]] * num_axes
        elif len(method_kwargs) != num_axes:
            raise ValueError('Method kwargs must have the same length as the input axes')

        keys = ('rows', 'columns')
        baseline = np.zeros(self._len)
        params = {}
        for i, axis in enumerate(axes):
            fitter = Baseline(
                (self.x, self.z)[axis], check_finite=self._check_finite, assume_sorted=True,
                output_dtype=self._dtype
            )
            fitter.pentapy_solver = self.pentapy_solver
            baseline_func = fitter._get_method(method)
            params[f'params_{keys[axis]}'] = defaultdict(list)
            func = partial(
                _update_params, baseline_func, params[f'params_{keys[axis]}'], **method_kwargs[i]
            )
            partial_baseline = np.apply_along_axis(func, axis, data - baseline)
            baseline += partial_baseline
            params[f'baseline_{keys[axis]}'] = partial_baseline

        return baseline, params


def _update_params(func, params, data, **kwargs):
    """
    A partial function to allow updating a params dictionary using NumPy's apply_aplong_axis.

    Parameters
    ----------
    func : Callable
        The baseline method to use.
    params : dict[str, list]
        The dictionary of parameters to be updated.
    data : numpy.ndarray
        The data to be baseline corrected.

    Returns
    -------
    baseline : numpy.ndarray
        The calculated basline.

    """
    baseline, baseline_params = func(data, **kwargs)
    for key, val in baseline_params.items():
        params[key].append(val)
    return baseline


def _determine_polyorders(y, poly_order, weights, fit_function, **fit_kwargs):
    """
    Selects the appropriate polynomial orders based on the baseline-to-signal ratio.

    Parameters
    ----------
    y : numpy.ndarray
        The array of y-values.
    poly_order : int
        The polynomial order for fitting.
    weights : numpy.ndarray
        The weight array for fitting.
    fit_function : Callable
        The function to use for the polynomial fit.
    **fit_kwargs
        Additional keyword arguments to pass to `fit_function`.

    Returns
    -------
    orders : numpy.ndarray, shape (2,)
        The two polynomial orders to use based on the baseline to signal
        ratio according to the reference.

    References
    ----------
    Cao, A., et al. A robust method for automated background subtraction
    of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38, 1199-1205.

    """
    baseline = fit_function(y, poly_order=poly_order, weights=weights, **fit_kwargs)[0]
    signal = y - baseline
    baseline_to_signal = (baseline.max() - baseline.min()) / (signal.max() - signal.min())
    # Table 2 in reference  # TODO in 2D does this need changed?
    if baseline_to_signal < 0.2:
        orders = (1, 2)
    elif baseline_to_signal < 0.75:
        orders = (2, 3)
    elif baseline_to_signal < 8.5:
        orders = (3, 4)
    elif baseline_to_signal < 55:
        orders = (4, 5)
    elif baseline_to_signal < 240:
        orders = (5, 6)
    elif baseline_to_signal < 517:
        orders = (6, 7)
    else:
        orders = (6, 8)  # not a typo, use 6 and 8 rather than 7 and 8

    return np.array(orders)
