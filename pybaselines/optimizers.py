# -*- coding: utf-8 -*-
"""High level functions for making better use of baseline algorithms.

Functions in this module make use of other baseline algorithms in
pybaselines to provide better results or optimize parameters.

Created on March 3, 2021
@author: Donald Erb

"""

from math import ceil

import numpy as np

from . import classification, misc, morphological, polynomial, smooth, spline, whittaker
from ._algorithm_setup import _Algorithm, _class_wrapper
from ._validation import _check_optional_array
from .utils import _check_scalar, _get_edges, _sort_array, gaussian


class _Optimizers(_Algorithm):
    """A base class for all optimizer algorithms."""

    @_Algorithm._register(ensure_1d=False, skip_sorting=True)
    def collab_pls(self, data, average_dataset=True, method='asls', method_kwargs=None):
        """
        Collaborative Penalized Least Squares (collab-PLS).

        Averages the data or the fit weights for an entire dataset to get more
        optimal results. Uses any Whittaker-smoothing-based or weighted spline algorithm.

        Parameters
        ----------
        data : array-like, shape (M, N)
            An array with shape (M, N) where M is the number of entries in
            the dataset and N is the number of data points in each entry.
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
        baselines : np.ndarray, shape (M, N)
            An array of all of the baselines.
        params : dict
            A dictionary with the following items:

            * 'average_weights': numpy.ndarray, shape (N,)
                The weight array used to fit all of the baselines.
            * 'average_alpha': numpy.ndarray, shape (N,)
                Only returned if `method` is 'aspls' or 'pspline_aspls'. The
                `alpha` array used to fit all of the baselines for the
                :meth:`~Baseline.aspls` or :meth:`~Baseline.pspline_aspls` methods.

            Additional items depend on the output of the selected method. Every
            other key will have a list of values, with each item corresponding to a
            fit.

        Notes
        -----
        If `method` is 'aspls' or 'pspline_aspls', `collab_pls` will also calculate
        the `alpha` array for the entire dataset in the same manner as the weights.

        References
        ----------
        Chen, L., et al. Collaborative Penalized Least Squares for Background
        Correction of Multiple Raman Spectra. Journal of Analytical Methods
        in Chemistry, 2018, 2018.

        """
        dataset, baseline_func, _, method_kws, _ = self._setup_optimizer(
            data, method, (whittaker, morphological, classification, spline), method_kwargs,
            True
        )
        data_shape = dataset.shape
        if len(data_shape) != 2:
            raise ValueError((
                'the input data must have a shape of (number of measurements, number of points), '
                f'but instead has a shape of {data_shape}'
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

    @_Algorithm._register(skip_sorting=True)
    def optimize_extended_range(self, data, method='asls', side='both', width_scale=0.1,
                                height_scale=1., sigma_scale=1. / 12., min_value=2, max_value=8,
                                step=1, pad_kwargs=None, method_kwargs=None):
        """
        Extends data and finds the best parameter value for the given baseline method.

        Adds additional data to the left and/or right of the input data, and then iterates
        through parameter values to find the best fit. Useful for calculating the optimum
        `lam` or `poly_order` value required to optimize other algorithms.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        method : str, optional
            A string indicating the Whittaker-smoothing-based, polynomial, or spline method
            to use for fitting the baseline. Default is 'asls'.
        side : {'both', 'left', 'right'}, optional
            The side of the measured data to extend. Default is 'both'.
        width_scale : float, optional
            The number of data points added to each side is `width_scale` * N. Default
            is 0.1.
        height_scale : float, optional
            The height of the added Gaussian peak(s) is calculated as
            `height_scale` * max(`data`). Default is 1.
        sigma_scale : float, optional
            The sigma value for the added Gaussian peak(s) is calculated as
            `sigma_scale` * `width_scale` * N. Default is 1/12, which will make
            the Gaussian span +- 6 sigma, making its total width about half of the
            added length.
        min_value : int or float, optional
            The minimum value for the `lam` or `poly_order` value to use with the
            indicated method. If using a polynomial method, `min_value` must be an
            integer. If using a Whittaker-smoothing-based method, `min_value` should
            be the exponent to raise to the power of 10 (eg. a `min_value` value of 2
            designates a `lam` value of 10**2).
            Default is 2.
        max_value : int or float, optional
            The maximum value for the `lam` or `poly_order` value to use with the
            indicated method. If using a polynomial method, `max_value` must be an
            integer. If using a Whittaker-smoothing-based method, `max_value` should
            be the exponent to raise to the power of 10 (eg. a `max_value` value of 3
            designates a `lam` value of 10**3).
            Default is 8.
        step : int or float, optional
            The step size for iterating the parameter value from `min_value` to `max_value`.
            If using a polynomial method, `step` must be an integer.
        pad_kwargs : dict, optional
            A dictionary of options to pass to :func:`.pad_edges` for padding
            the edges of the data when adding the extended left and/or right sections.
            Default is None, which will use an empty dictionary.
        method_kwargs : dict, optional
            A dictionary of keyword arguments to pass to the selected `method` function.
            Default is None, which will use an empty dictionary.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The baseline calculated with the optimum parameter.
        method_params : dict
            A dictionary with the following items:

            * 'optimal_parameter': int or float
                The `lam` or `poly_order` value that produced the lowest
                root-mean-squared-error.
            * 'min_rmse': float
                The minimum root-mean-squared-error obtained when using
                the optimal parameter.

            Additional items depend on the output of the selected method.

        Raises
        ------
        ValueError
            Raised if `side` is not 'left', 'right', or 'both'.
        TypeError
            Raised if using a polynomial method and `min_value`, `max_value`, or
            `step` is not an integer.
        ValueError
            Raised if using a Whittaker-smoothing-based method and `min_value`,
            `max_value`, or `step` is greater than 100.

        Notes
        -----
        Based on the extended range penalized least squares (erPLS) method from [5]_.
        The method proposed by [5]_ was for optimizing lambda only for the aspls
        method by extending only the right side of the spectrum. The method was
        modified by allowing extending either side following [6]_, and for optimizing
        lambda or the polynomial degree for all of the affected algorithms in
        pybaselines.

        References
        ----------
        .. [5] Zhang, F., et al. An Automatic Baseline Correction Method Based on
            the Penalized Least Squares Method. Sensors, 2020, 20(7), 2015.
        .. [6] Krishna, H., et al. Range-independent background subtraction algorithm
            for recovery of Raman spectra of biological tissue. Journal of Raman
            Spectroscopy. 2012, 43(12), 1884-1894.

        """
        side = side.lower()
        if side not in ('left', 'right', 'both'):
            raise ValueError('side must be "left", "right", or "both"')

        y, baseline_func, func_module, method_kws, fit_object = self._setup_optimizer(
            data, method, (whittaker, polynomial, morphological, spline, classification),
            method_kwargs, True
        )
        method = method.lower()
        if func_module == 'polynomial' or method in ('dietrich', 'cwt_br'):
            if any(not isinstance(val, int) for val in (min_value, max_value, step)):
                raise TypeError((
                    'min_value, max_value, and step must all be integers when'
                    ' using a polynomial method'
                ))
            param_name = 'poly_order'
        else:
            if any(val > 100 for val in (min_value, max_value, step)):
                raise ValueError((
                    'min_value, max_value, and step should be the power of 10 to use '
                    '(eg. min_value=2 denotes 10**2), not the actual "lam" value, and '
                    'thus should not be greater than 100'
                ))
            param_name = 'lam'

        added_window = int(self._len * width_scale)
        for key in ('weights', 'alpha'):
            if key in method_kws:
                method_kws[key] = np.pad(
                    method_kws[key],
                    [0 if side == 'right' else added_window, 0 if side == 'left' else added_window],
                    'constant', constant_values=1
                )

        min_x = self.x_domain[0]
        max_x = self.x_domain[1]
        x_range = max_x - min_x
        known_background = np.array([])
        fit_x_data = self.x
        fit_data = y
        lower_bound = upper_bound = 0

        if pad_kwargs is None:
            pad_kwargs = {}

        added_left, added_right = _get_edges(
            _sort_array(y, self._sort_order), added_window, **pad_kwargs
        )
        added_gaussian = gaussian(
            np.linspace(-added_window / 2, added_window / 2, added_window),
            height_scale * abs(y.max()), 0, added_window * sigma_scale
        )
        if side in ('right', 'both'):
            added_x = np.linspace(
                max_x, max_x + x_range * (width_scale / 2), added_window + 1
            )[1:]
            fit_x_data = np.concatenate((fit_x_data, added_x))
            fit_data = np.concatenate((fit_data, added_gaussian + added_right))
            known_background = added_right
            upper_bound += added_window
        if side in ('left', 'both'):
            added_x = np.linspace(
                min_x - x_range * (width_scale / 2), min_x, added_window + 1
            )[:-1]
            fit_x_data = np.concatenate((added_x, fit_x_data))
            fit_data = np.concatenate((added_gaussian + added_left, fit_data))
            known_background = np.concatenate((known_background, added_left))
            lower_bound += added_window

        added_len = 2 * added_window if side == 'both' else added_window
        upper_idx = len(fit_data) - upper_bound
        min_sum_squares = np.inf
        best_val = None

        if self._sort_order is None:
            new_sort_order = None
        else:
            if side == 'right':
                new_sort_order = np.concatenate((
                    self._sort_order, np.arange(self._len, self._len + added_len, dtype=np.intp)
                ), dtype=np.intp)
            elif side == 'left':
                new_sort_order = np.concatenate((
                    np.arange(added_len, dtype=np.intp), self._sort_order + added_len
                ), dtype=np.intp)
            else:
                new_sort_order = np.concatenate((
                    np.arange(added_window, dtype=np.intp),
                    self._sort_order + added_window,
                    np.arange(self._len + added_window, self._len + added_len, dtype=np.intp)
                ), dtype=np.intp)

        # TODO maybe switch to linspace since arange is inconsistent when using floats
        with fit_object._override_x(fit_x_data, new_sort_order=new_sort_order):
            for var in np.arange(min_value, max_value + step, step):
                if param_name == 'lam':
                    method_kws[param_name] = 10**var
                else:
                    method_kws[param_name] = var
                fit_baseline, fit_params = baseline_func(fit_data, **method_kws)
                # TODO change the known baseline so that np.roll does not have to be
                # calculated each time, since it requires additional time
                residual = (
                    known_background - np.roll(fit_baseline, upper_bound)[:added_len]
                )
                # just calculate the sum of squares to reduce time from using sqrt for rmse
                sum_squares = residual.dot(residual)
                if sum_squares < min_sum_squares:
                    baseline = fit_baseline[lower_bound:upper_idx]
                    params = fit_params
                    best_val = var
                    min_sum_squares = sum_squares

        params.update(
            {'optimal_parameter': best_val, 'min_rmse': np.sqrt(min_sum_squares / added_len)}
        )
        for key in ('weights', 'alpha'):
            if key in params:
                params[key] = params[key][
                    0 if side == 'right' else added_window:
                    None if side == 'left' else -added_window
                ]

        return baseline, params

    @_Algorithm._register(skip_sorting=True)
    def adaptive_minmax(self, data, poly_order=None, method='modpoly', weights=None,
                        constrained_fraction=0.01, constrained_weight=1e5,
                        estimation_poly_order=2, method_kwargs=None):
        """
        Fits polynomials of different orders and uses the maximum values as the baseline.

        Each polynomial order fit is done both unconstrained and constrained at the
        endpoints.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        poly_order : int or Sequence(int, int) or None, optional
            The two polynomial orders to use for fitting. If a single integer is given,
            then will use the input value and one plus the input value. Default is None,
            which will do a preliminary fit using a polynomial of order `estimation_poly_order`
            and then select the appropriate polynomial orders according to [7]_.
        method : {'modpoly', 'imodpoly'}, optional
            The method to use for fitting each polynomial. Default is 'modpoly'.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then will be an array with
            size equal to N and all values set to 1.
        constrained_fraction : float or Sequence(float, float), optional
            The fraction of points at the left and right edges to use for the
            constrained fit. Default is 0.01. If `constrained_fraction` is a sequence,
            the first item is the fraction for the left edge and the second is the
            fraction for the right edge.
        constrained_weight : float or Sequence(float, float), optional
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
        numpy.ndarray, shape (N,)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'weights': numpy.ndarray, shape (N,)
                The weight array used for fitting the data.
            * 'constrained_weights': numpy.ndarray, shape (N,)
                The weight array used for the endpoint-constrained fits.
            * 'poly_order': numpy.ndarray, shape (2,)
                An array of the two polynomial orders used for the fitting.

        References
        ----------
        .. [7] Cao, A., et al. A robust method for automated background subtraction
            of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
            1199-1205.

        """
        y, baseline_func, _, method_kws, _ = self._setup_optimizer(
            data, method, [polynomial], method_kwargs, False
        )
        sort_weights = weights is not None
        weight_array = _check_optional_array(self._len, weights, check_finite=self._check_finite)
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
        weightings = _check_scalar(constrained_weight, 2, True)[0]
        constrained_fractions = _check_scalar(constrained_fraction, 2, True)[0]
        if np.any(constrained_fractions < 0) or np.any(constrained_fractions > 1):
            raise ValueError('constrained_fraction must be between 0 and 1')

        # have to temporarily sort weights to match x- and y-ordering so that left and right edges
        # are correct
        if sort_weights:
            weight_array = _sort_array(weight_array, self._sort_order)

        constrained_weights = weight_array.copy()
        constrained_weights[:ceil(self._len * constrained_fractions[0])] = weightings[0]
        constrained_weights[
            self._len - ceil(self._len * constrained_fractions[1]):
        ] = weightings[1]

        # and now change back to original ordering
        if sort_weights:
            weight_array = _sort_array(weight_array, self._inverted_order)
            constrained_weights = _sort_array(constrained_weights, self._inverted_order)

        # TODO should make parameters available; a list with an item for each fit like collab_pls
        # TODO could maybe just use itertools.permutations, but would want to know the order in
        # which the parameters are used
        baselines = np.empty((4, self._len))
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

    @_Algorithm._register
    def custom_bc(self, data, method='asls', regions=((None, None),), sampling=1, lam=None,
                  diff_order=2, method_kwargs=None):
        """
        Customized baseline correction for fine tuned stiffness of the baseline at specific regions.

        Divides the data into regions with variable number of data points and then uses other
        baseline algorithms to fit the truncated data. Regions with less points effectively
        makes the fit baseline more stiff in those regions.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        method : str, optional
            A string indicating the algorithm to use for fitting the baseline; can be any
            non-optimizer algorithm in pybaselines. Default is 'asls'.
        regions : array-like, shape (M, 2), optional
            The two dimensional array containing the start and stop indices for each region of
            interest. Each region is defined as ``data[start:stop]``. Default is ((None, None),),
            which will use all points.
        sampling : int or array-like, optional
            The sampling step size for each region defined in `regions`. If `sampling` is an
            integer, then all regions will use the same index step size; if `sampling` is an
            array-like, its length must be equal to `M`, the first dimension in `regions`.
            Default is 1, which will use all points.
        lam : float or None, optional
            The value for smoothing the calculated interpolated baseline using Whittaker
            smoothing, in order to reduce the kinks between regions. Default is None, which
            will not smooth the baseline; a value of 0 will also not perform smoothing.
        diff_order : int, optional
            The difference order used for Whittaker smoothing of the calculated baseline.
            Default is 2.
        method_kwargs : dict, optional
            A dictionary of keyword arguments to pass to the selected `method` function.
            Default is None, which will use an empty dictionary.

        Returns
        -------
        baseline : numpy.ndarray, shape (N,)
            The baseline calculated with the optimum parameter.
        params : dict
            A dictionary with the following items:
                * 'x_fit': numpy.ndarray, shape (P,)
                    The truncated x-values used for fitting the baseline.
                * 'y_fit': numpy.ndarray, shape (P,)
                    The truncated y-values used for fitting the baseline.

            Additional items depend on the output of the selected method.

        Raises
        ------
        ValueError
            Raised if `regions` is not two dimensional, if `sampling` is not the same length
            as `rois.shape[0]`, if any values in `sampling` or `regions` is less than 1, if
            segments in `regions` overlap, or if any value in `regions` is greater than the
            length of the input data.

        Notes
        -----
        Uses Whittaker smoothing to smooth the transitions between regions rather than LOESS
        as used in [31]_.

        Uses binning rather than direct truncation of the regions in order to get better
        results for noisy data.

        References
        ----------
        .. [31] Liland, K., et al. Customized baseline correction. Chemometrics and
                Intelligent Laboratory Systems, 2011, 109(1), 51-56.

        """
        y, baseline_func, _, method_kws, fitting_object = self._setup_optimizer(
            data, method,
            (classification, misc, morphological, polynomial, smooth, spline, whittaker),
            method_kwargs, True
        )
        roi = np.atleast_2d(regions)
        roi_shape = roi.shape
        if len(roi_shape) != 2 or roi_shape[1] != 2:
            raise ValueError('rois must be a two dimensional sequence of (start, stop) values')

        steps = _check_scalar(sampling, roi_shape[0], fill_scalar=True, dtype=np.intp)[0]
        if np.any(steps < 1):
            raise ValueError('all step sizes in "sampling" must be >= 1')

        x_sections = []
        y_sections = []
        x_mask = np.ones(self._len, dtype=bool)
        last_stop = -1
        include_first = True
        include_last = True
        for (start, stop), step in zip(roi, steps):
            if start is None:
                start = 0
            if stop is None:
                stop = self._len
            if start < last_stop:
                raise ValueError('Sections cannot overlap')
            else:
                last_stop = stop
            if start < 0 or stop < 0:
                raise ValueError('values in regions must be positive')
            elif stop > self._len:
                raise ValueError('values in regions must be less than len(data)')

            sections = (stop - start) // step
            if sections == 0:
                sections = 1  # will create one section using the midpoint
            indices = np.linspace(start, stop, sections + 1, dtype=np.intp)
            for left_idx, right_idx in zip(indices[:-1], indices[1:]):
                if left_idx == 0 and right_idx == 1:
                    include_first = False
                elif right_idx == self._len and left_idx == self._len - 1:
                    include_last = False
                y_sections.append(np.mean(y[left_idx:right_idx]))
                x_sections.append(np.mean(self.x[left_idx:right_idx]))
            x_mask[start:stop] = False

        # ensure first and last indices are included in the fit to avoid edge effects
        if include_first:
            x_mask[0] = True
        if include_last:
            x_mask[-1] = True
        x_sections.extend(self.x[x_mask])
        y_sections.extend(y[x_mask])
        x_fit = np.array(x_sections)
        sort_order = np.argsort(x_fit, kind='mergesort')
        x_fit = x_fit[sort_order]
        y_fit = np.array(y_sections)[sort_order]

        # param sorting will be wrong, but most params that need sorting will have
        # no meaning since they correspond to a truncated dataset
        with fitting_object._override_x(x_fit):
            baseline_fit, params = baseline_func(y_fit, **method_kws)

        baseline = np.interp(self.x, x_fit, baseline_fit)
        params.update({'x_fit': x_fit, 'y_fit': y_fit})
        if lam is not None and lam != 0:
            self._setup_whittaker(y, lam=lam, diff_order=diff_order)
            baseline = self.whittaker_system.solve(
                self.whittaker_system.add_diagonal(1.), baseline,
                overwrite_ab=True, overwrite_b=True
            )

        return baseline, params


_optimizers_wrapper = _class_wrapper(_Optimizers)


@_optimizers_wrapper
def collab_pls(data, average_dataset=True, method='asls', method_kwargs=None, x_data=None):
    """
    Collaborative Penalized Least Squares (collab-PLS).

    Averages the data or the fit weights for an entire dataset to get more
    optimal results. Uses any Whittaker-smoothing-based or weighted spline algorithm.

    Parameters
    ----------
    data : array-like, shape (M, N)
        An array with shape (M, N) where M is the number of entries in
        the dataset and N is the number of data points in each entry.
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
    x_data : array-like, shape (N,), optional
        The x values for the data. Not used by most Whittaker-smoothing algorithms.

    Returns
    -------
    baselines : np.ndarray, shape (M, N)
        An array of all of the baselines.
    params : dict
        A dictionary with the following items:

        * 'average_weights': numpy.ndarray, shape (N,)
            The weight array used to fit all of the baselines.
        * 'average_alpha': numpy.ndarray, shape (N,)
            Only returned if `method` is 'aspls' or 'pspline_aspls'. The
            `alpha` array used to fit all of the baselines for the
            :meth:`~Baseline.aspls` or :meth:`~Baseline.pspline_aspls` methods.

        Additional items depend on the output of the selected method. Every
        other key will have a list of values, with each item corresponding to a
        fit.

    Notes
    -----
    If `method` is 'aspls' or 'pspline_aspls', `collab_pls` will also calculate
    the `alpha` array for the entire dataset in the same manner as the weights.

    References
    ----------
    Chen, L., et al. Collaborative Penalized Least Squares for Background
    Correction of Multiple Raman Spectra. Journal of Analytical Methods
    in Chemistry, 2018, 2018.

    """


@_optimizers_wrapper
def optimize_extended_range(data, x_data=None, method='asls', side='both', width_scale=0.1,
                            height_scale=1., sigma_scale=1. / 12., min_value=2, max_value=8,
                            step=1, pad_kwargs=None, method_kwargs=None):
    """
    Extends data and finds the best parameter value for the given baseline method.

    Adds additional data to the left and/or right of the input data, and then iterates
    through parameter values to find the best fit. Useful for calculating the optimum
    `lam` or `poly_order` value required to optimize other algorithms.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    method : str, optional
        A string indicating the Whittaker-smoothing-based, polynomial, or spline method
        to use for fitting the baseline. Default is 'asls'.
    side : {'both', 'left', 'right'}, optional
        The side of the measured data to extend. Default is 'both'.
    width_scale : float, optional
        The number of data points added to each side is `width_scale` * N. Default
        is 0.1.
    height_scale : float, optional
        The height of the added Gaussian peak(s) is calculated as
        `height_scale` * max(`data`). Default is 1.
    sigma_scale : float, optional
        The sigma value for the added Gaussian peak(s) is calculated as
        `sigma_scale` * `width_scale` * N. Default is 1/12, which will make
        the Gaussian span +- 6 sigma, making its total width about half of the
        added length.
    min_value : int or float, optional
        The minimum value for the `lam` or `poly_order` value to use with the
        indicated method. If using a polynomial method, `min_value` must be an
        integer. If using a Whittaker-smoothing-based method, `min_value` should
        be the exponent to raise to the power of 10 (eg. a `min_value` value of 2
        designates a `lam` value of 10**2).
        Default is 2.
    max_value : int or float, optional
        The maximum value for the `lam` or `poly_order` value to use with the
        indicated method. If using a polynomial method, `max_value` must be an
        integer. If using a Whittaker-smoothing-based method, `max_value` should
        be the exponent to raise to the power of 10 (eg. a `max_value` value of 3
        designates a `lam` value of 10**3).
        Default is 8.
    step : int or float, optional
        The step size for iterating the parameter value from `min_value` to `max_value`.
        If using a polynomial method, `step` must be an integer.
    pad_kwargs : dict, optional
        A dictionary of options to pass to :func:`.pad_edges` for padding
        the edges of the data when adding the extended left and/or right sections.
        Default is None, which will use an empty dictionary.
    method_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the selected `method` function.
        Default is None, which will use an empty dictionary.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline calculated with the optimum parameter.
    method_params : dict
        A dictionary with the following items:

        * 'optimal_parameter': int or float
            The `lam` or `poly_order` value that produced the lowest
            root-mean-squared-error.
        * 'min_rmse': float
            The minimum root-mean-squared-error obtained when using
            the optimal parameter.

        Additional items depend on the output of the selected method.

    Raises
    ------
    ValueError
        Raised if `side` is not 'left', 'right', or 'both'.
    TypeError
        Raised if using a polynomial method and `min_value`, `max_value`, or
        `step` is not an integer.
    ValueError
        Raised if using a Whittaker-smoothing-based method and `min_value`,
        `max_value`, or `step` is greater than 100.

    Notes
    -----
    Based on the extended range penalized least squares (erPLS) method from [1]_.
    The method proposed by [1]_ was for optimizing lambda only for the aspls
    method by extending only the right side of the spectrum. The method was
    modified by allowing extending either side following [2]_, and for optimizing
    lambda or the polynomial degree for all of the affected algorithms in
    pybaselines.

    References
    ----------
    .. [1] Zhang, F., et al. An Automatic Baseline Correction Method Based on
           the Penalized Least Squares Method. Sensors, 2020, 20(7), 2015.
    .. [2] Krishna, H., et al. Range-independent background subtraction algorithm
           for recovery of Raman spectra of biological tissue. Journal of Raman
           Spectroscopy. 2012, 43(12), 1884-1894.

    """


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
    # Table 2 in reference
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


@_optimizers_wrapper
def adaptive_minmax(data, x_data=None, poly_order=None, method='modpoly',
                    weights=None, constrained_fraction=0.01, constrained_weight=1e5,
                    estimation_poly_order=2, method_kwargs=None):
    """
    Fits polynomials of different orders and uses the maximum values as the baseline.

    Each polynomial order fit is done both unconstrained and constrained at the
    endpoints.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    x_data : array-like, shape (N,), optional
        The x-values of the measured data. Default is None, which will create an
        array from -1 to 1 with N points.
    poly_order : int or Sequence(int, int) or None, optional
        The two polynomial orders to use for fitting. If a single integer is given,
        then will use the input value and one plus the input value. Default is None,
        which will do a preliminary fit using a polynomial of order `estimation_poly_order`
        and then select the appropriate polynomial orders according to [3]_.
    method : {'modpoly', 'imodpoly'}, optional
        The method to use for fitting each polynomial. Default is 'modpoly'.
    weights : array-like, shape (N,), optional
        The weighting array. If None (default), then will be an array with
        size equal to N and all values set to 1.
    constrained_fraction : float or Sequence(float, float), optional
        The fraction of points at the left and right edges to use for the
        constrained fit. Default is 0.01. If `constrained_fraction` is a sequence,
        the first item is the fraction for the left edge and the second is the
        fraction for the right edge.
    constrained_weight : float or Sequence(float, float), optional
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
        Additional keyword arguments to pass to :meth:`~Baseline.modpoly` or
        :meth:`~Baseline.imodpoly`. These include `tol`, `max_iter`, `use_original`,
        `mask_initial_peaks`, and `num_std`.

    Returns
    -------
    numpy.ndarray, shape (N,)
        The calculated baseline.
    params : dict
        A dictionary with the following items:

        * 'weights': numpy.ndarray, shape (N,)
            The weight array used for fitting the data.
        * 'constrained_weights': numpy.ndarray, shape (N,)
            The weight array used for the endpoint-constrained fits.
        * 'poly_order': numpy.ndarray, shape (2,)
            An array of the two polynomial orders used for the fitting.

    References
    ----------
    .. [3] Cao, A., et al. A robust method for automated background subtraction
           of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
           1199-1205.

    """


@_optimizers_wrapper
def custom_bc(data, x_data=None, method='asls', regions=((None, None),), sampling=1, lam=None,
              diff_order=2, method_kwargs=None):
    """
    Customized baseline correction for fine tuned stiffness of the baseline at specific regions.

    Divides the data into regions with variable number of data points and then uses other
    baseline algorithms to fit the truncated data. Regions with less points effectively
    makes the fit baseline more stiff in those regions.

    Parameters
    ----------
    data : array-like, shape (N,)
        The y-values of the measured data, with N data points.
    method : str, optional
        A string indicating the algorithm to use for fitting the baseline; can be any
        non-optimizer algorithm in pybaselines. Default is 'asls'.
    regions : array-like, shape (M, 2), optional
        The two dimensional array containing the start and stop indices for each region of
        interest. Each region is defined as ``data[start:stop]``. Default is ((None, None),),
        which will use all points.
    sampling : int or array-like, optional
        The sampling step size for each region defined in `regions`. If `sampling` is an
        integer, then all regions will use the same index step size; if `sampling` is an
        array-like, its length must be equal to `M`, the first dimension in `regions`.
        Default is 1, which will use all points.
    lam : float or None, optional
        The value for smoothing the calculated interpolated baseline using Whittaker
        smoothing, in order to reduce the kinks between regions. Default is None, which
        will not smooth the baseline; a value of 0 will also not perform smoothing.
    diff_order : int, optional
        The difference order used for Whittaker smoothing of the calculated baseline.
        Default is 2.
    method_kwargs : dict, optional
        A dictionary of keyword arguments to pass to the selected `method` function.
        Default is None, which will use an empty dictionary.

    Returns
    -------
    baseline : numpy.ndarray, shape (N,)
        The baseline calculated with the optimum parameter.
    params : dict
        A dictionary with the following items:
            * 'x_fit': numpy.ndarray, shape (P,)
                The truncated x-values used for fitting the baseline.
            * 'y_fit': numpy.ndarray, shape (P,)
                The truncated y-values used for fitting the baseline.

        Additional items depend on the output of the selected method.

    Raises
    ------
    ValueError
        Raised if `regions` is not two dimensional, if `sampling` is not the same length
        as `rois.shape[0]`, if any values in `sampling` or `regions` is less than 1, if
        segments in `regions` overlap, or if any value in `regions` is greater than the
        length of the input data.

    Notes
    -----
    Uses Whittaker smoothing to smooth the transitions between regions rather than LOESS
    as used in [4]_.

    Uses binning rather than direct truncation of the regions in order to get better
    results for noisy data.

    References
    ----------
    .. [4] Liland, K., et al. Customized baseline correction. Chemometrics and
            Intelligent Laboratory Systems, 2011, 109(1), 51-56.

    """
