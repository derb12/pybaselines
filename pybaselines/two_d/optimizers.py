# -*- coding: utf-8 -*-
"""High level functions for making better use of baseline algorithms.

Functions in this module make use of other baseline algorithms in
pybaselines to provide better results or optimize parameters.

Created on January 14, 2024
@author: Donald Erb

"""

from collections import defaultdict
import itertools
from math import ceil

import numpy as np

from .._nd.optimizers import _OptimizersNDMixin
from .._validation import _check_optional_array, _check_scalar, _get_row_col_values
from ..api import Baseline
from ..optimizers import _determine_polyorders, _optimize_ed, _param_grid
from ..utils import _sort_array2d, _wrss
from ._algorithm_setup import _Algorithm2D


class _Optimizers(_Algorithm2D, _OptimizersNDMixin):
    """A base class for all optimizer algorithms."""

    @_Algorithm2D._handle_io(skip_sorting=True, mask_support=0)
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
            and then select the appropriate polynomial orders according to [1]_.
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
            :meth:`~.Baseline2D.modpoly` or :meth:`~.Baseline2D.imodpoly`. These include
            `tol`, `max_iter`, `use_original`, `mask_initial_peaks`, and `num_std`.

        Returns
        -------
        numpy.ndarray, shape (M, N)
            The calculated baseline.
        params : dict
            A dictionary with the following items:

            * 'poly_order': numpy.ndarray, shape (2,)
                An array of the two polynomial orders used for the fitting.
            * 'method_params': dict[str, list]
                A dictionary containing the output parameters for each individual fit.
                Keys will depend on the selected method and will have a list of values,
                with each item corresponding to a fit.

        Raises
        ------
        ValueError
            Raised if ``constrained_fraction`` is outside of the range [0, 1].

        References
        ----------
        .. [1] Cao, A., et al. A robust method for automated background subtraction
            of tissue fluorescence. Journal of Raman Spectroscopy, 2007, 38,
            1199-1205.

        """
        y, optimizer_obj, method_kws = self._setup_optimizer(
            data, method, method_param={None: 'poly_order'}, method_kwargs=method_kwargs,
            copy_kwargs=False, ensure_new=True, needed_params=('weights',)
        )
        sort_weights = weights is not None
        weight_array = _check_optional_array(
            self._shape, weights, check_finite=self._check_finite, ensure_1d=False, axis=slice(None)
        )
        if poly_order is None:
            poly_orders = _determine_polyorders(
                y, estimation_poly_order, weight_array, optimizer_obj.method_call,
                optimizer_obj.fitter, **method_kws
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
        constrained_weights[:ceil(self._shape[0] * constrained_fractions[0])] = weightings[0]
        constrained_weights[:, :ceil(self._shape[1] * constrained_fractions[2])] = weightings[2]
        constrained_weights[
            self._shape[0] - ceil(self._shape[0] * constrained_fractions[1]):
        ] = weightings[1]
        constrained_weights[
            :, self._shape[1] - ceil(self._shape[1] * constrained_fractions[3]):
        ] = weightings[3]
        # and now change back to original ordering
        if sort_weights:
            weight_array = _sort_array2d(weight_array, self._inverted_order)
            constrained_weights = _sort_array2d(constrained_weights, self._inverted_order)

        params = {
            'poly_order': poly_orders, 'method_params': defaultdict(list)
        }
        # order of inputs is (poly_orders[0], weight_array), (poly_orders[0], constrained_weights),
        # (poly_orders[1], weight_array), (poly_orders[1], constrained_weights)
        baselines = np.empty((4, *self._shape))
        for i, (p_order, weight) in enumerate(
            itertools.product(poly_orders, (weight_array, constrained_weights))
        ):
            baselines[i], method_params = optimizer_obj.method_call(
                data=y, poly_order=p_order, weights=weight, **method_kws
            )
            for key, value in method_params.items():
                params['method_params'][key].append(value)

        return np.maximum.reduce(baselines), params

    @_Algorithm2D._handle_io(skip_sorting=True, mask_support=0)
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
        baseline = np.zeros(self._shape)
        fit_data = data
        params = {}
        has_mask = self.mask is not None
        for i, axis in enumerate(axes):
            fitter = Baseline(
                (self.x, self.z)[axis], check_finite=self._check_finite, assume_sorted=True,
                output_dtype=self._dtype, strict_mask=self._strict_mask
            )
            fitter.banded_solver = self.banded_solver
            baseline_func = getattr(fitter, method.lower())
            params[f'params_{keys[axis]}'] = defaultdict(list)
            if axis == 0:
                indices = ((..., idx) for idx in range(self._shape[1]))
            else:
                indices = range(self._shape[0])

            partial_baseline = np.zeros(self._shape)
            for index in indices:
                if has_mask:
                    fitter.mask = self.mask[index]
                partial_baseline[index], method_params = baseline_func(
                    fit_data[index], **method_kwargs[i]
                )
                for key, val in method_params.items():
                    params[f'params_{keys[axis]}'][key].append(val)

            baseline += partial_baseline
            fit_data = data - partial_baseline
            # TODO should the individual baselines be deprecated from the params? If they were
            # wanted, could just call this twice with each axis
            params[f'baseline_{keys[axis]}'] = partial_baseline

        return baseline, params

    @_Algorithm2D._handle_io(skip_sorting=True)
    def optimize_pls(self, data, method='arpls', opt_method='V-Curve', min_value=4., max_value=7.,
                     step=0.5, method_kwargs=None, euclidean=False, rho=None, n_samples=0):
        """
        Optimizes the regularization parameters for penalized least squares methods.

        Parameters
        ----------
        data : array-like, shape (N,)
            The y-values of the measured data, with N data points.
        method : str, optional
            A string indicating the Whittaker-smoothing or spline method
            to use for fitting the baseline. Default is 'arpls'.
        opt_method : {'V-Curve', 'U-Curve', 'GCV', 'BIC'}, optional
            The optimization method used to optimize `lam`. Supported methods are:

            * 'V-Curve'
            * 'U-Curve'
            * 'GCV'
            * 'BIC'

            Details on each optimization method are in the Notes section below.
        min_value : float or tuple[float, float], optional
            The minimum value for `lam` to use with the indicated method. Should
            be the exponent to raise to the power of 10 (eg. a `min_value` value of 2
            designates a `lam` value of 10**2). Default is 4.
        max_value : float or tuple[float, float], optional
            The maximum values for `lam` to use with the indicated method. Should
            be the exponent to raise to the power of 10 (eg. a `max_value` value of 3
            designates a `lam` value of 10**3). Default is 7.
        step : float or tuple[float, float], optional
            The step size for iterating the parameter value from `min_value` to `max_value`.
            Should be the exponent to raise to the power of 10 (eg. a `step` value of 1
            designates a `lam` value of 10**1). Default is 0.5.
        method_kwargs : dict, optional
            A dictionary of keyword arguments to pass to the selected `method` function.
            Default is None, which will use an empty dictionary.
        euclidean : bool, optional
            Only used if `opt_method` is 'U-curve'. If False (default), the optimization metric
            is the minimum of the sum of the normalized fidelity and penalty values [1]_, which is
            equivalent to the minimum graph distance from the origin. If True, the metric is the
            euclidean distance from the origin, similar to [2]_ and [3]_.
        rho : float, optional
            Only used if `opt_method` is 'GCV'. The stabilization parameter for the modified
            generalized cross validation (mGCV) criteria. A value of 1 defines normal GCV, while
            higher values of `rho` stabilize the scores to make a single, global minima value
            more likely (when applied to smoothing). If None (default), the value of `rho` will
            be selected following [4]_, with the value being 1.3 if ``len(data)`` is less than
            100, otherwise 2.
        n_samples : int, optional
            Only used if `opt_method` is 'GCV' or 'BIC'. If 0 (default), will calculate the
            analytical trace. Otherwise, will use stochastic trace estimation with a matrix of
            (``M * N``, `n_samples`) Rademacher random variables (ie. either -1 or 1).

        Returns
        -------
        baseline : numpy.ndarray, shape (M, N)
            The baseline calculated with the optimum parameter.
        params : dict
            A dictionary with the following items:

            * 'optimal_parameter': tuple[float, float]
                The `lam` values that minimized the computed metric.
            * 'metric': numpy.ndarray, shape (P, Q)
                The computed metric for each pair of `lam` values tested.
            * 'method_params': dict
                A dictionary containing the output parameters for the optimal fit.
                Items will depend on the selected `method`.
            * 'fidelity': numpy.ndarray, shape (P, Q)
                Only returned if `opt_method` is 'U-curve'. The computed non-normalized
                fidelity term for each pair of `lam` values tested. For
                most algorithms within pybaselines, this is equivalent to the weighted residual
                sum of squares (eg. ``sum(weights * (data - baseline)**2)``)
            * 'penalty': numpy.ndarray, shape (P, Q)
                Only returned if `opt_method` is 'U-curve'. The computed non-normalized penalty
                values for each pair of `lam` values tested.
            * 'wrss': numpy.ndarray, shape (P, Q)
                Only returned if `opt_method` is 'GCV' or 'BIC'. The weighted residual sum of
                squares (eg. ``sum(weights * (data - baseline)**2)``) for each pair of `lam`
                values tested.
            * 'trace': numpy.ndarray, shape (P, Q)
                Only returned if `opt_method` is 'GCV' or 'BIC. The computed trace of the smoother
                matrix for each pair of `lam` values tested, which signifies the effective dimension
                for the system.

        Raises
        ------
        ValueError
            Raised if `opt_method` is 'GCV' and the input `rho` is less than 1.
        NotImplementedError
            Raised if `method` is 'beads' and `opt_method` is 'GCV' or 'BIC'.

        Notes
        -----
        `opt_method` 'U-Curve' requires that the sum of the normalized penalty and fidelity values
        is roughly 'U' shaped (see Figure 5 in [1]_), which depends on appropriate selection of
        `min_value` and `max_value` such that penalty continually decreases and fidelity
        continually increases as `lam` increases.

        For `opt_method` 'U-Curve', the multipliers on `lam` used in methods `drpls` or `aspls`,
        ``(1 - eta * weights)`` and ``alpha``, respectively, are omitted from the penalty term.
        Otherwise, the penalty term shows little change with varying `lam` and gives bad results.
        Likewise, for method='iasls', the penalty term from `lam_1` is omitted since its gradient
        with respect to `lam` is assumed to be 0. More advanced optimization varying both `lam`
        and `lam_1` is possible, but not supported within this method.

        Uses a grid search for optimization since the objective functions for all supported
        `opt_method` inputs are highly non-smooth (ie. many local minima) when performing
        baseline correction, due to the reliance of calculated weights on the input `lam`.
        Scalar minimization using :func:`scipy.optimize.minimize_scalar` was found to
        perform okay in most cases, but it would also not allow some methods like 'U-Curve'
        which requires calculating with all `lam` values before computing the objective.

        The range of values to test is generated using
        ``numpy.arange(min_value, max_value, step)``, so `max_value` is likely not included in
        the range of tested values.

        References
        ----------
        .. [1] Park, A., et al. Automatic Selection of Optimal Parameter for Baseline Correction
                using Asymmetrically Reweighted Penalized Least Squares. Journal of the Institute
                of Electronics and Information Engineers, 2016, 53(3), 124-131.
        .. [2] Belge, M., et al. Efficient determination of multiple regularization parameters in
                a generalized L-curve framework. Inverse Problems, 2002, 18, 1161-1183.
        .. [3] Andriyana, Y., et al. P-splines quantile regression estimation in varying
                coefficient models. TEST, 2014, 23, 153-194.
        .. [4] Lukas, M., et al. Practical use of robust GCV and modified GCV for spline
                smoothing. Computational Statistics, 2016, 31, 269-289.

        """
        y, optimizer_obj, method_kws = self._setup_optimizer(
            data, method, method_param={'beads': 'alpha', None: 'lam'},
            method_kwargs=method_kwargs, copy_kwargs=False
        )
        if 'lam' in method_kws:
            # TODO maybe just warn and pop out instead? Would need to copy input kwargs in that
            # case so that the original input is not modified
            raise ValueError('lam must not be specified within method_kwargs')
        min_rows, min_cols = _check_scalar(min_value, desired_length=2, fill_scalar=True)[0]
        max_rows, max_cols = _check_scalar(max_value, desired_length=2, fill_scalar=True)[0]
        step_rows, step_cols = _check_scalar(step, desired_length=2, fill_scalar=True)[0]
        lam_range_r = _param_grid(min_rows, max_rows, step_rows, polynomial_fit=False)
        lam_range_c = _param_grid(min_cols, max_cols, step_cols, polynomial_fit=False)
        selected_method = opt_method.lower().replace('-', '_').replace('_', '')
        if selected_method in ('vcurve', 'ucurve'):
            baseline, params = _optimize_lcurve2d(
                y, selected_method, optimizer_obj, method_kws, lam_range_r, lam_range_c, euclidean
            )
        elif selected_method in ('gcv', 'bic'):
            lam_range = np.stack(
                np.meshgrid(lam_range_r, lam_range_c, indexing='ij'), axis=-1
            ).reshape(-1, 2)
            baseline, params = _optimize_ed(
                y, selected_method, optimizer_obj, method_kws, lam_range, rho, n_samples
            )
            params['optimal_parameter'] = tuple(params['optimal_parameter'])
            output_shape = (lam_range_r.size, lam_range_c.size)
            for key in ('wrss', 'trace', 'metric'):
                params[key] = params[key].reshape(output_shape)
        else:
            raise ValueError(f'{opt_method} is not a supported opt_method input')

        return baseline, params


def _optimize_lcurve2d(y, opt_method, optimizer_obj, method_kws, lam_range_r, lam_range_c,
                       euclidean):
    """
    Performs L-curve optimization based on the fit fidelity and penalty.

    Parameters
    ----------
    y : _type_
        _description_
    opt_method : _type_
        _description_
    method : _type_
        _description_
    method_kws : _type_
        _description_
    baseline_func : _type_
        _description_
    baseline_obj : _Algorithm
        _description_
    lam_range : _type_
        _description_
    euclidean : bool, optional
        _description_. Default is False.

    Returns
    -------
    _type_
        _description_

    References
    ----------
    .. [1] Park, A., et al. Automatic Selection of Optimal Parameter for Baseline Correction using
           Asymmetrically Reweighted Penalized Least Squares. Journal of the Institute of
           Electronics and Information Engineers, 2016, 53(3), 124-131.
    .. [2] Andriyana, Y., et al. P-splines quantile regression estimation in varying coefficient
           models. TEST, 2014, 23(1), 153-194.

    """
    method_signature = optimizer_obj.method_signature.parameters
    spline_fit = 'spline_degree' in method_signature
    if (
        'num_eigens' in method_signature
        and None not in _check_scalar(
                method_kws.get('num_eigens', method_signature['num_eigens'].default),
                2, fill_scalar=True
            )[0]
    ):
        eigen_fit = True
    else:
        eigen_fit = False
    using_drpls = 'drpls' in optimizer_obj.method
    diff_order = _check_scalar(
        method_kws.get('diff_order', method_signature['diff_order'].default), 2, fill_scalar=True
    )[0]

    n_lams = (lam_range_r.size, lam_range_c.size)
    penalty_rows = np.empty(n_lams)
    penalty_cols = np.empty(n_lams)
    fidelity = np.empty(n_lams)
    for i, lam_r in enumerate(lam_range_r):
        for j, lam_c in enumerate(lam_range_c):
            fit_lams = (10**lam_r, 10**lam_c)
            fit_baseline, fit_params = optimizer_obj.method_call(
                y, lam=fit_lams, **method_kws
            )
            if eigen_fit:
                # approximately the same as taking the finite difference in each dimension, but
                # use the eigenvalues just for completeness; since eigenvalue penalty is a
                # diagonal matrix and coef is 1D, then:
                # coef.T @ diags(eigenvalues) @ coef == sum(eigenvalues * coef**2)
                system = fit_params['result']._penalized_object
                coef_sq = system.coef**2
                fit_penalty_r = (system.penalty_rows / system.lam[0]) @ coef_sq
                fit_penalty_c = (system.penalty_columns / system.lam[1]) @ coef_sq
            else:
                if spline_fit:
                    penalized_object = fit_params['result'].tck[1]  # the spline coefficients
                else:
                    # have to ensure sort order of the fit baseline since
                    # diff(y_ordered) != diff(y_disordered); spline coefficients are always
                    # sorted since they correspond to sorted x-values
                    penalized_object = _sort_array2d(fit_baseline, optimizer_obj.fitter._sort_order)

                diff_r = np.diff(penalized_object, diff_order[0], axis=0)
                diff_c = np.diff(penalized_object, diff_order[1], axis=1)
                fit_penalty_r = np.einsum('ij,ij->', diff_r, diff_r)
                fit_penalty_c = np.einsum('ij,ij->', diff_c, diff_c)

            fit_fidelity = _wrss(y - fit_baseline, fit_params['weights'])
            if using_drpls:
                if spline_fit:  # still need to sort the baseline
                    sorted_baseline = _sort_array2d(fit_baseline, optimizer_obj.fitter._sort_order)
                else:
                    sorted_baseline = penalized_object
                additional_fidelity_r = np.diff(sorted_baseline, 1, axis=0)
                additional_fidelity_c = np.diff(sorted_baseline, 1, axis=1)
                fit_fidelity += (
                    np.einsum('ij,ij->', additional_fidelity_r, additional_fidelity_r)
                    + np.einsum('ij,ij->', additional_fidelity_c, additional_fidelity_c)
                )

            penalty_rows[i, j] = fit_penalty_r
            penalty_cols[i, j] = fit_penalty_c
            fidelity[i, j] = fit_fidelity

    # add fidelity and penalty to params before further processing
    params = {'fidelity': fidelity, 'penalty_rows': penalty_rows, 'penalty_columns': penalty_cols}
    # TODO: for both metrics, need to check size along each axis and skip calcs accordingly
    if opt_method == 'ucurve':
        if fidelity.size > 1:
            penalty_rows = (penalty_rows - penalty_rows.min()) / np.ptp(penalty_rows)
            penalty_cols = (penalty_cols - penalty_cols.min()) / np.ptp(penalty_cols)
            fidelity = (fidelity - fidelity.min()) / np.ptp(fidelity)
        if euclidean:
            metric = np.sqrt(fidelity**2 + penalty_rows**2 + penalty_cols**2)
        else:  # graph distance from the origin, ie. only travelling along x, y, and z axes
            metric = fidelity + penalty_rows + penalty_cols
    elif opt_method == 'vcurve':
        if fidelity.size > 1:
            step_r = np.log10(lam_range_r[1] - lam_range_r[0])
            step_c = np.log10(lam_range_c[1] - lam_range_c[0])

            penalty_rows_grad = _gradient_magnitude(np.log10(penalty_rows), step_r, step_c)
            penalty_cols_grad = _gradient_magnitude(np.log10(penalty_cols), step_r, step_c)
            fidelity_grad = _gradient_magnitude(np.log10(fidelity), step_r, step_c)

            metric = np.sqrt(penalty_rows_grad**2 + penalty_cols_grad**2 + fidelity_grad**2)

        else:
            metric = np.zeros((1, 1))

    best_idx = np.unravel_index(np.argmin(metric), metric.shape)
    best_lam = (10**lam_range_r[best_idx[0]], 10**lam_range_c[best_idx[1]])
    baseline, best_params = optimizer_obj.method_call(y, lam=best_lam, **method_kws)
    params.update({'optimal_parameter': best_lam, 'metric': metric, 'method_params': best_params})

    return baseline, params


def _gradient_magnitude(array, row_step=1., col_step=1.):
    """
    Calculates the magnitude of the gradient in two dimensions.

    Parameters
    ----------
    array : numpy.ndarray, shape (N, M)
        The array to calculate the gradient of.
    row_step : float, optional
        The step size along the rows. Default is 1.
    col_step : float, optional
        The step size along the columns. Default is 1.

    Returns
    -------
    numpy.ndarray, shape (N, M)
        The magnitude of the gradient of the input array.

    """
    row_gradient, col_gradient = np.gradient(array, row_step, col_step)
    return np.sqrt(row_gradient**2 + col_gradient**2)
