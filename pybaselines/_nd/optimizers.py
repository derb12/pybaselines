# -*- coding: utf-8 -*-
"""High level methods for making better use of baseline algorithms.

Created on March 28, 2026
@author: Donald Erb

"""

from collections import defaultdict
import inspect

import numpy as np

from ._algorithm_setup import _handle_io


class _OptimizerHelper:
    """An object for optimizer-type methods to use for simplified usage.

    Attributes
    ----------
    fitter : _Algorithm or _Algorithm2D
        The object to use for fitting.
    method : str
        The method being used, in lowercase.
    method_call : Callable
        The actual method to use for fitting.
    method_param : str or None
        The parameter key that is used by the optimizer method. Is None if no
        key is required for the optimizer method being used.

    """

    def __init__(self, method, current_fitter, ensure_new=False, method_param=None,
                 needed_params=None):
        """
        Initializes the object.

        Parameters
        ----------
        method : str
            The string name of the desired function, like 'asls'. Case does not matter.
        current_fitter : _Algorithm or _Algorithm2D
            The current object used for fitting. May or may not be used for the actual
            fitting depending on the indicated `method` and `ensure_new` inputs.
        ensure_new : bool, optional
            If True, will ensure that the `fitter` and `method_call` attributes
            correspond to a new object rather than `current_fitter`. This is to ensure
            thread safety for methods which would modify internal state not typically
            assumed to change when using threading, such as changing polynomial degrees.
            Default is False.
        method_param : dict, optional
            A dictionary indicating potential parameter keys to use, with the default having
            a key of None. For example, a `method_param` of {'method1': 'a', None: ('b', 'c')}
            would specify that parameter 'a' should be used for a `method` of'method1'; otherwise,
            either 'b' or 'c' could be potential parameters, which would then be filtered by
            looking at the signature of the indicated method. Default is None, which indicates
            that the optimizer method being used does not require any parameter key.
        needed_params : Iterable, optional
            An iterature of other necessary parameter keys that the method must have in its
            signature. For example `['weights', 'tol']` would error if either 'weights' or 'tol'
            are not valid inputs for the specified `method`. Default is None.

        Raises
        ------
        ValueError
            Raised if the indicated `method` does not contain the appropriate parameters
            specified in `method_param` and `needed_params`.
        TypeError
            Raised if `method_param` gives more than one parameter for the given `method`,
            which indicates an internal issue.

        """
        self.method = method.lower()
        self.fitter = current_fitter._spawn_fitter(self.method, ensure_new=ensure_new)
        self.method_call = getattr(self.fitter, self.method)
        self._method_signature = None
        self.method_param = None

        if method_param is not None:
            param = method_param[self.method if self.method in method_param else None]
            signature_params = self.method_signature.parameters
            if isinstance(param, str):
                if param not in signature_params:
                    raise ValueError((
                        f'{method} is not a supported method because it is missing the '
                        f'required parameter: {param}'
                    ))
                self.method_param = param
            else:  # multiple valid keys
                possible_params = [key for key in param if key in signature_params]
                if not possible_params:
                    raise ValueError((
                        f'{method} is not a supported method because it is missing the '
                        f'required parameter: {" or ".join(param)}'
                    ))
                elif len(possible_params) > 1:  # something internally set wrong
                    raise TypeError((
                        f'expected one parameter key for {method}, but instead '
                        f'got {" and ".join(possible_params)}'
                    ))
                self.method_param = possible_params[0]

        if needed_params is not None:
            missing = [
                key for key in needed_params if key not in self.method_signature.parameters
            ]
            if missing:
                raise ValueError((
                    f'{method} is not a supported method because it is missing the '
                    f'required parameters: {", ".join(missing)}'
                ))

    @property
    def module(self):
        """
        The module the method is defined in.

        Returns
        -------
        str
            The method's module, not including the full path. For example,
            `method` 'modpoly' would give a `module` of 'polynomial' rather
            than 'pybaselines.polynomial' or 'pybaselines.two_d.polynomial'.

        """
        return inspect.getmodule(self.method_call).__name__.split('.')[-1]

    @property
    def method_signature(self):
        """
        The signature of the corresponding method.

        Lazy call since this is not always needed.

        Returns
        -------
        inspect.Signature
            The method's signature.

        """
        if self._method_signature is None:
            self._method_signature = inspect.signature(self.method_call)
        return self._method_signature


class _OptimizersNDMixin:
    """A mixin class for providing optimizer methods for 1D and 2D."""

    @_handle_io(ensure_dims=False, skip_sorting=True)
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
                :meth:`~.Baseline2D.aspls`.
            * 'method_params': dict[str, list]
                A dictionary containing the output parameters for each individual fit.
                Keys will depend on the selected method and will have a list of values,
                with each item corresponding to a fit.

        Raises
        ------
        ValueError
            Raised if the input data is not three dimensional.

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
        dataset, optimizer_obj, method_kws = self._setup_optimizer(
            data, method, method_param={None: 'weights'}, method_kwargs=method_kwargs,
            copy_kwargs=True
        )
        if dataset.ndim != len(self._shape) + 1:
            if len(self._shape) == 1:
                expected_shape = '(number of measurements, number of points in "data")'
            else:
                expected_shape = '(number of measurements, rows of "data", columns of "data")'
            raise ValueError((
                f'the input data must have a shape of {expected_shape}, but instead has a shape '
                f'of {dataset.shape}'
            ))
        # if using aspls or pspline_aspls, also need to calculate the alpha array
        # for the entire dataset
        calc_alpha = optimizer_obj.method in ('aspls', 'pspline_aspls')

        # step 1: calculate weights for the entire dataset
        if average_dataset:
            _, fit_params = optimizer_obj.method_call(np.mean(dataset, axis=0), **method_kws)
            method_kws['weights'] = fit_params['weights']
            if calc_alpha:
                method_kws['alpha'] = fit_params['alpha']
        else:
            weights = np.empty(dataset.shape)
            if calc_alpha:
                alpha = np.empty(dataset.shape)
            for i, entry in enumerate(dataset):
                _, fit_params = optimizer_obj.method_call(entry, **method_kws)
                # TODO should this also try looking at mask? Does this work
                # well for classifiers outside of fabc?
                weights[i] = fit_params['weights']
                if calc_alpha:
                    alpha[i] = fit_params['alpha']
            method_kws['weights'] = np.mean(weights, axis=0)
            if calc_alpha:
                method_kws['alpha'] = np.mean(alpha, axis=0)

        # step 2: use the dataset weights from step 1 (stored in method_kws['weights'])
        # to fit each individual data entry; set tol to infinity so that only one
        # iteration is done and new weights are not calculated
        if (
            'tol' in optimizer_obj.method_signature.parameters
            and optimizer_obj.method not in ('mpls', 'pspline_mpls')
        ):
            method_kws['tol'] = np.inf
        if 'tol_2' in optimizer_obj.method_signature.parameters:  # brpls
            method_kws['tol_2'] = np.inf
        baselines = np.empty(dataset.shape)
        params = {'average_weights': method_kws['weights'], 'method_params': defaultdict(list)}
        if calc_alpha:
            params['average_alpha'] = method_kws['alpha']
        if optimizer_obj.method == 'fabc':
            # set weights as mask so it just fits the data
            method_kws['weights_as_mask'] = True

        for i, entry in enumerate(dataset):
            baselines[i], param = optimizer_obj.method_call(entry, **method_kws)
            for key, value in param.items():
                params['method_params'][key].append(value)

        return baselines, params
