# -*- coding: utf-8 -*-
"""High level methods for making better use of baseline algorithms.

Created on March 28, 2026
@author: Donald Erb

"""

import inspect


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
