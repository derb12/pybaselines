# -*- coding: utf-8 -*-
"""Different techniques for fitting baselines to experimental data.

Polynomial
    1) ModPoly (Modified Polynomial)
    2) IModPoly (Improved Modified Polynomial)

Created on Feb. 27, 2021

@author: Donald Erb

"""

import numpy as np

from . import utils


def poly(data, x_data, poly_order=2, weights=None):
    """
    Computes a polynomial that fits the data.

    Parameters
    ----------
    data : [type]
        [description]
    x_data : [type]
        [description]
    poly_order : int, optional
        [description]. Default is 2.
    weights : [type], optional
        [description]. Default is None.

    Returns
    -------
    np.ndarray
        The background data from the polynomial.
    dict
        A dictionary containing the keys:

        * 'weights'
              The array of weights used.

    Notes
    -----
    To only fit regions without peaks, supply a weight array with zero values
    at the indices where peaks are located.

    """
    x, y = utils.get_array(x_data, data)
    if weights is not None:
        w = weights
    else:
        w = np.ones(y.shape[0])

    return np.polynomial.Polynomial.fit(x, y, poly_order, w=w)(x), {'weights': w}


def modpoly(data, x_data, poly_order=2, tol=0.001, max_iter=500, weights=None,
            mask_initial_peaks=False, use_original=False):
    """
    The ModPoly baseline algorithm.

    Adapted from:
    C.A. Lieber, A. Mahadevan-Jansen, Automated method
    for subtraction of fluorescence from biological raman spectra,
    Applied Spectroscopy 57(11) (2003) 1363-1367.

    and

    Gan, F., et al. Baseline correction by improved iterative polynomial fitting
    with automatic threshold. Chemometrics and Intelligent Laboratory Systems, 82
    (2006) 59-65.

    The mask_initial_peaks idea was adopted from the IModPoly:
    Zhao, J., et al., Automated Autofluorescence Background Subtraction
    Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy
    61(11) (2007) 1225-1232.

    use_original=True is Lieber's method, and use_original=False is Gan's method.

    """
    x, y = utils.get_array(x_data, data)
    if use_original:
        y0 = y
    if weights is not None:
        w = weights.copy()
    else:
        w = np.ones(y.shape[0])

    z = np.polynomial.Polynomial.fit(x, y, poly_order)(x)
    if mask_initial_peaks:
        w[z >= y] = 0

    for i in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z)
        z_new = np.polynomial.Polynomial.fit(x, y, poly_order, w=w)(x)
        if utils.relative_difference(z, z_new) < tol:
            break
        z = z_new

    return z, {'weights': w}


def imodpoly(data, x_data, poly_order=2, tol=0.001, max_iter=500, weights=None,
             mask_initial_peaks=True, use_original=False):
    """
    The IModPoly baseline algorithm.

    Adapted from:
    Zhao, J., et al., Automated Autofluorescence Background Subtraction
    Algorithm for Biomedical Raman Spectroscopy, Applied Spectroscopy
    61(11) (2007) 1225-1232.

    The use_original term is adapted from the ModPoly algorithm (is False for
    Zhao's original IModPoly implementation):
    C.A. Lieber, A. Mahadevan-Jansen, Automated method
    for subtraction of fluorescence from biological raman spectra,
    Applied Spectroscopy 57(11) (2003) 1363-1367.

    """
    x, y = utils.get_array(x_data, data)
    if use_original:
        y0 = y
    if weights is not None:
        w = weights.copy()
    else:
        w = np.ones(y.shape[0])

    z = np.polynomial.Polynomial.fit(x, y, poly_order)(x)
    deviation = np.std(y - z)
    if mask_initial_peaks:
        w[z + deviation >= y] = 0

    for i in range(max_iter - 1):
        y = np.minimum(y0 if use_original else y, z + deviation)
        z = np.polynomial.Polynomial.fit(x, y, poly_order, w=w)(x)
        new_deviation = np.std((y0 if use_original else y) - z)
        # use new_deviation as dividing term in relative difference
        if utils.relative_difference(new_deviation, deviation) < tol:
            break
        deviation = new_deviation

    return z, {'weights': w}
