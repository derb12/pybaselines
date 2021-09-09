# -*- coding: utf-8 -*-
"""Examples of fitting using various baselines in pybaselines.

Created on March 11, 2021

@author: Donald Erb

"""


if __name__ == '__main__':

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('This file requires matplotlib to run')
        raise
    import numpy as np

    from pybaselines.classification import dietrich, fastchrom, golotvin, std_distribution
    from pybaselines.misc import beads
    from pybaselines.morphological import (
        amormol, imor, mor, mormol, mpls, mpspline, mwmv, rolling_ball, tophat
    )
    from pybaselines.optimizers import adaptive_minmax, optimize_extended_range
    from pybaselines.polynomial import (
        goldindec, imodpoly, loess, modpoly, penalized_poly, poly, quant_reg
    )
    from pybaselines.spline import corner_cutting, irsqr, mixture_model
    from pybaselines.utils import gaussian
    from pybaselines.whittaker import (
        airpls, arpls, asls, aspls, derpsalsa, drpls, iarpls, iasls, psalsa
    )
    try:
        from pybaselines.smooth import noise_median, snip, swima
    except AttributeError:
        # pybaselines.window was renamed in pybaselines.smooth in version 0.6.0
        from pybaselines.window import noise_median, snip, swima

    x = np.linspace(100, 4200, 1000)
    np.random.seed(0)  # set random seed
    signal = (
        gaussian(x, 2, 700, 50)
        + gaussian(x, 3, 1200, 150)
        + gaussian(x, 5, 1600, 100)
        + gaussian(x, 4, 2500, 50)
        + gaussian(x, 7, 3300, 100)
    )
    true_baseline = (
        10 + 0.001 * x  # polynomial baseline
        + gaussian(x, 6, 2000, 2000)  # gaussian baseline
    )
    noise = np.random.normal(0, 0.2, x.size)

    y = signal + true_baseline + noise
    non_peak_mask = (x < 600) | ((x > 1900) & (x < 2500)) | ((x > 2600) & (x < 3100)) | (x > 3600)

    algorithms = {
        'whittaker': (
            (asls, (y, 1e7, 0.005)),
            (iasls, (y, x, 1e6, 0.04, 1e-3)),
            (airpls, (y, 1e7)),
            (drpls, (y, 1e8)),
            (arpls, (y, 1e7)),
            (iarpls, (y, 1e6)),
            (aspls, (y, 1e8)),
            (psalsa, (y, 1e7)),
            (derpsalsa, (y, 1e6, 0.1))
        ),
        'polynomial': (
            (poly, (y, x, 3)),
            (poly, (y, x, 3), {'weights': non_peak_mask}, ', fit only non-peaks'),
            (modpoly, (y, x, 3)),
            (imodpoly, (y, x, 3)),
            (penalized_poly, (y, x, 3), {'threshold': 0.02 * (max(y) - min(y))}),
            (loess, (y, x, 0.6)),
            (quant_reg, (y, x, 3, 0.2)),
            (goldindec, (y, x, 3), {'peak_ratio': 0.3})
        ),
        'morphological': (
            (mpls, (y, 100, 1e7, 0.002)),
            (mor, (y, 100)),
            (imor, (y, 25)),
            (mormol, (y, 100), {'pad_kwargs': {'extrapolate_window': 50}, 'smooth_half_window': 3}),
            (amormol, (y, 45), {'pad_kwargs': {'extrapolate_window': 50}}),
            (rolling_ball, (y, 125, 100), {'pad_kwargs': {'extrapolate_window': 50}}),
            (mwmv, (y, 80)),
            (tophat, (y, 125)),
            (mpspline, (y, 100), {'pad_kwargs': {'extrapolate_window': 50}})
        ),
        'smooth': (
            (noise_median, (y, 250, 150, 50), {'extrapolate_window': 50}),
            (snip, (y, 40), {'extrapolate_window': 50}),
            (snip, (y, 40, True, 1), {'extrapolate_window': 50}, ', decreasing & smooth'),
            (swima, (y,), {'extrapolate_window': 50})
        ),
        'optimizers': (
            (optimize_extended_range, (y, x, 'aspls', 'both', 0.25),
             {'pad_kwargs': {'extrapolate_window': 50}}),
            (adaptive_minmax, (y, x), {'constrained_fraction': 0.05}),
        ),
        'misc': (
            (beads, (y, 0.005, 0.01, 0.01, 0.01)),
        ),
        'spline': (
            (mixture_model, (y, 1e8)),
            (irsqr, (y, 1e8, 0.15)),
            (corner_cutting, (y, x, 9)),
        ),
        'classification': (
            (dietrich, (y, x, 11, 1.9)),
            (golotvin, (y, x, 45, 8)),
            (std_distribution, (y, x, 45)),
            (fastchrom, (y, x, 45))
        )
    }

    for module, func_calls in algorithms.items():
        fig, ax = plt.subplots(num=module, tight_layout=True)
        ax.plot(x, y, label='data', lw=1.5)
        ax.plot(x, true_baseline, label='true baseline', lw=4)
        for call in func_calls:
            if len(call) == 2:
                func, args = call
                kwargs = {}
                extra_naming = ''
            elif len(call) == 3:
                func, args, kwargs = call
                extra_naming = ''
            else:
                func, args, kwargs, extra_naming = call
            ax.plot(x, func(*args, **kwargs)[0], '--', label=func.__name__ + extra_naming)

        ax.legend()
        ax.set_title(f'pybaselines.{module}')

    plt.show()
