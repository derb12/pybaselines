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
    from pybaselines.morphological import amormol, imor, mor, mormol, mpls, rolling_ball
    from pybaselines.polynomial import (imodpoly, loess, modpoly,
                                        penalized_poly, poly)
    from pybaselines.utils import gaussian
    from pybaselines.whittaker import (airpls, arpls, asls, aspls, drpls,
                                       iarpls, iasls, psalsa)
    from pybaselines.window import noise_median, snip

    x = np.linspace(100, 4200, 2000)
    signal = (
        gaussian(x, 2, 700, 50)
        + gaussian(x, 3, 1200, 150)
        + gaussian(x, 5, 1600, 100)
        + gaussian(x, 4, 2500, 50)
        + gaussian(x, 7, 3300, 100)
        + np.random.default_rng(0).normal(0, 0.2, x.size)  # noise
    )
    true_baseline = (
        10 + 0.001 * x  # polynomial baseline
        + gaussian(x, 6, 2000, 2000)  # gaussian baseline
    )

    y = signal + true_baseline
    non_peak_mask = (x < 600) | ((x > 1900) & (x < 2500)) | ((x > 2600) & (x < 3100)) | (x > 3600)

    algorithms = {
        'whittaker': (
            (asls, (y, 1e8, 0.005)),
            (iasls, (y, x, 1e7, 0.04, 1e-3)),
            (airpls, (y, 1e8)),
            (drpls, (y, 1e9)),
            (arpls, (y, 1e8)),
            (iarpls, (y, 1e7)),
            (aspls, (y, 1e9)),
            (psalsa, (y, 1e8))
        ),
        'polynomial': (
            (poly, (y, x, 3)),
            (poly, (y, x, 3), {'weights': non_peak_mask}, ', fit only non-peaks'),
            (modpoly, (y, x, 3)),
            (imodpoly, (y, x, 3), {'use_original': True}),
            (penalized_poly, (y, x, 3), {'threshold': 0.02 * (max(y) - min(y))}),
            (loess, (y, x, 0.6))
        ),
        'morphological': (
            (mpls, (y, 200, 1e8, 0.002)),
            (mor, (y, 200)),
            (imor, (y, 50)),
            (mormol, (y, 200), {'pad_kwargs': {'extrapolate_window': 100}, 'smooth_half_window': 7}),
            (amormol, (y, 70), {'pad_kwargs': {'extrapolate_window': 100}}),
            (rolling_ball, (y, 250, 200), {'pad_kwargs': {'extrapolate_window': 100}})
        ),
        'window': (
            (noise_median, (y, 800, 200), {'extrapolate_window': 100}),
            (snip, (y, 70)),
            (snip, (y, 70, True, True), {}, ', decreasing & smooth = True')
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
