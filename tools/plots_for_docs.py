# -*- coding: utf-8 -*-
"""Creates plots of the various baseline algorithms in pybaselines for documentation.

Manually create the plots for each of the modules in pybaselines rather than
using matplotlib's plot directive so that matplotlib does not need to be a
requirement for documentation.

Created on March 28, 2021

@author: Donald Erb

"""


if __name__ == '__main__':

    from pathlib import Path

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('This file requires matplotlib to run')
        raise
    import numpy as np

    from pybaselines.morphological import (amormol, imor, mor, mormol, mpls,
                                           rolling_ball)
    from pybaselines.optimizers import adaptive_minmax, optimize_extended_range
    from pybaselines.polynomial import (imodpoly, loess, modpoly,
                                        penalized_poly, poly)
    from pybaselines.utils import gaussian
    from pybaselines.whittaker import (airpls, arpls, asls, aspls, drpls,
                                       iarpls, iasls, psalsa)
    from pybaselines.window import noise_median, snip, swima

    x = np.linspace(100, 4200, 1000)
    signal = (
        gaussian(x, 2, 700, 50)
        + gaussian(x, 3, 1200, 150)
        + gaussian(x, 5, 1600, 100)
        + gaussian(x, 4, 2500, 50)
        + gaussian(x, 7, 3300, 100)
        + np.random.default_rng(1).normal(0, 0.2, x.size)  # noise
    )
    true_baseline = (
        10 + 0.001 * x  # polynomial baseline
        + gaussian(x, 6, 2000, 2000)  # gaussian baseline
    )

    y = signal + true_baseline
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
            (psalsa, (y, 1e7))
        ),
        'polynomial': (
            (poly, (y, x, 3)),
            (poly, (y, x, 3), {'weights': non_peak_mask}, ', fit only non-peaks'),
            (modpoly, (y, x, 3)),
            (imodpoly, (y, x, 3)),
            (penalized_poly, (y, x, 3), {'threshold': 0.02 * (max(y) - min(y))}),
            (loess, (y, x, 0.6))
        ),
        'morphological': (
            (mpls, (y, 100, 1e7, 0.002)),
            (mor, (y, 100)),
            (imor, (y, 25)),
            (mormol, (y, 100), {'pad_kwargs': {'extrapolate_window': 50}, 'smooth_half_window': 3}),
            (amormol, (y, 45), {'pad_kwargs': {'extrapolate_window': 50}}),
            (rolling_ball, (y, 125, 100), {'pad_kwargs': {'extrapolate_window': 50}})
        ),
        'window': (
            (noise_median, (y, 250, 150, 50), {'extrapolate_window': 50}),
            (snip, (y, 40), {'extrapolate_window': 50}),
            (snip, (y, 40, True, 1), {'extrapolate_window': 50}, ', decreasing & smooth'),
            (swima, (y,), {'extrapolate_window': 50})
        ),
        'optimizers': (
            (optimize_extended_range, (y, x, 'aspls', 'both')),
            (adaptive_minmax, (y, x)),
        )
    }

    # assumes file is in pybaselines/tools
    image_directory = Path(__file__).parent.parent.joinpath('docs/images')
    with plt.rc_context(
        {'interactive': False, 'lines.linewidth': 2.5, 'legend.frameon': False,
         'figure.figsize': (4.5, 4), 'figure.dpi': 100}
    ):
        for module, func_calls in algorithms.items():
            fig, ax = plt.subplots(num='plot', tight_layout={'pad': 0.15})
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
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'pybaselines.{module}')
            fig.savefig(
                image_directory.joinpath(f'{module}.jpg'),
                pil_kwargs={'quality': 90, 'optimize': True, 'progressive': True}
            )
            plt.close(fig)

        fig, ax = plt.subplots(num='plot', tight_layout={'pad': 0.15})

        bkg_1 = modpoly(y, x, poly_order=3)[0]
        bkg_2 = asls(y, lam=1e7, p=0.01)[0]
        bkg_3 = imor(y, half_window=25)[0]
        bkg_4 = snip(y, max_half_window=40, decreasing=True, smooth_half_window=1)[0]

        plt.plot(x, y, label='raw data', lw=1.5)
        plt.plot(x, true_baseline, lw=3, label='true baseline')
        plt.plot(x, bkg_1, '--', label='modpoly')
        plt.plot(x, bkg_2, '--', label='asls')
        plt.plot(x, bkg_3, '--', label='imor')
        plt.plot(x, bkg_4, '--', label='snip')

        plt.legend()
        fig.savefig(
            image_directory.joinpath('quickstart.jpg'),
            pil_kwargs={'quality': 90, 'optimize': True, 'progressive': True}
        )
        plt.close(fig)
