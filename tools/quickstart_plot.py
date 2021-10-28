# -*- coding: utf-8 -*-
"""Creates the quickstart plot for documentation.

Manually creates the quickstart plot for documentation for use in the README.

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

    import pybaselines
    from pybaselines import utils

    # assumes file is in pybaselines/tools
    image_directory = Path(__file__).parent.parent.joinpath('docs/images')
    with plt.rc_context(
        {'interactive': False, 'lines.linewidth': 2.5,
         'figure.figsize': (4.5, 4), 'figure.dpi': 100}
    ):
        fig, ax = plt.subplots(tight_layout={'pad': 0.15})

        x = np.linspace(1, 1000, 1000)
        # a measured signal containing several Gaussian peaks
        signal = (
            utils.gaussian(x, 4, 120, 5)
            + utils.gaussian(x, 5, 220, 12)
            + utils.gaussian(x, 5, 350, 10)
            + utils.gaussian(x, 7, 400, 8)
            + utils.gaussian(x, 4, 550, 6)
            + utils.gaussian(x, 5, 680, 14)
            + utils.gaussian(x, 4, 750, 12)
            + utils.gaussian(x, 5, 880, 8)
        )
        # exponentially decaying baseline
        true_baseline = 2 + 10 * np.exp(-x / 400)
        noise = np.random.default_rng(1).normal(0, 0.2, x.size)

        y = signal + true_baseline + noise

        bkg_1 = pybaselines.polynomial.modpoly(y, x, poly_order=3)[0]
        bkg_2 = pybaselines.whittaker.asls(y, lam=1e7, p=0.02)[0]
        bkg_3 = pybaselines.morphological.mor(y, half_window=30)[0]
        try:
            bkg_4 = pybaselines.smooth.snip(
                y, max_half_window=40, decreasing=True, smooth_half_window=3
            )[0]
        except AttributeError:
            # pybaselines.window was renamed to pybaselines.smooth in version 0.6
            bkg_4 = pybaselines.window.snip(
                y, max_half_window=40, decreasing=True, smooth_half_window=3
            )[0]

        plt.plot(x, y, label='raw data', lw=1.5)
        plt.plot(x, true_baseline, lw=3, label='true baseline')
        plt.plot(x, bkg_1, '--', label='modpoly')
        plt.plot(x, bkg_2, '--', label='asls')
        plt.plot(x, bkg_3, '--', label='mor')
        plt.plot(x, bkg_4, '--', label='snip')

        plt.legend(frameon=False)
        fig.savefig(
            image_directory.joinpath('quickstart.jpg'),
            pil_kwargs={'quality': 90, 'optimize': True, 'progressive': True}
        )
        plt.close(fig)
