# -*- coding: utf-8 -*-
"""Creates the logo plot.

Creates the plot used for the logo; further edits are done in Inkscape to finalize the logo.

Created on November 15, 2021

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

    from pybaselines import Baseline, utils

    # assumes file is in pybaselines/tools
    image_directory = Path(__file__).parent
    with plt.rc_context(
        {'interactive': False, 'lines.linewidth': 2.5,
         'figure.dpi': 300}
    ):
        fig, ax = plt.subplots(tight_layout={'pad': 0.1}, frameon=False)

        x = np.linspace(1, 1000, 1000)
        signal = (
            utils.gaussian(x, 4, 120, 5)
            + utils.gaussian(x, 5, 220, 12)
            + utils.gaussian(x, 5, 350, 10)
            + utils.gaussian(x, 4, 550, 26)
            + utils.gaussian(x, 5, 680, 14)
            + utils.gaussian(x, 4, 750, 12)
            + utils.gaussian(x, 5, 880, 8)
        )
        true_baseline = 2 + 1e-3 * x + utils.gaussian(x, 1, 600, 300)
        noise = np.random.default_rng(1).normal(0, 0.05, x.size)

        y = signal + true_baseline + noise
        baseline = Baseline().arpls(y, lam=1e7)[0]

        blue = '#0952ff'
        pink = '#ff5255'
        ax.plot(x, y, lw=1.5, color=blue)
        ax.plot(x, baseline, lw=4, color=pink)

        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # save as an svg so that it can be edited/scaled in inkskape without
        # losing image quality
        fig.savefig(image_directory.joinpath('logo_new.svg'), transparent=True)
        plt.close(fig)
