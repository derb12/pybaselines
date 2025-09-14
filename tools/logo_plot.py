# -*- coding: utf-8 -*-
"""Creates the logo for pybaselines.

Created on November 15, 2021

@author: Donald Erb

"""


if __name__ == '__main__':

    from pathlib import Path

    try:
        import matplotlib.pyplot as plt
        from matplotlib import patheffects
    except ImportError:
        print('This file requires matplotlib to run')
        raise
    import numpy as np

    from pybaselines import Baseline, utils

    # assumes file is in pybaselines/tools
    image_directory = Path(__file__).parent
    figure_dpi = 300
    with plt.rc_context({
        'font.family': 'sans-serif',
        'font.sans-serif': 'arial',
    }):
        fig, ax = plt.subplots(
            tight_layout={'pad': 0.1}, frameon=False, figsize=(1710 / figure_dpi, 473 / figure_dpi),
            dpi=figure_dpi
        )

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
        noise = np.random.default_rng(1).normal(0, 0.01, x.size)

        y = signal + true_baseline + noise
        baseline = Baseline().arpls(y, lam=1e7)[0]

        # for reference, see the matplotlib examples
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/rainbow_text.html
        # and https://matplotlib.org/stable/gallery/misc/patheffect_demo.html for how to make
        # multicolored aligned text with borders
        blue = '#137bff'
        pink = '#ff5255'
        text_size = 52
        text_border = [patheffects.withStroke(linewidth=1.5, foreground='black')]

        ax.plot(x, y, lw=2.5, color=blue)
        ax.plot(x, baseline, lw=3, color=pink)

        x_lims = ax.get_xlim()
        ax.set_xlim(x_lims[0] - 200, x_lims[1] + 200)
        ax.set_ylim(ax.get_ylim()[0] - 5)
        text = ax.text(1, -1.8, 'py', color=blue, size=text_size, path_effects=text_border)
        text = ax.annotate(
            'baselines', xycoords=text, xy=(1, 0), verticalalignment='bottom', size=text_size,
            color=pink, path_effects=text_border
        )

        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # No idea what's happening here, but for the first save, the scaling of the plot gets
        # shrunk legthwise such that it does not look like the displayed plot, but after saving
        # again it looks correct... just overwrite the first and ignore whatever is causing this
        fig.savefig(image_directory.joinpath('logo_new.png'), transparent=True)
        fig.savefig(image_directory.joinpath('logo_new.png'), transparent=True)

        plt.show()
