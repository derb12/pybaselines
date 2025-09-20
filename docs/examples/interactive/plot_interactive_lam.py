"""
Interactive ``lam`` Plot
-------------------------

This example shows how to make a very simple interactive plot that allows changing ``lam``
values for any Whittaker-smoothing or penalized spline algorithm. Note that in reality, several
other parameters will likely need to be set for adequate fits.

In order to use this program, download the source Python file and then run it. Feel free
to replace `x` and `y` in this example with your actual data.

For a more comprehensive and well-built GUI interface for pybaselines, see
`HyperSpy <https://hyperspy.org>`_.

"""

import inspect
import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pybaselines import Baseline, utils


def get_methods():
    """Gets specific Baseline methods that use the ``lam`` parameter."""
    methods = []
    for (method_name, method) in inspect.getmembers(Baseline):
        if (
            inspect.isfunction(method)
            and not method_name.startswith('_')
            and 'lam' in inspect.signature(method).parameters.keys()
            # custom_bc only uses lam to smooth connecting regions, so it's not useful for this example
            and method_name != 'custom_bc'
        ):
            methods.append(method_name)

    return methods


class LamFitter:
    """A basic GUI for varying ``lam`` for various methods in pybaselines."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.baseline_fitter = Baseline(self.x, check_finite=False)
        self.baseline_fitter.banded_solver = 3
        self.root = tk.Tk()
        self.line = None
        self.wt_line = None

        self.make_gui()

    def make_gui(self):
        """Constructs the GUI."""
        self.fig, (self.ax, self.wt_ax) = plt.subplots(
            nrows=2, sharex=True, height_ratios=(2, 1), gridspec_kw={'hspace': 0.1}
        )
        self.fig.subplots_adjust(bottom=0.2)
        self.ax_variable = self.fig.add_axes([0.2, 0.05, 0.6, 0.1])
        self.ax.plot(self.x, self.y, label='data')
        # add an empty line as a placeholder
        self.line = self.ax.plot(self.x, np.full_like(self.x, np.nan), label='baseline')[0]
        self.ax.legend()

        self.wt_ax.set_ylabel('Weights')

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()

        # add the interactive widgets
        self.slider = Slider(
            self.ax_variable, label='log$_{10}$(lam)', valmin=-4, valmax=12, valinit=6
        )
        self.selected_method = tk.StringVar(value='arpls')
        self.method_selector = ttk.Combobox(
            self.root, values=get_methods(), textvariable=self.selected_method, state='readonly'
        )

        # bind events to all of the widgets
        self.slider.on_changed(self.update_fit)
        self.method_selector.bind('<<ComboboxSelected>>', self.update_fit)

        # add everything into the gui
        self.method_selector.pack(side='bottom')
        self.toolbar.pack(side='top', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.update_fit()

    def update_fit(self, *args):
        """Updates the baseline fit with the selected method and value.

        Notes
        -----
        If adapting this for personal use, will have to take care to modify these
        default parameters for the data being fit.
        """
        method = self.selected_method.get()
        baseline_func = getattr(self.baseline_fitter, method)
        lam = 10.0**self.slider.val
        kwargs = {}
        if method == 'rubberband':  # split into 4 sections for rubberband to find good anchor points
            kwargs['segments'] = 4
        elif method == 'fabc':  # set the scale so peak detection works well for this example
            kwargs['scale'] = 20
        elif method in ('mpls', 'mpspline'):
            # set half-window so that peak-finding works for this example
            kwargs['half_window'] = 75

        fit_baseline, params = baseline_func(y, lam=lam, **kwargs)
        if method in ('mpls', 'mpspline'):
            # mpls and mpspline can miss edge regions, so refit with explicit weights on the ends
            kwargs['weights'] = params['weights']
            kwargs['weights'][0] = 1
            kwargs['weights'][-1] = 1
            fit_baseline, params = baseline_func(y, lam=lam, **kwargs)

        self.line.set_ydata(fit_baseline)

        if 'weights' in params:
            wt_vals = params['weights']
        elif 'mask' in params:
            wt_vals = params['mask'].astype(float)
        else:
            wt_vals = np.full_like(self.x, np.nan)
        if self.wt_line is None:
            self.wt_line = self.wt_ax.plot(self.x, wt_vals, 'r.')[0]
        else:
            self.wt_line.set_ydata(wt_vals)
            # some methods have weights > 1, so have to rescale the y-axis
            if wt_vals.max() > 1:
                self.wt_ax.relim()
                self.wt_ax.autoscale_view(scalex=False)

        self.canvas.draw_idle()


x = np.linspace(0, 1000, 1000)
signal = (
    utils.gaussian(x, 9, 100, 12)
    + utils.gaussian(x, 6, 180, 5)
    + utils.gaussian(x, 8, 350, 11)
    + utils.gaussian(x, 15, 400, 18)
    + utils.gaussian(x, 6, 550, 6)
    + utils.gaussian(x, 13, 700, 8)
    + utils.gaussian(x, 9, 800, 9)
    + utils.gaussian(x, 9, 880, 7)
)
baseline = 5 + 10 * np.exp(-x / 800) + np.sin(x / 100)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

gui = LamFitter(x, y)
# have to ensure GUI is automatically closed if building documentation, so the
# environmental variable PB_BUILDING_DOCS is set to '1' when invoking sphinx-build
if os.getenv('PB_BUILDING_DOCS', '0') == '1':
    gui.root.destroy()
else:
    tk.mainloop()
