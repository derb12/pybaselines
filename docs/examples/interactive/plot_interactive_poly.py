"""
Interactive ``poly_order`` Plot
--------------------------------

This example shows how to make a very simple interactive plot that allows changing ``poly_order``
values for any polynomial algorithm. Note that in reality, several other
parameters will likely need to be set for adequate fits.

In order to use this program, download the source Python file and then run it. Feel free
to replace `x` and `y` in this example with your actual data.

For a more comprehensive and well-built GUI interface for pybaselines, see
`HyperSpy <https://hyperspy.org>`_.

"""

import inspect
import os
import tkinter as tk
from tkinter import ttk

# use Agg backend so that tkinter does not raise an error when building on readthedocs
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pybaselines import Baseline, utils


def get_methods():
    """Gets specific Baseline methods that use the ``poly_order`` parameter."""
    methods = []
    for (method_name, method) in inspect.getmembers(Baseline):
        if (
            inspect.isfunction(method)
            and not method_name.startswith('_')
            and 'poly_order' in inspect.signature(method).parameters.keys()
        ):
            methods.append(method_name)

    return methods


class PolyFitter:
    """A basic GUI for varying ``poly_order`` for various methods in pybaselines."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.baseline_fitter = Baseline(self.x, check_finite=False)
        self.root = tk.Tk()
        self.line = None

        self.make_gui()

    def make_gui(self):
        """Constructs the GUI."""
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.ax_variable = self.fig.add_axes([0.2, 0.05, 0.6, 0.1])
        self.ax.plot(self.x, self.y, label='data')
        # add an empty line as a placeholder
        self.line = self.ax.plot(self.x, np.full_like(self.x, np.nan), label='baseline')[0]
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
        self.toolbar.update()

        # add the interactive widgets
        self.slider = Slider(
            self.ax_variable, label='poly_order', valmin=1, valmax=12, valinit=4, valstep=1
        )
        self.selected_method = tk.StringVar(value='imodpoly')
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
        kwargs = {}
        if method == 'loess':
            kwargs['fraction'] = 0.3

        fit_baseline = baseline_func(y, poly_order=self.slider.val, **kwargs)[0]
        self.line.set_ydata(fit_baseline)
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
baseline = 5 + 10 * np.exp(-x / 400) + utils.gaussian(x, 10, 850, 500)
noise = np.random.default_rng(0).normal(0, 0.05, len(x))
y = signal + baseline + noise

gui = PolyFitter(x, y)
# have to ensure GUI is automatically closed if building documentation, so the
# environmental variable PB_BUILDING_DOCS is set to '1' when invoking sphinx-build
if os.getenv('PB_BUILDING_DOCS', '0') == '1':
    gui.root.destroy()
else:
    tk.mainloop()
