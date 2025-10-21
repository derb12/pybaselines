"""
Interactive ``half_window`` Plot
--------------------------------

This example shows how to make a very simple interactive plot that allows changing ``half_window``
values for any morphological or smoothing algorithm. Note that in reality, several other
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

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from pybaselines import Baseline, utils


# When building documentation on readthedocs, the environment is headless and raises an
# error when trying to use tkinter, so just skip making the GUI when building documentation
# -> set environmental variable PB_BUILDING_DOCS to '1' when invoking sphinx-build
BUILDING_DOCS = os.environ.get('PB_BUILDING_DOCS', '0') == '1'


def get_methods():
    """Gets specific Baseline methods that use the ``half_window`` parameter."""
    methods = []
    for (method_name, method) in inspect.getmembers(Baseline):
        if (
            inspect.isfunction(method)
            and not method_name.startswith('_')
            # snip and swima used max_half_window instead of half_window until pybaselines version 1.3
            and (
                'half_window' in inspect.signature(method).parameters.keys()
                or method_name in ('snip', 'swima')
            )
        ):
            methods.append(method_name)

    return methods


class WindowFitter:
    """A basic GUI for varying ``half_window`` for various methods in pybaselines."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.baseline_fitter = Baseline(self.x, check_finite=False)
        self.line = None
        self.default_method = 'mor'

        # skip creating the full tkinter GUI if this program is ran during
        # the documentation build since it raises a tkinter error on readthedocs
        if BUILDING_DOCS:
            self.root = None
        else:
            self.root = tk.Tk()

        self.make_gui()

    def make_gui(self):
        """Constructs the GUI."""
        if BUILDING_DOCS:
            # use the pyplot interface so that the example output image will show a thumbnail
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = Figure()
            self.ax = self.fig.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.ax_variable = self.fig.add_axes([0.2, 0.05, 0.6, 0.1])
        self.ax.plot(self.x, self.y, label='data')
        # add an empty line as a placeholder
        self.line = self.ax.plot(self.x, np.full_like(self.x, np.nan), label='baseline')[0]
        self.ax.legend()

        if self.root is not None:
            self.canvas = FigureCanvasTkAgg(self.fig, self.root)
            self.canvas.draw()
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.root, pack_toolbar=False)
            self.toolbar.update()
        else:
            self.canvas = self.fig.canvas

        # add the interactive widgets
        self.slider = Slider(
            self.ax_variable, label='half_window', valmin=1, valmax=200, valinit=30, valstep=1
        )
        self.slider.on_changed(self.update_fit)
        if self.root is not None:
            self.selected_method = tk.StringVar(value=self.default_method)
            self.method_selector = ttk.Combobox(
                self.root, values=get_methods(), textvariable=self.selected_method, state='readonly'
            )
            self.method_selector.bind('<<ComboboxSelected>>', self.update_fit)

            # add everything into the gui
            self.method_selector.pack(side='bottom')
            self.toolbar.pack(side='top', fill='x')
            self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

        self.update_fit()

    @property
    def baseline_method(self):
        """The method to use for baseline correction."""
        if self.root is None:
            return self.default_method
        else:
            return self.selected_method.get()

    def update_fit(self, *args):
        """Updates the baseline fit with the selected method and value.

        Notes
        -----
        If adapting this for personal use, will have to take care to modify these
        default parameters for the data being fit.
        """
        method = self.baseline_method
        baseline_func = getattr(self.baseline_fitter, method)
        kwargs = {}
        method_parameters = inspect.signature(baseline_func).parameters.keys()
        if 'smooth_half_window' in method_parameters:
            # set smooth_half_window to 1 since it is otherwise typically set
            # to the same value as half_window and would shadow the half_window effects
            kwargs['smooth_half_window'] = 1
        if 'pad_kwargs' in method_parameters:
            # likewise, have to fix extrapolate_window for padding since it is otherwise typically
            # set to the same value as half_window and would shadow the half_window effects
            kwargs['pad_kwargs'] = {'extrapolate_window': 20}
        if 'lam' in method_parameters:
            # default lam values are a little too high for the sine baseline in this example
            kwargs['lam'] = 1e2

        # snip and swima used max_half_window instead of half_window until pybaselines version 1.3
        if method in ('snip', 'swima'):
            key = 'max_half_window'
        else:
            key = 'half_window'
        kwargs[key] = self.slider.val

        fit_baseline = baseline_func(y, **kwargs)[0]
        self.line.set_ydata(fit_baseline)
        self.canvas.draw_idle()


x = np.linspace(0, 1000, 1000)
signal = (
    utils.gaussian(x, 9, 100, 12)
    + utils.gaussian(x, 6, 180, 5)
    + utils.gaussian(x, 8, 350, 11)
    + utils.gaussian(x, 6, 550, 6)
    + utils.gaussian(x, 13, 700, 8)
    + utils.gaussian(x, 9, 800, 9)
    + utils.gaussian(x, 9, 880, 7)
)
baseline = 5 + 10 * np.exp(-x / 1200) + np.sin(x / 100)
noise = np.random.default_rng(0).normal(0, 0.1, len(x))
y = signal + baseline + noise

gui = WindowFitter(x, y)
# only invoke tkinter if not in a documentation build
if not BUILDING_DOCS:
    tk.mainloop()
