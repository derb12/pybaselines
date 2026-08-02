=============
2D Algorithms
=============

pybaselines extends a subset of the one dimensional (1D) baseline correction algorithms to work
with two dimensional (2D) data. Note that this is only intended for data in which there is some
global baseline; otherwise, it is more appropriate and usually significantly faster to simply
use the 1D algorithms on each individual row and/or column in the data, which can be done using
:meth:`.Baseline2D.individual_axes` or using :class:`.Baseline` with for-loops.

This section of the documentation is to help provide some context for how the algorithms
were extended to work with 2D data. It will not be as comprehensive as the
:doc:`1D Algorithms section <../algorithms_1d/index>`, so to help understand any algorithm,
it is suggested to start there.

The :class:`.Baseline2D` class in pybaselines assumes that the data to be fit is defined on a
`rectilinear grid <https://wikipedia.org/wiki/Regular_grid>`_, with the two independent variables
defined as ``x_data`` along the rows and ``z_data`` along the columns. Thus, if the input ``data``
has shape (M, N), then ``len(x_data)`` is M and ``len(z_data)`` is N. The figure below shows an
illustration of this, as well as the corresponding indexing within the input data. In hindsight,
the naming conventions are slightly ambiguous, but they will not be changed in order to maintain
backwards compatibility. However, the documentation and API reference for all 2D methods are clear
in the row and column distinction.

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: False

   import matplotlib.pyplot as plt
   import numpy as np

   _, ax = plt.subplots(tight_layout=True)
   ax.set_aspect('equal')  # could make it asymmetric, but it gets too cramped
   plt.annotate(
      '',
      xy=(-0.05, 0), xytext=(-0.05, 1), arrowprops={'arrowstyle': '->'}, xycoords='axes fraction',
   )
   plt.annotate(
      '',
      xy=(1, 1.05), xytext=(0, 1.05), arrowprops={'arrowstyle': '->'}, xycoords='axes fraction',
   )
   plt.annotate('x_data (along rows)', xy=(-0.1, 0.65), xycoords='axes fraction', rotation=90)
   plt.annotate('z_data (along columns)', xy=(0.05, 1.07), xycoords='axes fraction')

   row_grid_centers = np.linspace(0.08, 0.92, 6)
   col_grid_centers = np.linspace(0.1, 0.9, 5)
   for i, x in enumerate(col_grid_centers):
      for j, z in enumerate(row_grid_centers):
         plt.annotate(
               f'data[{row_grid_centers.size - j - 1},{i}]', xy=(x, z), horizontalalignment='center',
               verticalalignment='center'
         )

   ax.set_yticks(np.linspace(0, 1, row_grid_centers.size + 1))
   plt.grid()

   for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()):
      tick.tick1line.set_visible(False)
      tick.tick2line.set_visible(False)
      tick.label1.set_visible(False)
      tick.label2.set_visible(False)
   plt.show()


.. toctree::
   :maxdepth: 2

   polynomial_2d
   whittaker_2d
   morphological_2d
   spline_2d
   smooth_2d
   optimizers_2d
