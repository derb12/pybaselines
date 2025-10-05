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

The two independant variables for 2D baseline algorithms in pybaselines are defined as ``x_data``
along the rows and ``z_data`` along the columns. The figure below shows an illustration of this,
as well as the corresonding indexing within the input ``data``. In hindsight, it would have likely
made more sense to switch the meanings of the two variables (pobody's nerfect ¯\\_(ツ)_/¯), but
they will not be changed in order to maintain backwards compatibility. However, the documentation
and API reference for all 2D methods are clear in the row and column distinction.

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: False

   import matplotlib.pyplot as plt
   import numpy as np

   _, ax = plt.subplots(tight_layout=True)
   plt.grid()
   ax.set_aspect('equal')
   plt.annotate(
      '',
      xy=(-0.05, 0.2), xytext=(-0.05, 1), arrowprops={'arrowstyle': '->'}, xycoords='axes fraction',
   )
   plt.annotate(
      '',
      xy=(0.8, 1.05), xytext=(0, 1.05), arrowprops={'arrowstyle': '->'}, xycoords='axes fraction',
   )
   plt.annotate('x_data (along rows)', xy=(-0.1, 0.65), xycoords='axes fraction', rotation=90)
   plt.annotate('z_data (along columns)', xy=(0.05, 1.07), xycoords='axes fraction')

   grid_centers = np.linspace(0.1, 0.9, 5)
   print(grid_centers)
   for i, x in enumerate(grid_centers):
      for j, y in enumerate(grid_centers):
         plt.annotate(
               f'data[{4 - j}, {i}]', xy=(x, y), horizontalalignment='center',
               verticalalignment='center'
         )


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
