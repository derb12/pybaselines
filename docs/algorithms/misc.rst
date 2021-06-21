=======================
Miscellaneous Baselines
=======================

The contents of :mod:`pybaselines.misc` contain miscellaneous baseline algorithms
that do not fit in other categories.

Algorithms
----------

interp_pts (Interpolation between points)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`.interp_pts` interpolates between input points using line segments
or splines of different orders. The function is mainly intended for usage
with user interfaces and is not encouraged otherwise.

.. note::
   Unlike most other algorithms in pybaselines, interp_pts requires inputting
   the x-values of the data rather than the y-values.


.. plot::
   :align: center

    import numpy as np
    import matplotlib.pyplot as plt
    from pybaselines.utils import gaussian
    from pybaselines.misc import interp_pts

    x = np.linspace(1, 1000, 500)
    signal = (
        gaussian(x, 6, 180, 5)
        + gaussian(x, 8, 350, 10)
        + gaussian(x, 15, 400, 8)
        + gaussian(x, 13, 700, 12)
        + gaussian(x, 9, 800, 10)
    )
    baseline = 5 + 15 * np.exp(-x / 400)
    noise = np.random.default_rng(0).normal(0, 0.2, x.size)
    y = signal + baseline + noise

    # (x, y) pairs representing each point
    points = (
        (0, 20), (140, 15.5), (250, 12.8),
        (540, 8.75), (750, 7.23), (1001, 6.25)
    )

    linear_baseline = interp_pts(x, points)[0]
    spline_baseline = interp_pts(x, points, interp_method='cubic')[0]

    fig, ax = plt.subplots(tight_layout={'pad': 0.2})
    data_handle = ax.plot(x, y)
    baseline_handle = ax.plot(x, linear_baseline, '--')
    baseline_handle_2 = ax.plot(x, spline_baseline, ':')
    points_handle = ax.plot(*list(zip(*points)), 'ro')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.legend(
        (data_handle[0], baseline_handle[0], baseline_handle_2[0], points_handle[0]),
        ('data', 'linear interpolation', 'cubic spline interpolation', 'anchor points'),
        frameon=False
    )
    plt.show()


There is no figure showing the fits for various baseline types for this method
since it solely depends on the user-defined anchor points.
