===========
Quick Start
===========

To use the various functions in pybaselines, simply input the measured
data and any required parameters. All baseline functions in pybaselines
will output two items: a numpy array of the calculated baseline and a
dictionary of parameters that can be helpful for reusing the functions.

A simple example is shown below.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import pybaselines
    from pybaselines import utils

    x = np.linspace(100, 4200, 1000)
    # a measured signal containing several Gaussian peaks
    signal = (
        utils.gaussian(x, 2, 700, 50)
        + utils.gaussian(x, 3, 1200, 150)
        + utils.gaussian(x, 5, 1600, 100)
        + utils.gaussian(x, 4, 2500, 50)
        + utils.gaussian(x, 7, 3300, 100)
    )
    # baseline is a polynomial plus a broad gaussian
    true_baseline = (
        10 + 0.001 * x
        + utils.gaussian(x, 6, 2000, 2000)
    )
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)

    y = signal + true_baseline + noise

    bkg_1 = pybaselines.polynomial.modpoly(y, x, poly_order=3)[0]
    bkg_2 = pybaselines.whittaker.asls(y, lam=1e7, p=0.01)[0]
    bkg_3 = pybaselines.morphological.imor(y, half_window=25)[0]
    bkg_4 = pybaselines.window.snip(
        y, max_half_window=40, decreasing=True, smooth_half_window=1
    )[0]

    plt.plot(x, y, label='raw data', lw=1.5)
    plt.plot(x, true_baseline, lw=3, label='true baseline')
    plt.plot(x, bkg_1, '--', label='modpoly')
    plt.plot(x, bkg_2, '--', label='asls')
    plt.plot(x, bkg_3, '--', label='imor')
    plt.plot(x, bkg_4, '--', label='snip')

    plt.legend()
    plt.show()


The above code will produce the image shown below.

.. image:: images/quickstart.jpg
   :align: center
   :alt: various baselines
