===========
pybaselines
===========

.. image:: https://img.shields.io/pypi/v/pybaselines.svg
    :target: https://pypi.python.org/pypi/pybaselines
    :alt: Most Recent Version

.. image:: https://readthedocs.org/projects/pybaselines/badge/?version=latest
    :target: https://pybaselines.readthedocs.io
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/pyversions/pybaselines.svg
    :target: https://pypi.python.org/pypi/pybaselines
    :alt: Supported Python versions

.. image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
    :target: https://github.com/derb12/pybaselines/tree/main/LICENSE.txt
    :alt: BSD 3-clause license


pybaselines is a collection of baseline algorithms for fitting experimental data.

* For Python 3.6+
* Open Source: BSD 3-Clause License
* Source Code: https://github.com/derb12/pybaselines
* Documentation: https://pybaselines.readthedocs.io.


.. contents:: **Contents**
    :depth: 1


Introduction
------------

pybaselines provides different techniques for fitting baselines to experimental data.

Baseline fitting techniques are grouped accordingly (note: when a method
is labelled as 'improved', that is the method's name, not editorialization):

a) Polynomial (pybaselines.polynomial)

    1) poly (Regular Polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)
    4) penalized_poly (Penalized Polynomial)
    5) loess (Locally Estimated Scatterplot Smoothing)

b) Whittaker-smoothing-based techniques (pybaselines.whittaker)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)

c) Morphological (pybaselines.morphological)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) mormol (Morphological and Mollified Baseline)
    5) amormol (Averaging Morphological and Mollified Baseline)

d) Window-based (pybaselines.window)

    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)

e) Optimizers (pybaselines.optimizers)

    1) collab_pls (Collaborative Penalized Least Squares)
    2) optimize_extended_range

f) Manual methods (pybaselines.manual)

    1) linear_interp (Linear interpolation between points)


Installation
------------

Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.9)
* `SciPy <https://www.scipy.org/scipylib/index.html>`_


All of the required libraries should be automatically installed when installing pybaselines
using either of the two installation methods below.


Stable Release
~~~~~~~~~~~~~~

pybaselines is easily installed using `pip <https://pip.pypa.io>`_, simply by running
the following command in your terminal:

.. code-block:: console

    pip install --upgrade pybaselines

This is the preferred method to install pybaselines, as it will always install the
most recent stable release.


Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the `Github repo`_.

The public repository can be cloned using:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git


Once the repository is downloaded, it can be installed with:

.. code-block:: console

    cd pybaselines
    python setup.py install


.. _Github repo: https://github.com/derb12/pybaselines


Quick Start
-----------

To use the various functions in pybaselines, simply input the measured
data and any required parameters. All baseline functions in pybaselines
will output two items: the calculated baseline and a dictionary of parameters
that can be helpful for reusing the functions.

A simple example is shown below.

.. code-block:: python

    import pybaselines
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(100, 4200, 2000)
    # a measured signal containing several Gaussian peaks
    signal = (
        pybaselines.utils.gaussian(x, 2, 700, 50)
        + pybaselines.utils.gaussian(x, 3, 1200, 150)
        + pybaselines.utils.gaussian(x, 5, 1600, 100)
        + pybaselines.utils.gaussian(x, 4, 2500, 50)
        + pybaselines.utils.gaussian(x, 7, 3300, 100)
    )
    true_baseline = (
        10 + 0.001 * x  # polynomial baseline
        + pybaselines.utils.gaussian(x, 6, 2000, 2000)  # gaussian baseline
    )
    noise = np.random.default_rng(0).normal(0, 0.2, x.size)

    y = signal + true_baseline + noise

    bkg_1 = pybaselines.polynomial.modpoly(y, x, poly_order=3)[0]
    bkg_2 = pybaselines.whittaker.asls(y, lam=1e8, p=0.01)[0]
    bkg_3 = pybaselines.morphological.imor(y, half_window=50)[0]
    bkg_4 = pybaselines.window.snip(y, max_half_window=70, decreasing=True, smooth=True)[0]

    plt.plot(x, y, label='raw data')
    plt.plot(x, true_baseline, lw=3, label='true baseline')
    plt.plot(x, bkg_1, '--', label='modpoly')
    plt.plot(x, bkg_2, '--', label='asls')
    plt.plot(x, bkg_3, '--', label='imor')
    plt.plot(x, bkg_4, '--', label='snip')

    plt.legend()
    plt.show()


The above code will produce the image shown below.

.. image:: https://github.com/derb12/baselines/raw/main/docs/images/quickstart.jpg
   :align: center
   :alt: various baselines


Contributing
------------

Contributions are welcomed and greatly appreciated. For information on submitting bug reports,
pull requests, or general feedback, please refer to the `contributing guide`_.

.. _contributing guide: https://github.com/derb12/pybaselines/tree/main/docs/contributing.rst


Changelog
---------

Refer to the changelog_ for information on pybaselines's changes.

.. _changelog: https://github.com/derb12/pybaselines/tree/main/CHANGELOG.rst


License
-------

pybaselines is open source and freely available under the BSD 3-clause license.
For more information, refer to the license_.

.. _license: https://github.com/derb12/pybaselines/tree/main/LICENSE.txt


Author
------

* Donald Erb <donnie.erb@gmail.com>
