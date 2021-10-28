===========
pybaselines
===========

.. image:: https://img.shields.io/pypi/v/pybaselines.svg
    :target: https://pypi.python.org/pypi/pybaselines
    :alt: Most Recent Version

.. image:: https://github.com/derb12/pybaselines/actions/workflows/python-test.yml/badge.svg
    :target: https://github.com/derb12/pybaselines/actions
    :alt: GitHub Actions test status

.. image:: https://readthedocs.org/projects/pybaselines/badge/?version=latest
    :target: https://pybaselines.readthedocs.io
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
    :target: https://github.com/derb12/pybaselines/tree/main/LICENSE.txt
    :alt: BSD 3-clause license

.. image:: https://img.shields.io/pypi/pyversions/pybaselines.svg
    :target: https://pypi.python.org/pypi/pybaselines
    :alt: Supported Python versions

.. image:: https://zenodo.org/badge/350510397.svg
    :target: https://zenodo.org/badge/latestdoi/350510397
    :alt: Zenodo DOI

pybaselines is a library of algorithms for the baseline correction of experimental data.

* For Python 3.6+
* Open Source: BSD 3-Clause License
* Source Code: https://github.com/derb12/pybaselines
* Documentation: https://pybaselines.readthedocs.io.


.. contents:: **Contents**
    :depth: 1


Introduction
------------

pybaselines provides many different baseline correction algorithms for fitting baselines
to data from experimental techniques such as Raman, FTIR, NMR, XRD, PIXE, etc. The aim of
the project is to provide a semi-unified API to allow quickly testing and comparing
multiple baseline correction algorithms to find the best one for a set of data.

pybaselines has 45+ baseline correction algorithms. Whenever possible, the original
names of the algorithms were used. The algorithms are grouped accordingly:

* Polynomial methods (pybaselines.polynomial)

  * poly (Regular Polynomial)
  * modpoly (Modified Polynomial)
  * imodpoly (Improved Modified Polynomial)
  * penalized_poly (Penalized Polynomial)
  * loess (Locally Estimated Scatterplot Smoothing)
  * quant_reg (Quantile Regression)
  * goldindec (Goldindec Method)

* Whittaker-smoothing-based methods (pybaselines.whittaker)

  * asls (Asymmetric Least Squares)
  * iasls (Improved Asymmetric Least Squares)
  * airpls (Adaptive Iteratively Reweighted Penalized Least Squares)
  * arpls (Asymmetrically Reweighted Penalized Least Squares)
  * drpls (Doubly Reweighted Penalized Least Squares)
  * iarpls (Improved Asymmetrically Reweighted Penalized Least Squares)
  * aspls (Adaptive Smoothness Penalized Least Squares)
  * psalsa (Peaked Signal's Asymmetric Least Squares Algorithm)
  * derpsalsa (Derivative Peak-Screening Asymmetric Least Squares Algorithm)

* Morphological methods (pybaselines.morphological)

  * mpls (Morphological Penalized Least Squares)
  * mor (Morphological)
  * imor (Improved Morphological)
  * mormol (Morphological and Mollified Baseline)
  * amormol (Averaging Morphological and Mollified Baseline)
  * rolling_ball (Rolling Ball Baseline)
  * mwmv (Moving Window Minimum Value)
  * tophat (Top-hat Transformation)
  * mpspline (Morphology-Based Penalized Spline)
  * jbcd (Joint Baseline Correction and Denoising)

* Smoothing-based methods (pybaselines.smooth)

  * noise_median (Noise Median method)
  * snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
  * swima (Small-Window Moving Average)

* Spline methods (pybaselines.spline)

  * mixture_model (Mixture Model)
  * irsqr (Iterative Reweighted Spline Quantile Regression)
  * corner_cutting (Corner-Cutting Method)
  * ipsa (Iterative Polynomial Smoothing Algorithm)
  * ria (Range Independent Algorithm)

* Baseline/Peak Classification methods (pybaselines.classification)

  * dietrich (Dietrich's Classification Method)
  * golotvin (Golotvin's Classification Method)
  * std_distribution (Standard Deviation Distribution)
  * fastchrom (FastChrom's Baseline Method)
  * cwt_br (Continuous Wavelet Transform Baseline Recognition)
  * fabc (Fully Automatic Baseline Correction)

* Optimizers (pybaselines.optimizers)

  * collab_pls (Collaborative Penalized Least Squares)
  * optimize_extended_range
  * adaptive_minmax (Adaptive MinMax)

* Miscellaneous methods (pybaselines.misc)

  * interp_pts (Interpolation between points)
  * beads (Baseline Estimation And Denoising with Sparsity)


Installation
------------

Stable Release
~~~~~~~~~~~~~~

pybaselines is easily installed from `pypi <https://pypi.org/project/pybaselines>`_
using `pip <https://pip.pypa.io>`_, by running the following command in the terminal:

.. code-block:: console

    pip install pybaselines

To also install the `optional dependencies`_ when installing pybaselines, run:

.. code-block:: console

    pip install pybaselines[full]


.. _optional dependencies: https://pybaselines.readthedocs.io/en/latest/installation.html#optional-dependencies

Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the `Github repo`_.

The public repository can be cloned using:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git


Once the repository is downloaded, it can be installed with:

.. code-block:: console

    cd pybaselines
    pip install .


.. _Github repo: https://github.com/derb12/pybaselines


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later
and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.14)
* `SciPy <https://www.scipy.org/scipylib/index.html>`_ (>= 1.0)


All of the required libraries should be automatically installed when
installing pybaselines using either of the two installation methods above.

The optional dependencies for pybaselines are listed in the
`documentation <https://pybaselines.readthedocs.io/en/latest/installation.html#optional-dependencies>`_.


Quick Start
-----------

To use the various functions in pybaselines, simply input the measured
data and any required parameters. All baseline correction functions in pybaselines
will output two items: a numpy array of the calculated baseline and a
dictionary of potentially useful parameters.

For more details on each baseline algorithm, refer to the `algorithms section`_ of
pybaselines's documentation. For examples of their usage, refer to the `examples section`_.

.. _algorithms section: https://pybaselines.readthedocs.io/en/latest/algorithms/index.html

.. _examples section: https://pybaselines.readthedocs.io/en/latest/examples/index.html

A simple example is shown below.

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import pybaselines
    from pybaselines import utils

    x = np.linspace(1, 1000, 1000)
    # a measured signal containing several Gaussian peaks
    signal = (
        utils.gaussian(x, 4, 120, 5)
        + utils.gaussian(x, 5, 220, 12)
        + utils.gaussian(x, 5, 350, 10)
        + utils.gaussian(x, 7, 400, 8)
        + utils.gaussian(x, 4, 550, 6)
        + utils.gaussian(x, 5, 680, 14)
        + utils.gaussian(x, 4, 750, 12)
        + utils.gaussian(x, 5, 880, 8)
    )
    # exponentially decaying baseline
    true_baseline = 2 + 10 * np.exp(-x / 400)
    noise = np.random.default_rng(1).normal(0, 0.2, x.size)

    y = signal + true_baseline + noise

    bkg_1 = pybaselines.polynomial.modpoly(y, x, poly_order=3)[0]
    bkg_2 = pybaselines.whittaker.asls(y, lam=1e7, p=0.02)[0]
    bkg_3 = pybaselines.morphological.mor(y, half_window=30)[0]
    bkg_4 = pybaselines.smooth.snip(
        y, max_half_window=40, decreasing=True, smooth_half_window=3
    )[0]

    plt.plot(x, y, label='raw data', lw=1.5)
    plt.plot(x, true_baseline, lw=3, label='true baseline')
    plt.plot(x, bkg_1, '--', label='modpoly')
    plt.plot(x, bkg_2, '--', label='asls')
    plt.plot(x, bkg_3, '--', label='mor')
    plt.plot(x, bkg_4, '--', label='snip')

    plt.legend()
    plt.show()


The above code will produce the image shown below.

.. image:: https://github.com/derb12/pybaselines/raw/main/docs/images/quickstart.jpg
   :align: center
   :alt: various baselines


Contributing
------------

Contributions are welcomed and greatly appreciated. For information on
submitting bug reports, pull requests, or general feedback, please refer
to the `contributing guide`_.

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
