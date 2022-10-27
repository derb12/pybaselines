===========
pybaselines
===========

.. image:: https://github.com/derb12/pybaselines/raw/main/docs/images/logo.png
    :alt: Logo
    :align: center

.. image:: https://img.shields.io/pypi/v/pybaselines.svg
    :target: https://pypi.python.org/pypi/pybaselines
    :alt: Current Pypi Version

.. image:: https://img.shields.io/conda/vn/conda-forge/pybaselines.svg
    :target: https://anaconda.org/conda-forge/pybaselines
    :alt: Current conda Version

.. image:: https://github.com/derb12/pybaselines/actions/workflows/python-test.yml/badge.svg
    :target: https://github.com/derb12/pybaselines/actions
    :alt: GitHub Actions test status

.. image:: https://readthedocs.org/projects/pybaselines/badge/?version=latest
    :target: https://pybaselines.readthedocs.io
    :alt: Documentation Status

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

pybaselines is a Python library that provides many different algorithms for
performing baseline correction on data from experimental techniques such as
Raman, FTIR, NMR, XRD, XRF, PIXE, etc. The aim of the project is to provide a
semi-unified API to allow quickly testing and comparing multiple baseline
correction algorithms to find the best one for a set of data.

pybaselines has 50+ baseline correction algorithms. These include popular algorithms,
such as AsLS, airPLS, ModPoly, and SNIP, as well as many lesser known algorithms. Most
algorithms are adapted directly from literature, although there are a few that are unique
to pybaselines, such as penalized spline versions of Whittaker-smoothing-based algorithms.
The full list of implemented algorithms can be found in the
`documentation <https://pybaselines.readthedocs.io/en/latest/introduction.html>`_.


Installation
------------

Stable Release
~~~~~~~~~~~~~~

pybaselines can be installed from `pypi <https://pypi.org/project/pybaselines>`_
using `pip <https://pip.pypa.io>`_, by running the following command in the terminal:

.. code-block:: console

    pip install pybaselines

pybaselines can alternatively be installed from the
`conda-forge <https://anaconda.org/conda-forge/pybaselines>`_ channel using conda by running:

.. code-block:: console

    conda install -c conda-forge pybaselines


Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the `GitHub repo`_.
To install the current version of pybaselines from GitHub, run:

.. code-block:: console

    pip install git+https://github.com/derb12/pybaselines.git#egg=pybaselines


.. _GitHub repo: https://github.com/derb12/pybaselines


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later
and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.14)
* `SciPy <https://www.scipy.org>`_ (>= 1.0)


All of the required libraries should be automatically installed when
installing pybaselines using any of the installation methods above.

The `optional dependencies <https://pybaselines.readthedocs.io/en/latest/installation.html#optional-dependencies>`_
for pybaselines are listed in the documentation . To also install the optional
dependencies when installing pybaselines with pip, run:

.. code-block:: console

    pip install pybaselines[full]

If installing with conda, the optional dependencies have to be specified manually.

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

A simple example is shown below (if using a version earlier than 1.0, see the `quickstart`_ in
the documentation for the old version).

.. _quickstart: https://pybaselines.readthedocs.io/en/latest/quickstart.html#pre-version-1-0-0-quick-start


.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from pybaselines import Baseline, utils

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

    baseline_fitter = Baseline(x_data=x)

    bkg_1 = baseline_fitter.modpoly(y, poly_order=3)[0]
    bkg_2 = baseline_fitter.asls(y, lam=1e7, p=0.02)[0]
    bkg_3 = baseline_fitter.mor(y, half_window=30)[0]
    bkg_4 = baseline_fitter.snip(
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


Citing
------

If you use pybaselines for published research, please consider citing
by following the `guidelines in the documentation
<https://pybaselines.readthedocs.io/en/latest/citing.html>`_.


Author
------

* Donald Erb <donnie.erb@gmail.com>
