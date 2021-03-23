===========
pybaselines
===========

.. image:: https://github.com/derb12/baselines/raw/main/docs/images/logo.png
    :alt: pybaselines Logo
    :align: center

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

a) Whittaker-smoothing-based techniques (pybaselines.whittaker)

    1) asls (Asymmetric Least Squares)
    2) iasls (Improved Asymmetric Least Squares)
    3) airpls (Adaptive iteratively reweighted penalized least squares)
    4) arpls (Asymmetrically reweighted penalized least squares)
    5) drpls (Doubly reweighted penalized least squares)
    6) iarpls (Improved Asymmetrically reweighted penalized least squares)
    7) aspls (Adaptive smoothness penalized least squares)

b) Morphological (pybaselines.morphological)

    1) mpls (Morphological Penalized Least Squares)
    2) mor (Morphological)
    3) imor (Improved Morphological)
    4) iamor (Iterative averaging morphological)

c) Polynomial (pybaselines.polynomial)

    1) poly (regular polynomial)
    2) modpoly (Modified Polynomial)
    3) imodpoly (Improved Modified Polynomial)

d) Window-based (pybaselines.window)

    1) noise_median (Noise Median method)
    2) snip (Statistics-sensitive Non-linear Iterative Peak-clipping)


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


Usage
-----------

To be added...


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
