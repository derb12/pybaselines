.. highlight:: shell

============
Installation
============


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.14)
* `SciPy <https://www.scipy.org/scipylib/index.html>`_ (>= 0.17)


All of the required libraries should be automatically installed when
installing pybaselines using either of the two installation methods below.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

pybaselines has the following optional dependencies:

* `pentapy <https://github.com/GeoStat-Framework/pentapy>`_ (>= 1.0):
  provides a faster banded solver for Whittaker-smoothing-based algorithms
  (all functions in :mod:`pybaselines.whittaker` as well as
  :func:`pybaselines.morphological.mpls`)
* `numba <https://github.com/numba/numba>`_ (>= 0.45):
  speeds up calculations used by the following functions:

    * :func:`.loess`
    * :func:`.dietrich`
    * :func:`.golotvin`
    * :func:`.std_distribution`
    * :func:`.fastchrom`
    * :func:`.beads`
    * :func:`.mixture_model`
    * :func:`.corner_cutting`

Stable Release
~~~~~~~~~~~~~~

pybaselines is easily installed from `pypi <https://pypi.org/project/pybaselines>`_
using `pip <https://pip.pypa.io>`_, by running the following command in the terminal:

.. code-block:: console

    pip install --upgrade pybaselines

This is the preferred method to install pybaselines, as it will always install the most
recent stable release. Note that the ``--upgrade`` tag is used to ensure that the
most recent version of pybaselines is downloaded and installed, even if an older version
is currently installed.

To also install the optional dependencies when installing pybaselines, run:

.. code-block:: console

    pip install --upgrade pybaselines[full]


Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the `Github repo <https://github.com/derb12/pybaselines>`_.

The public repository can be cloned using:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git


Once the repository is downloaded, it can be installed with:

.. code-block:: console

    cd pybaselines
    pip install .
