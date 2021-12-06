.. highlight:: shell

============
Installation
============


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.14)
* `SciPy <https://www.scipy.org/scipylib/index.html>`_ (>= 1.0)


All of the required libraries should be automatically installed when
installing pybaselines using any of the installation methods below.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

pybaselines has the following optional dependencies:

* `numba <https://github.com/numba/numba>`_ (>= 0.45):
  speeds up calculations used by the following functions:

    * :func:`.loess`
    * :func:`.dietrich`
    * :func:`.golotvin`
    * :func:`.std_distribution`
    * :func:`.fastchrom`
    * :func:`.beads`
    * :func:`.mpspline`
    * all functions in :mod:`pybaselines.spline`

* `pentapy <https://github.com/GeoStat-Framework/pentapy>`_ (>= 1.0):
  provides a faster solver for banded pentadiagonal linear systems, which are
  used by the following functions (when ``diff_order=2``):

    * all functions in :mod:`pybaselines.whittaker`
    * :func:`.mpls`
    * :func:`.jbcd`
    * :func:`.fabc`


Stable Release
~~~~~~~~~~~~~~

pybaselines can be installed from `pypi <https://pypi.org/project/pybaselines>`_
using `pip <https://pip.pypa.io>`_, by running the following command in the terminal:

.. code-block:: console

    pip install pybaselines

To also install the optional dependencies when installing pybaselines with pip, run:

.. code-block:: console

    pip install pybaselines[full]

pybaselines can alternatively be installed from the
`conda-forge <https://anaconda.org/conda-forge/pybaselines>`_ channel using conda by running:

.. code-block:: console

    conda install -c conda-forge pybaselines

If installing with conda, the optional dependencies have to be specified manually.


Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the
`GitHub repo <https://github.com/derb12/pybaselines>`_.

To directly install the current version of pybaselines from GitHub, run:

.. code-block:: console

    pip install git+https://github.com/derb12/pybaselines.git#egg=pybaselines

Alternatively, to download the entire public repository and install pybaselines, run:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git
    cd pybaselines
    pip install .
