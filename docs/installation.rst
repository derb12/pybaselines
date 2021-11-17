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

* `pentapy <https://github.com/GeoStat-Framework/pentapy>`_ (>= 1.0):
  provides a faster solver for banded pentadiagonal linear systems
  (all functions in :mod:`pybaselines.whittaker` as well as
  :func:`.mpls`, :func:`.jcbd`, and :func:`.fabc`)
* `numba <https://github.com/numba/numba>`_ (>= 0.45):
  speeds up calculations used by the following functions:

    * :func:`.loess`
    * :func:`.dietrich`
    * :func:`.golotvin`
    * :func:`.std_distribution`
    * :func:`.fastchrom`
    * :func:`.beads`
    * :func:`.mixture_model`
    * :func:`.irsqr`
    * :func:`.corner_cutting`

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

The sources for pybaselines can be downloaded from the `GitHub repo <https://github.com/derb12/pybaselines>`_.

The public repository can be cloned using:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git


Once the repository is downloaded, it can be installed with:

.. code-block:: console

    cd pybaselines
    pip install .
