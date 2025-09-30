.. highlight:: shell

============
Installation
============


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://www.python.org>`_ version 3.9 or later and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.20)
* `SciPy <https://scipy.org>`_ (>= 1.6)


All of the required libraries should be automatically installed when
installing pybaselines using any of the installation methods below.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

pybaselines has the following optional dependencies:

* `Numba <https://github.com/numba/numba>`_ (>= 0.53):
  speeds up calculations used by the following methods:

  * :meth:`~.Baseline.loess`
  * :meth:`~.Baseline.dietrich`
  * :meth:`~.Baseline.golotvin`
  * :meth:`~.Baseline.std_distribution`
  * :meth:`~.Baseline.fastchrom`
  * :meth:`~.Baseline.beads`
  * :meth:`~.Baseline.mpspline`
  * :meth:`~.Baseline.mpls`
  * :meth:`~.Baseline.jbcd`
  * :meth:`~.Baseline.fabc`
  * all :ref:`spline <api/Baseline:Spline Algorithms>` methods
  * all :ref:`Whittaker smoothing <api/Baseline:Whittaker Smoothing Algorithms>` methods


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
`GitHub repository <https://github.com/derb12/pybaselines>`_.

To directly install the current version of pybaselines from GitHub,
ensure `git <https://git-scm.com>`_ is installed and then run:

.. code-block:: console

    pip install git+https://github.com/derb12/pybaselines.git

Alternatively, to download the entire public repository and install pybaselines, run:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git
    cd pybaselines
    pip install .
