.. highlight:: shell

============
Installation
============


Dependencies
~~~~~~~~~~~~

pybaselines requires `Python <https://python.org>`_ version 3.6 or later and the following libraries:

* `NumPy <https://numpy.org>`_ (>= 1.14)
* `SciPy <https://www.scipy.org/scipylib/index.html>`_


All of the required libraries should be automatically installed when installing pybaselines
using either of the two installation methods below.


Stable Release
~~~~~~~~~~~~~~

pybaselines is easily installed from `pypi <https://pypi.org/project/pybaselines>`_
using `pip <https://pip.pypa.io>`_, simply by running the following command in your terminal:

.. code-block:: console

    pip install --upgrade pybaselines

This is the preferred method to install pybaselines, as it will always install the most
recent stable release. Note that the ``--upgrade`` tag is used to ensure that the
most recent version of pybaselines is downloaded and installed, even if an older version
is currently installed.


Development Version
~~~~~~~~~~~~~~~~~~~

The sources for pybaselines can be downloaded from the `Github repo <https://github.com/derb12/pybaselines>`_.

The public repository can be cloned using:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git


Once the repository is downloaded, it can be installed with:

.. code-block:: console

    cd pybaselines
    python setup.py install
