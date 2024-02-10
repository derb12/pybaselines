============
Contributing
============

Contributions are welcomed and greatly appreciated.

Bugs Reports/Feedback
~~~~~~~~~~~~~~~~~~~~~

Report bugs or give feedback by filing an issue at https://github.com/derb12/pybaselines/issues.

If you are reporting a bug, please include:

* Your operating system, Python version, and pybaselines version.
* Any further details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce/identify the bug, including the code used and any tracebacks.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.

Pull Requests
~~~~~~~~~~~~~

Pull requests are welcomed for this project. Generally, it is preferred to file an issue first,
so that details can be discussed/finalized before a pull request is created.

Any new code or documentation must be able to be covered by the BSD 3-clause license
used by pybaselines.

When submitting a pull request, follow similar procedures for a feature request, namely:

* Explain in detail how it works.
* Keep the scope as narrow as possible to make it easier to incorporate.

The following sections will detail how to setup a development environment for contributing
code to pybaselines and all of the potential checks to run.


Setting Up Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To clone the GitHub repository and install the necessary libraries for development,
ensure `git <https://git-scm.com>`_ is installed and then run:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git
    cd pybaselines
    pip install -e .[dev]

All sections below assume the above commands were ran.

Style Guidelines
^^^^^^^^^^^^^^^^

pybaselines has the following guidelines (note: if you have any questions/concerns about
these guidelines, please feel free to open an issue or email the author; the guidelines
are meant to help keep a consistent style, not to discourage contributing :) ).

Any new code should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008>`_ standards
as closely as possible and be fully documented using
`Numpy style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
docstrings. To check that new code has the correct formatting, run the following command in the
terminal while in the pybaselines directory:

.. code-block:: console

    ruff check .


Testing
^^^^^^^

If adding new code, please add any necessary tests. To check that tests pass
locally, run the following command in the terminal while in the pybaselines directory:

.. code-block:: console

    pytest .


The testing steps below are just for reference and not necessary.

If checking coverage (not necessary, but can be helpful to know), install
`pytest-cov <https://pypi.org/project/pytest-cov>`_ and run:

.. code-block:: console

    pytest . --cov=pybaselines tests/ --cov-report=html

If checking coverage both with and without the optional dependencies, run the
above command first, install the optional dependencies, and then run:

.. code-block:: console

    pytest . --cov=pybaselines tests/ --cov-report=html --cov-append

which will append the test results with the optional dependencies to the original
coverage report to show the total code that is covered by the tests.

Documentation
^^^^^^^^^^^^^

If submitting changes to the documentation or adding documentation for a new feature/algorithm,
please ensure the documentation builds locally by running the following command while in the
``pybaselines/docs`` directory:

.. code-block:: console

    make html

and ensure that no warnings or errors are raised during building. The built documentation can
then be viewed in the ``pybaselines/docs/_build/html`` folder.

If adding a new baseline algorithm, please add a short summary of the algorithm to the
appropriate module in the
`algorithms section <https://pybaselines.readthedocs.io/en/latest/algorithms/index.html>`_,
and, if possible, add a plot showing how the algorithm fits different baselines using
matplotlib's ``plot`` directive. Look at the rst sources for any of the files in the
``pybaselines/docs/algorithms`` folder for examples.
