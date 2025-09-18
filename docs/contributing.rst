============
Contributing
============

Contributions are welcomed and greatly appreciated.

Bugs Reports/Feedback
~~~~~~~~~~~~~~~~~~~~~

Report bugs, ask questions, or give feedback by filing an issue
at https://github.com/derb12/pybaselines/issues.

If you are reporting a bug, please include:

* Your operating system, Python version, and pybaselines version.
* Any further details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce/identify the bug, including the code used and any tracebacks.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.

In addition, suggestions for improvements to the clarity/content of the documentation
or new examples showing unique applications of pybaselines would be immensely helpful!

Pull Requests
~~~~~~~~~~~~~

Pull requests are welcomed for this project. Generally, it is preferred to file an issue first,
so that details can be discussed/finalized before a pull request is created.

Any new code or documentation must be able to be covered by the BSD-3-clause license
used by pybaselines.

When submitting a pull request, follow similar procedures for a feature request, namely:

* Explain in detail how it works.
* Keep the scope as narrow as possible to make it easier to incorporate.

The following section will detail how to setup a development environment for contributing
code to pybaselines and all of the potential checks to run before submitting a pull request.

Development Workflow
~~~~~~~~~~~~~~~~~~~~

If you are unfamiliar with using `git <https://git-scm.com>`_ or GitHub, see
`GitHub's helpful guide <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project>`_
on how to create a fork with the intent to contribute to a project.

The sections below will detail the aspects of this process that are specific
to pybaselines.

Setting Up Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To clone the GitHub repository and install the necessary libraries for development,
ensure `git <https://git-scm.com>`_ is installed and then run:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git
    cd pybaselines
    pip install --editable ".[dev]"

All sections below assume the above commands were ran such that necessary
developement dependencies are available.

Style Guidelines
^^^^^^^^^^^^^^^^

Any new code should follow `PEP 8 <https://peps.python.org/pep-0008/>`_ standards
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


By default, all tests except those that test multithreading will run. Threaded tests only run
by default on free-threaded CPython builds, but they can also be enabled or disabled by
setting the command line option ``--test_threading`` to 1 or 0, respectively. For example,
to run threaded tests on a non free-threaded CPython version, run:

.. code-block:: console

    pytest . --test_threading=1


The tests for the two dimensional algorithms are quite time consuming, so if the relevant
tests are not concerened with two dimensional code, the 2D tests can be skipped with the following:

.. code-block:: console

    pytest . -k "not two_d"


Alternatively, `pytest-xdist <https://pypi.org/project/pytest-xdist/>`_ can be installed to allow
running tests in parallel to also reduce the total testing time.

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
then be viewed in the ``pybaselines/docs/_build/html`` directory.

The files generated for the examples and API documentation (rendered from the docstrings of code)
during the build process are output to the ``pybaselines/docs/generated`` directory. If you
are working on these sections of the documentation and changes are not being properly reflected in
the built documentation, it can be helpful to delete the ``pybaselines/docs/generated`` directory
so that all of the contents are regenerated upon the next documentation build (and if that does
not work, run :code:`make clean` to also remove all of the built documentation before building
the documentation again).

Doctests
^^^^^^^^

If adding new code, it is often good to show an example usage of the class/method/function
within its docstring using
`doctest format <https://numpydoc.readthedocs.io/en/latest/format.html#examples>`_. pybaselines
uses `scipy-doctest <https://github.com/scipy/scipy_doctest>`_ to ensure that these examples are
correctly formatted and will run without errors. To perform these doctests, run the following
command in the terminal while in the pybaselines directory:

.. code-block:: console

    pytest pybaselines --doctest-modules

Adding New Algorithms
^^^^^^^^^^^^^^^^^^^^^

If adding a new baseline algorithm to pybaselines:

*   Add tests for the method. pybaselines supplies testing classes within the
    ``pybaselines/tests/base_tests.py`` file that should be subclassed to ensure all basic
    requirements for a new algorithm are met. Additional tests should also be added as needed.
    See existing tests for examples.
*   Try to add example usages of the algorithm within its docstring, showing basic usage and
    any noteworthy fine-tuning. Ensure these examples run correctly by performing doctests as
    outlined above.
*   Add a short summary of the algorithm to the appropriate place in the
    `algorithms section <https://pybaselines.readthedocs.io/en/latest/algorithms/index.html>`_,
    and, if possible, add a plot showing how the algorithm fits different baselines using
    matplotlib's ``plot`` directive. Look at the rst sources for any of the files in the
    ``pybaselines/docs/algorithms`` folder for examples.
*   Add the algorithm to the `API section <https://pybaselines.readthedocs.io/en/latest/api/index.html>`_
    of the documentation.
