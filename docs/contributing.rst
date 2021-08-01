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

Pull requests are welcomed for this project, but please note that
unsolicited pull requests are discouraged. Please file an issue first,
so that details can be discussed/finalized before a pull request is created.

Any new code or documentation must be able to be covered by the BSD 3-clause license
used by pybaselines.

When submitting a pull request, follow similar procedures for a feature request, namely:

* Explain in detail how it works.
* Keep the scope as narrow as possible to make it easier to incorporate.

Setup Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To clone the GitHub repository and install the necessary libraries for development:

.. code-block:: console

    git clone https://github.com/derb12/pybaselines.git
    pip -r install pybaselines/requirements/requirements-development.txt


Style Guidelines
^^^^^^^^^^^^^^^^

pybaselines has the following guidelines (note: if you have any questions/concerns about
these guidelines, please feel free to open an issue or email the author; the guidelines
are meant to help keep a consistent style, not to discourage contributing :) )

Any new code should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008>`_ standards
as closely as possible and be fully documented using
`Numpy style <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
docstrings. To check that new code has the correct formatting, ensure that flake8 and
flake8-docstrings are installed (they should be if the development environment was setup
as described in the section above), and run the following command in the terminal while in
the pybaselines directory:

.. code-block:: console

    flake8 .

If implementing a new feature, please provide documentation discussing its
implementation, and any necessary tests. To check that tests pass locally, ensure
that pytest is installed and run the following command in the terminal while in the
pybaselines directory:

.. code-block:: console

    pytest . --verbose
