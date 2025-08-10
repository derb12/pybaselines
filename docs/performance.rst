=====================
Improving Performance
=====================

pybaselines was designed for performant single-threaded, single-process usage. This page
gives tips for improving the performance when fitting multiple datasets.

When fitting multiple datasets that share the same independant variables, it is more efficient to
reuse the same :class:`~.Baseline` object rather than creating a new ``Baseline`` object for each
method call since much of the setup only needs to be done once and can be reused otherwise, as
shown in :ref:`this example <sphx_glr_generated_examples_general_plot_reuse_baseline.py>`.

For methods that require a ``half_window`` parameter, such as :doc:`morphological <algorithms/morphological>`
and :doc:`smoothing <algorithms/smooth>` algorithms, the ``half_window`` is estimated using
the :func:`~.optimize_window` function if no ``half_window`` value is given, which can significantly increase
computation time when fitting multiple datasets. If all data have similar peak widths, it would be much
faster to either specify the ``half_window`` value or use :func:`~.optimize_window` on a single set of data
and then use the output ``half_window`` value for all subsequent baseline fits for the dataset.

For fitting datasets that are quite large (>~ 5,000 individual spectra/diffractograms), users
can opt to use multiprocessing or, potentially, threading to reduce the computation time. These
uses will be addressed below.


Parallel Processing
-------------------

Multiprocessing through the standard library
`multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_ module or
third-party libraries works well with pybaselines. A simple usage is shown below:

.. code-block:: python

        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        from pybaselines import Baseline


        x = ...  # the x-values for the data
        dataset = ...  # the total data, with shape (number of datasets, number of data points)
        kwargs = {...}  # any keyword arguments to pass to the method

        baseline_fitter = Baseline(x)
        # bind any needed keyword arguments to the method (arpls in this example)
        partial_func = partial(baseline_fitter.arpls, **kwargs)
        baselines = np.empty_like(dataset)
        with ProcessPoolExecutor() as pool:
            for i, (baseline, params) in enumerate(pool.map(partial_func, dataset)):
                baselines[i] = baseline


In pybaselines versions earlier than 1.2.0, the :meth:`~.Baseline.loess` method could cause issues
when used with multiprocessing on POSIX systems, since ``loess`` spawned its own internal threads
and conflicted with the ``fork`` method of spawning processes (the default method of spawning processes
on non-macOS POSIX systems prior to Python version 3.14). To work around this, the process start method
simply needs to be explicitly set to ``spawn`` instead when using ``loess``. The above example would be
modified like so:


.. code-block:: python

        from multiprocessing import get_context

        with ProcessPoolExecutor(mp_context=get_context('spawn')) as pool:
            ...  # do the processing


Threading
---------

Starting with pybaselines version 1.2.0, pybaselines has experimental support for the free-threaded
build of CPython (see https://py-free-threading.github.io/ for more information) to allow the use of
multithreading through the standard library `threading <https://docs.python.org/3/library/threading.html>`_
module to decrease computation time. In CPython versions earlier than 3.13, or for non-free-threaded
CPython builds, it is not recommended to use multithreading with pybaselines since most operations
within pybaselines do not release the GIL.

If using pybaselines version 1.2.0 or later, :class:`~.Baseline` and :class:`~.Baseline2D` objects are
thread-safe, so the same object can be used for all threads. An example use case is shown below.

.. code-block:: python

        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        from pybaselines import Baseline


        x = ...  # the x-values for the data
        dataset = ...  # the total data, with shape (number of datasets, number of data points)
        kwargs = {...}  # any keyword arguments to pass to the method

        baseline_fitter = Baseline(x)
        # bind any needed keyword arguments to the method (arpls in this example)
        partial_func = partial(baseline_fitter.arpls, **kwargs)
        baselines = np.empty_like(dataset)
        with ThreadPoolExecutor() as pool:
            for i, (baseline, params) in enumerate(pool.map(partial_func, dataset)):
                baselines[i] = baseline


Note that thread-safety is only guaranteed if non-data inputs (eg. ``lam``, ``poly_order``,
``half_window``, etc.) are the same for all method calls. Otherwise, race conditions are likely
(and threading is likely not a good choice for the user in the first place...).

In pybaselines versions earlier than 1.2.0, several methods of :class:`~.Baseline` and :class:`~.Baseline2D`
were not thread-safe, so the proper way to use multithreading would be to spawn a new :class:`~.Baseline`
or :class:`~.Baseline2D` object for each method call, as shown below.

.. code-block:: python


        def func(x, baseline_method, data, **kwargs):
            """Helper to make a new Baseline each function call."""
            return getattr(Baseline(x), baseline_method)(data, **kwargs)

        method = 'arpls'  # a string designating the method to use
        partial_func = partial(func, x, method, **kwargs)
        baselines = np.empty_like(dataset)
        with ThreadPoolExecutor() as pool:
            for i, (baseline, params) in enumerate(pool.map(partial_func, dataset)):
                baselines[i] = baseline
