=======
Masking
=======

There are cases where data needs to be removed/masked before baseline correction. For example, a faulty
detector can result in problematic regions in measurements, or spurious values/peaks below the
expected baseline can cause issues with many algorithms which only expect positive peaks. In these
cases, baseline correction on the raw data could lead to severely incorrect fits. One such use of
masking in literature is presented by `Temmink, et al. <https://doi.org/10.1051/0004-6361/202348911>`_
for removing downward spikes in mid-infrared data collected from the James Webb Space Telescope before
performing baseline correction.

Beginning in version 1.3.0, pybaselines has added direct support for masking to many algorithms, which
can be used by initializing a :class:`~.Baseline` with a mask or setting the :attr:`~.Baseline.mask`
property of an existing instance. Using the same conventions as :class:`numpy.ma.MaskedArray` and
:func:`astropy.convolution.convolve`, the mask should be a Boolean
array with ``True`` values indicating the indices within the data to omit from fitting.

A simple example is shown below. In the example, the mask is set manually, but it could
alternatively be set following some metric. For example,
`Temmink, et al. <https://doi.org/10.1051/0004-6361/202348911>`_, used
Savitzky-Golay filtering combined with iterative thresholding to define a mask that excluded
negative peaks.

.. plot::
   :align: center
   :context: reset
   :include-source: True

    import matplotlib.pyplot as plt
    import numpy as np
    from pybaselines import Baseline
    from pybaselines.utils import gaussian

    x = np.linspace(500, 4000, 1000)
    signal = (
        + gaussian(x, 8, 650, 16)
        + gaussian(x, 9, 1100, 50)
        + gaussian(x, 8, 1350, 20)
        + gaussian(x, 11, 2800, 20)
        + gaussian(x, 8, 2900, 20)
        + gaussian(x, 5, 3400, 40)
    )
    baseline = 0.08 + 0.00004 * (x - 1000) + gaussian(x, 10, 1900, 800)
    rng = np.random.default_rng(123)
    noise = rng.normal(0, 0.1, len(x))
    y = signal + baseline + noise

    # simulate an issue with the detector in the indicated region
    bad_region = (x > 2000) & (x < 2500)
    y[bad_region] = rng.normal(0, 0.25, len(x[bad_region]))

    # defining the mask manually by eye, should be True in regions to ignore
    mask = (x > 1900) & (x < 2550)

    baseline_fitter = Baseline(x)
    non_masked_fit, non_masked_params = baseline_fitter.arpls(y, lam=1e5)
    baseline_fitter.mask = mask  # can also set mask upon initializing a new Baseline object
    masked_fit, masked_params = baseline_fitter.arpls(y, lam=1e5)

    plt.plot(x, y)
    plt.plot(x, non_masked_fit, label='not masked')
    plt.plot(x, masked_fit, label='masked')
    plt.legend()


When possible, the supplied mask is used to completely omit the indicated values
during the baseline fitting while also allowing estimation of the baseline in the masked regions,
for example through weighted interpolation.
Some methods, however, do not (currently) support masking in such a numerically correct way,
so by default these methods will raise an error when trying to call them if the ``mask`` property
is not ``None``. If the :class:`~.Baseline` object is initialized with ``strict_mask=False``,
then these methods will use linear interpolation to fill masked regions before performing
baseline correction, similar to the
:ref:`No Masking Support <user_guide/masking/index:No Masking Support>` section below.
The code snippet below demonstrates this.

.. code-block:: python

    Baseline(mask=mask).beads(y)  # raises NotImplementedError
    Baseline(mask=mask, strict_mask=False).beads(y)  # interpolates, then calculates


The tables below indicate all baseline correction methods that currently support masking,
meaning ``strict_mask`` does not need to be set to ``False`` to use masking with them.

.. plot::
   :align: center
   :context: close-figs
   :include-source: False
   :show-source-link: False
   :nofigs:

    import inspect
    from pathlib import Path

    from pybaselines import Baseline, Baseline2D, utils

    def make_support_table(baseline_class: Baseline | Baseline2D):
        methods = []
        for (method_name, method) in inspect.getmembers(baseline_class):
            if (
                    inspect.isfunction(method)
                    and not method_name.startswith('_')
            ):
                    methods.append(method_name)
        methods.sort()

        _, y = utils.make_data(100 if baseline_class is Baseline else 30)
        if baseline_class is Baseline2D:
            y = y * np.ones((15, len(y)))
        masked_methods = set()
        fitter = baseline_class(mask=np.zeros(y.shape, dtype=bool))
        for method in methods:
            try:
                getattr(fitter, method)(y)
                masked_methods.add(method)
            except NotImplementedError:
                pass
            except ValueError:
                if method == 'collab_pls':
                    # fails since it expects 2D/3D input, but does support masking
                    masked_methods.add(method)
                else:
                    raise

        txt = ".. list-table::\n  :align: center\n  :header-rows: 1\n\n  * - Method\n    - Supports Masking"

        # TODO should the version mask support was added also be included in the future? Would
        # need to save the mask support table then.
        for method in methods:
            txt += f"\n  * - :meth:`~.{baseline_class.__name__}.{method}`\n    - {'✓' if method in masked_methods else ' '}"

        # use the generated/images folder since it's not used by other extensions
        output_path = Path('../../generated/images')
        output_path.mkdir(exist_ok=True, parents=True)
        dim = '1d' if baseline_class is Baseline else '2d'
        with output_path.joinpath(f'mask_support_table_{dim}.rst').open('w', encoding='utf-8') as f:
            f.write(txt)

    make_support_table(Baseline)
    make_support_table(Baseline2D)

.. dropdown:: Mask Support Table for ``Baseline``

  .. include:: ../../generated/images/mask_support_table_1d.rst

.. dropdown:: Mask Support Table for ``Baseline2D``

  .. include:: ../../generated/images/mask_support_table_2d.rst


NaN Values
----------

Outside of masking, there is no additional special-case handling of NaN values within
pybaselines. A rough comparison with the ``nan_policy`` usage in :mod:`scipy.stats` is:

* ``nan_policy='raise'`` corresponds to initializing a ``Baseline`` object with
  ``check_finite=True`` and ``mask=None``. Any non-finite value will raise an exception.
* ``nan_policy='omit'`` roughly corresponds to the ``mask`` usage described above,
  although masked regions are filled in the output using e.g. weighted interpolation.
  To have NaN values in the output, simply do:

  .. code-block:: python

      fit, params = Baseline(mask=mask).modpoly(y)
      fit[mask] = np.nan

  In regards to the interaction between ``check_finite`` and ``mask``:
  values within y are ignored following the input mask. This means
  that if ``mask[i]`` is ``True``, it does not matter if ``y[i]`` is NaN, infinite,
  or a finite value, it will be ignored. However, if ``mask[i]`` is ``False``, the
  handling of non-finite values follows the same ``check_finite`` behavior as without
  masking, i.e. raising if ``y[i]`` is non-finite.
* ``nan_policy='propagate'`` roughly corresponds to initializing a ``Baseline`` object
  with ``check_finite=False`` and ``mask=None``, except there is no guarantee on how any
  single method handles NaN values and as such is undefined (and unsupported) behavior
  in pybaselines. To illustrate, some methods that use convolution may indeed propagate
  NaN values in an expected way, others that use sliding windows will fully carry
  NaN values in the result once one is encountered (as demonstrated in
  `this SciPy issue <https://github.com/scipy/scipy/issues/7818>`_ ), and some may raise
  exceptions.


Masking in Earlier Versions
---------------------------

Prior to pybaselines version 1.3.0, masking had to be done by users. The various
baseline correction methods in pybaselines fall in one of three categories:

1) Directly support masking
2) Indirectly support masking
3) No mask support

Further details on how to handle each category will be covered in detail below.

Direct Masking Support
^^^^^^^^^^^^^^^^^^^^^^

The only methods that directly supported masking were non-iteratively-reweighted polynomial
methods, which includes all :doc:`polynomial <../algorithms/algorithms_1d/polynomial>` methods
except for :meth:`~.Baseline.loess` and :meth:`~.Baseline.quant_reg`. For these methods,
the inverse of the mask needs to be input as weights (0 or ``False`` in regions to ignore),
as shown in the example below. As discussed above, these methods are not NaN-aware, however, so
if working with missing data, that has to be accounted for before baseline correction, such
as by using :func:`numpy.nan_to_num`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: True

    # using same x, y, and mask as created above
    baseline_fitter = Baseline(x)

    weights = np.logical_not(mask)

    non_masked_fit, non_masked_params = baseline_fitter.imodpoly(y, poly_order=7)
    masked_fit, masked_params = baseline_fitter.imodpoly(y, poly_order=7, weights=weights)

    plt.plot(x, y)
    plt.plot(x, non_masked_fit, label='not masked')
    plt.plot(x, masked_fit, label='masked')
    plt.legend()


Indirect Masking Support
^^^^^^^^^^^^^^^^^^^^^^^^

Next are algorithms that use iterative reweighting, which allow for indirect mask support.
This includes :doc:`Whittaker smoothing methods <../algorithms/algorithms_1d/whittaker>`, most
:doc:`spline methods <../algorithms/algorithms_1d/spline>`, :meth:`~.Baseline.loess` and
:meth:`~.Baseline.quant_reg`.

These methods allow inputting weights, but the input weights are just used for the first
iteration to jump-start the calculation and are ignored in subsequent iterations.
To emulate "mask-aware" behavior with these algorithms, interpolate the data in the mask
regions and fit an initial baseline, take the output weights and set the weights in mask regions
to 0, and then call the method again while setting ``tol=np.inf`` to only perform one iteration.
The end result will be a weighted interpolation in mask regions that typically closely approximates
an actual "mask-aware" implementation. Note that the interpolation of the input for the
first step will not affect the final result much, so simple linear interpolation will suffice.

.. plot::
   :align: center
   :context: close-figs
   :include-source: True

    # using same x, y, and mask as created above
    baseline_fitter = Baseline(x)

    fit_mask = np.logical_not(mask)
    y_interp = np.interp(x, x[fit_mask], y[fit_mask])

    non_masked_fit = baseline_fitter.arpls(y, lam=1e5)[0]
    initial_fit, params = baseline_fitter.arpls(y_interp, lam=1e5)
    weights = params['weights']
    weights[mask] = 0
    weighted_fit = baseline_fitter.arpls(y_interp, lam=1e5, weights=weights, tol=np.inf)[0]

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, non_masked_fit, label='not masked')
    plt.plot(x, initial_fit, label='initial interpolated fit')
    plt.plot(x, weighted_fit, '--', label='final weighted interpolation')

    plt.legend()


No Masking Support
^^^^^^^^^^^^^^^^^^

All other algorithms that are not covered above do not have a direct way of incorporating
masking for external code. For these algorithms, the input data must be interpolated following
the mask before performing baseline correction, similar to the "indirect masking" algorithms
covered above; however, for these methods, the quality of the interpolation can have
a more pronounced effect on the calculated baseline. The example below shows the
difference between linear interpolation and
:class:`PCHIP interpolation <scipy.interpolate.PchipInterpolator>`
using :meth:`~.Baseline.mor`.

.. plot::
   :align: center
   :context: close-figs
   :include-source: True

    from scipy.interpolate import PchipInterpolator

    # using same x, y, and mask as created above
    baseline_fitter = Baseline(x)

    fit_mask = np.logical_not(mask)
    y_linear = np.interp(x, x[fit_mask], y[fit_mask])
    y_pchip = PchipInterpolator(x[fit_mask], y[fit_mask])(x)

    _, (ax1, ax2) = plt.subplots(2, layout='constrained')
    ax1.set_title('Interpolated Data')
    ax2.set_title('Calculated Baselines using "mor"')

    ax1.plot(x, y)
    ax1.plot(x, y_linear, label='linear interpolation')
    ax1.plot(x, y_pchip, label='PCHIP interpolation')
    ax1.legend()

    half_window = 35
    mor_linear = baseline_fitter.mor(y_linear, half_window=half_window)[0]
    mor_pchip = baseline_fitter.mor(y_pchip, half_window=half_window)[0]
    non_masked = baseline_fitter.mor(y, half_window=half_window)[0]

    ax2.plot(x, y)
    ax2.plot(x, mor_linear, label='linear interpolation')
    ax2.plot(x, mor_pchip, label='PCHIP interpolation')
    ax2.plot(x, non_masked, label='not masked')
    ax2.legend()

    plt.legend()
