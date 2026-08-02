==============================
Using Masking with pybaselines
==============================

pybaselines version 1.3.0 added direct support for masking to many algorithms by adding
the :attr:`.Baseline.mask` property. Much like :class:`numpy.ma.MaskedArray` and
:func:`astropy.convolution.convolve`, the mask should be a boolean array with
``True`` values indicating the indices within the data to omit from fitting. A simple
example is shown below.

.. plot::
   :align: center
   :context: reset
   :include-source: True

    import matplotlib.pyplot as plt
    import numpy as np
    from pybaselines import Baseline, utils

    x, y = utils.make_data()
    # simulate an issue with the detector in the indicated region
    bad_region = (x > 600) & (x < 650)
    y[bad_region] = np.random.default_rng().normal(0.5, 0.25, len(x[bad_region]))

    baseline_fitter = Baseline(x)
    non_masked_fit, non_masked_params = baseline_fitter.arpls(y)
    baseline_fitter.mask = bad_region  # can also set mask upon initializing a new Baseline object
    masked_fit, masked_params = baseline_fitter.arpls(y)

    plt.plot(x, y)
    plt.plot(x, non_masked_fit, label='not masked')
    plt.plot(x, masked_fit, label='masked')
    plt.legend()


When possible, the supplied mask is used to completely omit the indicated values
from the baseline fitting while also allowing estimation of the baseline in the masked regions,
for example by setting weights to 0. Some methods, however, do not support masking
in such a numerically correct way, so by default these methods will raise an error when trying
to call them if the ``mask`` property is not None. If the :class:`~.Baseline` object is
initialized with ``strict_mask=False``, then these methods will use linear interpolation
to fill masked regions to indirectly use masking.

The table below indicates all baseline correction methods that currently support masking
(i.e. ``strict_mask`` does not need to be set to ``False``).

.. plot::
   :align: center
   :context: reset
   :include-source: False
   :show-source-link: False
   :nofigs:

    import inspect
    from pathlib import Path

    from pybaselines import Baseline, utils

    methods = []
    for (method_name, method) in inspect.getmembers(Baseline):
        if (
                inspect.isfunction(method)
                and not method_name.startswith('_')
        ):
                methods.append(method_name)
    methods.sort()

    x, y = utils.make_data()
    masked_methods = set()
    fitter = Baseline(x, mask=np.zeros(y.shape, dtype=bool))
    for method in methods:
        try:
            getattr(fitter, method)(y)
            masked_methods.add(method)
        except NotImplementedError:
            pass
        except ValueError:
            if method == 'collab_pls':  # fails since it expects 2D input
                masked_methods.add(method)
            pass

    txt = """
    .. list-table::
      :align: center
      :header-rows: 1

      * - Method
        - Supports Masking"""

    # TODO should the version mask support was added also be included in the future? Would
    # need to save the mask support table then.
    for method in methods:
        txt += f"\n  * - :meth:`~.Baseline.{method}`\n    - {'✓' if method in masked_methods else ' '}"

    # use the generated/images folder since it's not used by other extensions
    output_path = Path('../../generated/images')
    output_path.mkdir(exist_ok=True, parents=True)
    with output_path.joinpath('mask_support_table.rst').open('w', encoding='utf-8') as f:
        f.write(txt)

.. include:: ../../generated/images/mask_support_table.rst


Masking in Earlier Versions
---------------------------

Prior to pybaselines version 1.3.0, masking had to be done by users. The various
baseline correction methods in pybaselines fall in one of three categories:

1) Directly support masking
2) Indirectly support masking
3) No mask support

Direct Masking Support
----------------------

discuss setting weights as ~mask for polynomials.

Indirect Masking Support
------------------------

discuss reweighting here

No Masking Support
------------------

discuss interpolation here
