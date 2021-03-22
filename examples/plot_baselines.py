# -*- coding: utf-8 -*-
"""Examples of fitting using various baselines in pybaselines.

Created on March 11, 2021

@author: Donald Erb

"""


if __name__ == '__main__':

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('This example requires matplotlib to run')
        raise
    import numpy as np

    from pybaselines.baselines import optimize_parameter
    from pybaselines.morphological import mpls, imor, mor, iamor
    from pybaselines.penalized_least_squares import iarpls, airpls, arpls, asls, aspls, drpls, iasls
    from pybaselines.polynomial import imodpoly, modpoly
    from pybaselines.utils import gaussian

    x = np.linspace(200, 4000, 2000)
    signal = (
        gaussian(x, 2, 700, 50)
        + gaussian(x, 5, 1200, 300)
        + gaussian(x, 10, 1350, 200)
        + gaussian(x, 8, 1600, 100)
        + gaussian(x, 5, 2500, 50)
        + gaussian(x, 10, 3200, 200)
    )
    y = (
        0.001 * x  # linear baseline
        + gaussian(x, 10, 2000, 2000)  # gaussian baseline
        + np.random.randn(x.size) * 0.1  # noise
        + signal
    )

    baseline = asls(y, 1e7, 0.005)[0]
    baseline_2 = iasls(y, x, 10**6, 0.01, 10**-4)[0]
    baseline_3 = airpls(y, 3000, 1)[0]
    baseline_4 = arpls(y, lam=2 * 10**7)[0]
    baseline_5 = iarpls(y, lam=1e6)[0]
    baseline_6 = drpls(y, lam=1e6, eta=0.3)[0]

    baseline_8 = modpoly(y, x, 11)[0]
    baseline_10 = imodpoly(y, x, 11)[0]
    baseline_9 = aspls(y, 1e7)[0]
    baseline_11 = mpls(y, half_window=250, lam=1e6, p=0)[0]
    baseline_7 = iamor(y, half_window=70)[0]
    baseline_12 = imor(y, half_window=100, smooth=False)[0]
    baseline_14 = mor(y, half_window=250)[0]
    baseline_13 = optimize_parameter(y, x, 'mpls', 'both', half_window=250)[0]

    fig = plt.figure()
    plt.plot(x, y, label='data')
    plt.plot(x, y - baseline, label='AsLS')
    plt.plot(x, y - baseline_2, label='iAsLS')
    plt.plot(x, y - baseline_3, label='airPLS')
    plt.plot(x, y - baseline_5, label='IarPLS')
    plt.plot(x, y - baseline_6, label='drPLS')
    plt.plot(x, y - baseline_4, label='ArPLS')
    plt.plot(x, y - baseline_8, label='ModPoly')
    plt.plot(x, y - baseline_11, label='MPLS')
    plt.plot(x, y - baseline_12, label='IMor')
    plt.plot(x, y - baseline_7, label='iaMor')
    plt.plot(x, y - baseline_14, label='Mor')
    plt.plot(x, y - baseline_13, label='erPLS')
    plt.plot(x, y - baseline_9, '--', label='aspls')
    plt.plot(x, y - baseline_10, '--', label='imodpoly')
    plt.plot(x, signal, '--', label='actual')

    plt.legend().set_draggable(True)
    plt.show

    fig = plt.figure()
    plt.plot(
        x, y,
        x, baseline,
        x, baseline_2,
        x, baseline_3,
        x, baseline_5,
        x, baseline_4,
        x, baseline_6,
        x, baseline_8,
        x, baseline_11,
        x, baseline_12,
        x, baseline_7,
        x, baseline_14,
        x, baseline_13,
        x, baseline_9,
        x, baseline_10,
        x, y - signal, '--'
    )
    plt.legend([
        'original',
        'AsLS',
        'iAsLS',
        'airPLS',
        'IarPLS',
        'ArPLS',
        'drPLS',
        'ModPoly',
        'MPLS',
        'IMor',
        'iaMor',
        'Mor',
        'erPLS',
        'aspls',
        'imodpoly',
        'actual'
    ]).set_draggable(True)
    plt.show(block=True)
