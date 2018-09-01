"""
Density-based measures
----------------------

This module is based on code initially written by Irene Witte (University of
Hohenheim).
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import batman as bat


def cusunoro(sample, data, plabels=None, fname=None):
    """CUmulative SUms of NOrmalised Reordered Output.

    1. Data are normalized (mean=0, variance=1),
    2. Chose a feature and order its values,
    3. Order normalized data accordingly,
    4. Compute the cumulative sum vector.
    5. Plot and repeat for all features.

    :param array_like sample: Sample of parameters of Shape
      (n_samples, n_params).
    :param array_like data: Sample of realization which corresponds to the
      sample of parameters :attr:`sample` (n_samples, ).
    :param list(str) plabels: Names of each parameters (n_features).
    :param str fname: wether to export to filename or display the figures.
    :returns: figure, axis and sensitivity indices.
    :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instance,
      array_like.
    """
    sample = np.asarray(sample)
    data = np.asarray(data).flatten()

    ns, dim = sample.shape
    if plabels is None:
        plabels = ['x' + str(i) for i in range(dim)]
    else:
        plabels = plabels

    """Cusunoro computation and plot."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Normalization (mean=0, var=1) for first and second moment
    diff_y = (data - data.mean())
    w = diff_y ** 2
    ynorm = (diff_y) / (ns * np.std(data))
    ynorm2 = (w - w.mean()) / (ns * np.std(w))

    s_indices = np.zeros(dim)

    interval = int(ns / 500) if ns > 500 else 1
    n_partition = int(np.ceil(np.sqrt(ns)))  # from Plischke

    for i in range(dim):
        # Reordering and cumulative sum
        idx = sample[:, i].argsort()
        cumsum = np.cumsum(ynorm[idx])
        cumsum2 = np.cumsum(ynorm2[idx])

        # Estimation of the first order effect from the cusunoro curve
        s_indices[i] = sum(np.diff(cumsum[::n_partition])**2) * n_partition
        # Si_max = (condmax[1]**2)*(1/condmax[0] + 1/(1 - condmax[0]))

        # Plot
        yplot = np.append(cumsum[::interval], 0)
        yplot2 = cumsum2[::interval]

        ax[0].plot(np.linspace(0, 1, yplot.shape[0]), yplot)
        ax[1].plot(np.linspace(0, 1, yplot2.shape[0]), yplot2, label=plabels[i])

    for k in [0, 1]:
        ax[k].set_xlabel("Empirical CDF of model parameters")
    ax[0].set_ylabel('Cumulative sums of normalized model output')
    ax[1].set_ylabel('Cumulative sums for second moment')

    ax[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    bat.visualization.save_show(fname, [fig])

    return fig, ax, s_indices
