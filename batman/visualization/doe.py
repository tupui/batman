"""
Design of experiments
---------------------

Define function related to design of experiments.

* :func:`doe`,
"""
from itertools import combinations_with_replacement
import numpy as np
import matplotlib.pyplot as plt
import batman as bat
from .uncertainty import kernel_smoothing


def doe(sample, plabels=None, resampling=0, multifidelity=False, fname=None):
    """Plot the space of parameters 2d-by-2d.

    A n-variate plot is constructed with all couple of variables.
    The distribution on each variable is shown on the diagonal.

    :param array_like sample: sample (n_samples, n_featrues).
    :param list(str) plabels: parameters' names.
    :param int resampling: number of resampling points.
    :param bool multifidelity: whether or not the model is a multifidelity.
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    sample = np.asarray(sample)
    n_samples, dim = sample.shape
    len_sampling = n_samples - resampling

    if plabels is None:
        plabels = ["x" + str(i) for i in range(dim)]

    if multifidelity:
        sample = sample[:, 1:]
        dim -= 1
        plabels = plabels[1:]

    fig, sub_ax = plt.subplots()
    if dim < 2:
        plt.scatter(sample[0:len_sampling],
                    [0] * len_sampling, c='k', marker='o')
        plt.scatter(sample[len_sampling:],
                    [0] * resampling, c='r', marker='^')
        plt.xlabel(plabels[0])
        plt.tick_params(axis='y', which='both',
                        labelleft='off', left='off')
    else:
        # num figs = ((n-1)**2+(n-1))/2 + diag
        plt.tick_params(axis='both', labelsize=8)

        # Bivariate space
        sub_ax = []  # Axis stored as a list
        plt.tick_params(axis='both', labelsize=8)
        # Axis are created and stored top to bottom, left to right
        for i, j in combinations_with_replacement(range(dim), 2):
            ax = plt.subplot2grid((dim, dim), (j, i))
            ax.tick_params(axis='both', labelsize=(10 - dim))

            if i == j:  # diag
                x_plot = np.linspace(min(sample[:, i]),
                                     max(sample[:, i]), 100)[:, np.newaxis]
                sample_ = sample[:, i, np.newaxis]
                _ks = kernel_smoothing(sample_[:199], False)
                pdf = np.exp(_ks.score_samples(x_plot))
                ax.plot(x_plot, pdf)
                ax.hist(sample_, 30, fc='gray', histtype='stepfilled',
                        alpha=0.2, density=True)
                ax.set_ylim(ymin=0)
            elif i < j:  # lower corners
                ax.scatter(sample[0:len_sampling, i],
                           sample[0:len_sampling, j], s=5, c='k', marker='o')
                ax.scatter(sample[len_sampling:, i],
                           sample[len_sampling:, j], s=5, c='r', marker='^')

            if i == 0:
                ax.set_ylabel(plabels[j])
            if j == (dim - 1):
                ax.set_xlabel(plabels[i])

            sub_ax.append(ax)

    bat.visualization.save_show(fname, [fig])

    return fig, sub_ax
