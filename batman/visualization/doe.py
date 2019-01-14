"""
Design of experiments
---------------------

Define function related to Design of Experiments.

* :func:`doe`,
* :func:`doe_ascii`,
* :func:`pairplot`.
"""
from itertools import combinations_with_replacement
import numpy as np
from sklearn import preprocessing
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
                ax.set_ylim(bottom=0)
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


def doe_ascii(sample, bounds=None, plabels=None, fname=None):
    """Plot the space of parameters 2d-by-2d in ASCII.

    :param array_like sample: sample (n_samples, n_featrues).
    :param array_like bounds: Desired range of transformed data.
          The transformation apply the bounds on the sample and not the
          theoretical space, unit cube. Thus min and max values of the sample
          will coincide with the bounds. ([min, k_vars], [max, k_vars]).
    :param list(str) plabels: parameters' names.
    :param str fname: whether to export to filename or display on console.
    """
    if bounds is not None:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(bounds)
        sample = scaler.transform(sample)
    else:
        sample = np.asarray(sample)

    print(sample)

    n_dim = sample.shape[1]

    if plabels is None:
        plabels = ['x' + str(i) for i in range(n_dim)]

    console = ''

    n_lines = 30
    n_cols = 80
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            console += '\n\nCoordinates ({}, {})\n'.format(plabels[i], plabels[j])
            console += '-' * (n_cols + 2) + '\n'  # top border
            tab = [[' ' for _ in range(n_cols)] for _ in range(n_lines)]
            for s in sample:
                x = int(s[i] * n_lines)
                y = int(s[j] * n_cols)
                tab[x][y] = '*'

            # side border
            tab = np.array(tab)
            tab = np.column_stack([['|'] * n_lines, tab, ['|'] * n_lines])

            for t in tab:
                console += ''.join(t) + '\n'

            console += '-' * (n_cols + 2) + '\n'  # bottom border

    if fname is None:
        print(console)
    else:
        with open(fname, 'w') as f:
            f.write(console)


def pairplot(sample, data, plabels=None, flabel=None, fname=None):
    """Output function of the input parameter space.

    A n-variate plot is constructed with all couple of variables - output.

    :param array_like sample: sample (n_samples, n_featrues).
    :param array_like data: data (n_samples, 1).
    :param list(str) plabels: parameters' names.
    :param str flabel: label for y axis.
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instances.
    """
    sample = np.asarray(sample)
    n_dim = sample.shape[1]

    if plabels is None:
        plabels = ['x' + str(i) for i in range(n_dim)]

    fig, axs = plt.subplots(1, n_dim)
    for i in range(n_dim):
        axs[i].scatter(sample[:, i], data, marker='+')
        axs[i].set_xlabel(plabels[i])

    axs[0].set_ylabel('F' if flabel is None else flabel)

    bat.visualization.save_show(fname, [fig])

    return fig, axs
