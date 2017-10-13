"""
Design of experiments
---------------------

Define function related to design of experiments.

* :func:`doe`,
* :func:`response_surface`.
"""
from itertools import combinations_with_replacement
import numpy as np
from scipy.interpolate import griddata
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
import batman as bat
from .uncertainty import kernel_smoothing


def doe(sample, p_lst=None, resampling=0, multifidelity=False, fname=None,
        show=True):
    """Plot the space of parameters 2d-by-2d.

    A n-variate plot is constructed with all couple of variables.
    The distribution on each variable is shown on the diagonal.

    :param array_like sample: sample (n_samples, n_featrues).
    :param list(str) p_lst: parameters' names.
    :param int resampling: number of resampling points.
    :param bool multifidelity: whether or not the model is a multifidelity.
    :param str fname: whether to export to filename or display the figures.
    :param bool show: whether to show the plot if not :attr:`fname`.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    sample = np.asarray(sample)
    n_samples, dim = sample.shape
    len_sampling = n_samples - resampling

    if p_lst is None:
        p_lst = ["x" + str(i) for i in range(dim)]

    if multifidelity:
        sample = sample[:, 1:]
        dim -= 1

    fig, sub_ax = plt.subplots()
    if dim < 2:
        plt.scatter(sample[0:len_sampling],
                    [0] * len_sampling, c='k', marker='o')
        plt.scatter(sample[len_sampling:],
                    [0] * resampling, c='r', marker='^')
        plt.xlabel(p_lst[0])
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
                scaler = preprocessing.MinMaxScaler().fit(sample[:, i, np.newaxis])
                sample_scaled = scaler.transform(sample[:, i, np.newaxis])
                _ks = kernel_smoothing(sample_scaled, False)
                pdf = np.exp(_ks.score_samples(scaler.transform(x_plot)))
                ax.plot(x_plot, pdf)
                ax.fill_between(x_plot[:, 0], pdf, [0] * x_plot.shape[0],
                                color='gray', alpha=0.1)
            elif i < j:  # lower corners
                ax.scatter(sample[0:len_sampling, i], sample[
                    0:len_sampling, j], s=5, c='k', marker='o')
                ax.scatter(sample[len_sampling:, i], sample[
                    len_sampling:, j], s=5, c='r', marker='^')

            if i == 0:
                ax.set_ylabel(p_lst[j])
            if j == (dim - 1):
                ax.set_xlabel(p_lst[i])

            sub_ax.append(ax)

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight')
    elif show:
        plt.show()
    plt.close('all')

    return fig, sub_ax


def response_surface(bounds, sample=None, data=None, fun=None, doe=None,
                     resampling=0, xdata=None, flabel='F', plabels=None,
                     fname=None):
    """Response surface visualization.

    You have to set either (i) :attr:`sample` with :attr:`data` or  (ii)
    :attr:`fun` depending on your data. If (i), the data are interpolated
    on a mesh in order to be plotted as a surface. Otherwize, :attr:`fun` is
    directly used to generate correct data.

    The DoE can also be plotted by setting :attr:`doe` along with
    :attr:`resampling`.

    :param array_like bounds: sample boundaries
    ([min, n_features], [max, n_features]).
    :param array_like sample: sample (n_samples, n_featrues).
    :param array_like data: function evaluations(n_samples, [n_featrues]).
    :param callable fun: function to plot the response from.
    :param array_like doe: design of experiment (n_samples, n_features).
    :param int resampling: number of resampling points.
    :param array_like xdata: 1D discretization of the function (n_features,).
    :param str flabel: name of the quantity of interest.
    :param list(str) plabels: parameters' labels.
    :param str fname: wether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    dim = len(bounds[0])
    if dim == 1:
        n_samples = 50
    elif dim == 2:
        n_samples = 625
    n_samples = int(np.floor(np.power(n_samples, 1.0 / dim)))

    grids = [np.linspace(bounds[0][i], bounds[1][i], n_samples) for i in range(dim)]

    if dim == 2:
        grids = np.meshgrid(*grids)
        xsample, ysample = grids
        xsample = xsample.flatten()
        ysample = ysample.flatten()

    if fun is not None:
        data = fun(np.stack([grid.flatten() for grid in grids]).T)

    if xdata is not None:
        data = np.trapz(data[:], xdata) / (np.max(xdata) - np.min(xdata))

    if fun is None:
        data = griddata(sample, data, tuple(grids), method='nearest')

    data = data.flatten()

    if plabels is None:
        plabels = ["x" + str(i) for i in range(dim)]

    fig = plt.figure('Response Surface')

    if dim == 1:
        plt.plot(grids[0], data)
        plt.ylabel(flabel, fontsize=28)
    elif dim == 2:
        plt.tricontourf(xsample, ysample, data,
                        antialiased=True, cmap=cm.viridis, label=None)
        if doe is not None:
            doe = np.asarray(doe)
            len_sampling = len(doe) - resampling
            plt.plot(doe[:, 0][0:len_sampling], doe[:, 1][0:len_sampling], 'ko')
            plt.plot(doe[:, 0][len_sampling:], doe[:, 1][len_sampling:], 'r^')

        plt.ylabel(plabels[1], fontsize=28)
        cbar = plt.colorbar()
        cbar.set_label(flabel, fontsize=28)
        cbar.ax.tick_params(labelsize=28)

    plt.xlabel(plabels[0], fontsize=28)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)

    bat.visualization.save_show(fname, [fig])

    return fig
