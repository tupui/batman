import numpy as np
from scipy.interpolate import griddata
import itertools
from matplotlib import cm
import matplotlib.pyplot as plt


def doe(sample, p_lst=None, resampling=0, multifidelity=False, fname=None):
    """Plot the space of parameters 2d-by-2d.

    :param array_like sample: sample (n_samples, n_featrues).
    :param list(str) p_lst: parameters' names.
    :param int resampling: number of resampling points.
    :param bool multifidelity: whether or not the model is a multifidelity.
    :param str fname: whether to export to filename or display the figures.
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

    fig = plt.figure('Design of Experiment')

    if dim < 2:
        plt.scatter(sample[0:len_sampling],
                    [0] * len_sampling, c='k', marker='o')
        plt.scatter(sample[len_sampling:],
                    [0] * resampling, c='r', marker='^')
        plt.xlabel(p_lst[0])
        plt.tick_params(axis='y', which='both',
                        labelleft='off', left='off')
    else:
        # num figs = ((n-1)**2+(n-1))/2
        plt.tick_params(axis='both', labelsize=8)

        for i, j in itertools.combinations(range(0, dim), 2):
            ax = plt.subplot2grid((dim, dim), (j, i))
            ax.scatter(sample[0:len_sampling, i], sample[
                0:len_sampling, j], s=5, c='k', marker='o')
            ax.scatter(sample[len_sampling:, i], sample[
                len_sampling:, j], s=5, c='r', marker='^')
            ax.tick_params(axis='both', labelsize=(10 - dim))
            if i == 0:
                ax.set_ylabel(p_lst[j])
            if j == (dim - 1):
                ax.set_xlabel(p_lst[i])

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')

    return fig


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
    n_samples = int(np.floor(np.power(n_samples, 1 / dim)))

    grids = [np.linspace(bounds[0][i], bounds[1][i], n_samples) for i in range(dim)]

    if dim == 1:
        grids = grids
    else:
        grids = np.meshgrid(*grids)
        xsample, ysample = grids
        xsample = xsample.flatten()
        ysample = ysample.flatten()

    if fun is not None:
        data = fun(np.stack([grid.flatten() for grid in grids]).T)

    if xdata is not None:
        data = np.trapz(data[:], xdata) / (np.max(xdata) - np.min(xdata))

    if fun is None:
        data = griddata(sample, data, (*grids,), method='nearest')

    data = data.flatten()

    if plabels is None:
        plabels = ["x" + str(i) for i in range(dim)]

    c_map = cm.viridis
    fig = plt.figure('Response Surface')

    if dim == 1:
        plt.plot(*grids, data)
        plt.ylabel(flabel, fontsize=28)
    elif dim == 2:
        plt.tricontourf(xsample, ysample, data,
                        antialiased=True, cmap=c_map)
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
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')

    return fig
