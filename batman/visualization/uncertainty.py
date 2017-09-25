"""
Uncertainty visualization tools
-------------------------------

It regoups various functions for graph visualizations.
"""
import numpy as np
import openturns as ot
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import differential_evolution

from ..input_output import (IOFormatSelector, Dataset)
import batman as bat

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt


def kernel_smoothing(data, optimize=False):
    """Create gaussian kernel.

    The optimization option could lead to longer computation of the PDF.

    :param bool optimize: use global optimization of grid search.
    :return: gaussian kernel.
    :rtype: :class:`sklearn.neighbors.KernelDensity`.
    """
    n_samples, dim = data.shape
    cv = n_samples if n_samples < 50 else 50

    if optimize:
        def bw_score(bw):
            score = cross_val_score(KernelDensity(bandwidth=bw),
                                    data, cv=cv, n_jobs=-1)
            return - score.mean()

        bounds = [(0.1, 5.)]
        results = differential_evolution(bw_score, bounds, maxiter=5)
        bw = results.x
        ks_gaussian = KernelDensity(bandwidth=bw)
        ks_gaussian.fit(data)
    else:
        scott = n_samples ** (-1. / (dim + 4))
        silverman = (n_samples * (dim + 2) / 4.) ** (-1. / (dim + 4))
        bandwidth = np.hstack([np.linspace(0.1, 5.0, 30), scott, silverman])
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidth},
                            cv=cv, n_jobs=-1)  # 20-fold cross-validation
        grid.fit(data)
        ks_gaussian = grid.best_estimator_

    return ks_gaussian


def pdf(data, xdata=None, labels=['x', 'F'], moments=False, fname=None):
    """Plot PDF in 1D or 2D.

    :param np.ndarray/dict data: array of shape (n_samples, n_features)
    or a dictionary with the following::

        - `bounds`, array like of shape (2, n_features) first line is mins and
            second line is maxs.
        - `model`, :class:`batman.surrogate.SurrogateModel` instance or str
            path to the surrogate data.
        - `method`, str, surrogate model method.
        - `dist`, :class:`openturns.ComposedDistribution` instance.

    :param list(str) labels: `x` label and `PDF` label.
    :param bool moments: whether to plot moments along with PDF if dim > 1.
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    dx = 100
    if isinstance(data, dict):
        try:
            f = bat.surrogate.SurrogateModel(data['method'], data['bounds'])
            f.read(data['model'])
        except TypeError:
            f = data['model']
        output_len = len(data['bounds'][0])
        sample = np.array(ot.LHSExperiment(data['dist'], 500).generate())
        z_array, _ = f(sample)
    else:
        z_array = data

    # Compute PDF
    output_len = z_array.shape[1]
    if output_len > 1:
        pdf = []
        ydata = []
        for i in range(output_len):
            ks_gaussian = kernel_smoothing(z_array[:, i].reshape(-1, 1), False)
            xpdf = np.linspace(min(z_array[:, i]), max(z_array[:, i]), dx).reshape(-1, 1)
            pdf.append(np.exp(ks_gaussian.score_samples(xpdf)))
            ydata.append(xpdf.flatten())
        pdf = np.array(pdf).T
        if xdata is None:
            xdata = np.linspace(0, 1, output_len)
        xdata = np.tile(xdata, dx).reshape(-1, output_len)
        ydata = np.array(ydata).T
    else:
        z_array = z_array.reshape(-1, 1)
        ks_gaussian = kernel_smoothing(z_array, True)
        xdata = np.linspace(min(z_array), max(z_array), dx).reshape(-1, 1)
        pdf = np.exp(ks_gaussian.score_samples(xdata))

    # Get moments
    if moments:
        mean = np.mean(z_array, axis=0)
        sd = np.std(z_array, axis=0)
        sd_min = mean - sd
        sd_max = mean + sd
        min_ = np.min(z_array, axis=0)
        max_ = np.max(z_array, axis=0)

    # Plotting
    c_map = cm.viridis
    fig = plt.figure('PDF')
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    if output_len > 1:
        bound_pdf = np.linspace(0., np.max(pdf), 50, endpoint=True)
        plt.contourf(xdata, ydata, pdf, bound_pdf, cmap=c_map, label=None)
        cbar = plt.colorbar()
        cbar.set_label(r"PDF")
        plt.xlabel(labels[0], fontsize=26)
        plt.ylabel(labels[1], fontsize=26)
        if moments:
            plt.plot(xdata[0], sd_min, color='k', ls='-.', linewidth=2, label="Standard Deviation")
            plt.plot(xdata[0], mean, color='k', ls='-', linewidth=2, label="Mean")
            plt.plot(xdata[0], sd_max, color='k', ls='-.', linewidth=2, label=None)
            plt.legend(fontsize=26, loc='best')
    else:
        plt.plot(xdata, pdf, color='k', ls='-', linewidth=3)
        plt.fill_between(xdata[:, 0], pdf, [0] * xdata.shape[0],
                         color='gray', alpha=0.1)
        z_delta = np.max(z_array) * 5e-4
        plt.plot(z_array[:, 0],
                 -z_delta - z_delta * np.random.random(z_array.shape[0]), '+k')
        plt.xlabel(labels[1], fontsize=26)
        plt.ylabel("PDF", fontsize=26)

    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname, transparent=True, bbox_inches='tight')
        # Write PDF to file
        xdata = xdata.flatten('C')
        pdf = pdf.flatten('F')
        names = ['output', 'PDF']
        if output_len > 1:
            ydata = np.array(ydata).flatten('C')
            names = ['x'] + names
            data = np.array([xdata, ydata, pdf])
        else:
            data = np.array([xdata, pdf])

        io = IOFormatSelector('fmt_tp_fortran')
        dataset = Dataset(names=names, data=data)
        io.write(fname.split('.')[0] + '.dat', dataset)

        # Write moments to file
        if moments:
            data = np.append([min_], [sd_min, mean, sd_max, max_])
            names = ["Min", "SD_min", "Mean", "SD_max", "Max"]
            if output_len != 1:
                names = ['x'] + names
                data = np.append(xdata, data)

            dataset = Dataset(names=names, shape=[output_len, 1, 1], data=data)
            io.write(fname.split('.')[0] + '-moment.dat', dataset)
    else:
        plt.show()
    plt.close('all')

    return fig


def sobol(sobols, conf=None, p_lst=None, xdata=None, xlabel='x', fname=None):
    """Plot total Sobol' indices.

    If `len(sobols)>2` map indices are also plotted along with aggregated
    indices.

    :param list(str) p_lst: parameters' names.
    :param array_like sobols: `[first (n_params), total (n_params),
    first (xdata, n_params), total (xdata, n_params)]`.
    :param float/array_like conf: relative error around indices. If float,
    same error is applied for all parameters. Otherwise shape
    ([min, n_features], [max, n_features])
    :param str fname: wether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    p_len = len(sobols[0])
    if p_lst is None:
        p_lst = ["x" + str(i) for i in range(p_len)]
    objects = [[r"$S_{" + p + r"}$", r"$S_{T_{" + p + r"}}$"]
               for i, p in enumerate(p_lst)]
    color = [[cm.Pastel1(i), cm.Pastel1(i)]
             for i, p in enumerate(p_lst)]

    s_lst = [item for sublist in objects for item in sublist]
    color = [item for sublist in color for item in sublist]
    y_pos = np.arange(2 * p_len)

    figures = []
    fig = plt.figure('Aggregated Indices')
    figures.append(fig)
    plt.bar(y_pos, np.array(sobols[:2]).flatten('F'),
            yerr=conf, align='center', alpha=0.5, color=color)
    plt.set_cmap('Pastel2')
    plt.xticks(y_pos, s_lst)
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    plt.ylabel("Sobol' aggregated indices", fontsize=20)
    plt.xlabel("Input parameters", fontsize=20)

    if len(sobols) > 2:
        n_xdata = len(sobols[3])
        if xdata is None:
            xdata = np.linspace(0, 1, n_xdata)
        fig = plt.figure('Sensitivity Map')
        figures.append(fig)
        sobols = np.hstack(sobols[2:]).T
        s_lst = np.array(objects).T.flatten('C').tolist()
        for sobol, label in zip(sobols, s_lst):
            plt.plot(xdata, sobol, linewidth=3, label=label)
        plt.xlabel(xlabel, fontsize=26)
        plt.ylabel(r"Indices", fontsize=26)
        plt.ylim(-0.1, 1.1)
        plt.tick_params(axis='x', labelsize=23)
        plt.tick_params(axis='y', labelsize=23)
        plt.legend(fontsize=26, loc='center right')

    plt.tight_layout()

    if fname is not None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
        for fig in figures:
            pdf.savefig(fig, transparent=True, bbox_inches='tight')
        pdf.close()
    else:
        plt.show()
    plt.close('all')

    return figures
