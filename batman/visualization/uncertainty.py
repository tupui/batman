"""
Uncertainty visualization tools
-------------------------------
"""
import numpy as np
import re
import os
import itertools
import openturns as ot
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import differential_evolution

from ..input_output import (IOFormatSelector, Dataset)
import batman as bat

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


def kernel_smoothing(data, optimize=False):
    """Create gaussian kernel.

    :param bool optimize: use global optimization of grid search
    :return: gaussian kernel
    :rtype: :class:`sklearn.neighbors.KernelDensity`
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


def pdf(data, xdata=None, labels=['x', 'F'], fname=None):
    """Plot PDF in 1D or 2D.

    :param np.ndarray/dict data: 1D array of shape (n_sample, n_feature)
    or a dictionary with the following::

        - `bounds`, array like of shape (2, n_feature) first line is mins and
            second line is maxs.
        - `model`, :class:`batman.surrogate.SurrogateModel` instance or str
            path to the surrogate data.
        - `dist`, :class:`openturns.ComposedDistribution` instance.

    :param list(str) labels: `x` label and `PDF` label
    :param str fname: wether to export to filename or display the figures
    """
    dx = 100
    if isinstance(data, dict):
        try:
            f = bat.surrogate.SurrogateModel('kriging', data['bounds'])
            f.read(data['model'])
        except TypeError:
            f = data['model']
        output_len = len(data['bounds'][0])
        sample = np.array(ot.LHSExperiment(data['dist'], 500).generate())
        z_array, _ = f(sample)
    else:
        z_array = data

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

    # Plotting
    c_map = cm.viridis
    if output_len > 1:
        fig = plt.figure('PDF')
        bound_pdf = np.linspace(0., np.max(pdf), 50, endpoint=True)
        plt.contourf(xdata, ydata, pdf, bound_pdf, cmap=c_map)
        cbar = plt.colorbar()
        cbar.set_label(r"PDF")
        plt.xlabel(labels[0], fontsize=26)
        plt.ylabel(labels[1], fontsize=26)
        plt.tick_params(axis='x', labelsize=26)
        plt.tick_params(axis='y', labelsize=26)
    else:
        fig = plt.figure('PDF')
        plt.plot(xdata, pdf, color='k', ls='-', linewidth=3)
        plt.fill_between(xdata[:, 0], pdf, [0] * xdata.shape[0],
                         color='gray', alpha=0.1)
        z_delta = np.max(z_array) * 5e-4
        plt.plot(z_array[:, 0],
                 -z_delta - z_delta * np.random.random(z_array.shape[0]), '+k')
        plt.xlabel(labels[1], fontsize=26)
        plt.ylabel("PDF", fontsize=26)
        plt.tick_params(axis='x', labelsize=26)
        plt.tick_params(axis='y', labelsize=26)
        plt.legend(fontsize=26, loc='upper right')

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
        io.write(fname + '.dat', dataset)
    else:
        plt.show()
    plt.close('all')

    return fig
