"""
Uncertainty visualization tools
-------------------------------

It regoups various functions for graph visualizations.

* :func:`kernel_smoothing`,
* :func:`pdf`,
* :func:`sobol`,
* :func:`corr_cov`.
"""
import numpy as np
import openturns as ot
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.optimize import differential_evolution
from matplotlib import cm
import matplotlib.pyplot as plt
import batman as bat
from ..input_output import (IOFormatSelector, Dataset)


def kernel_smoothing(data, optimize=False):
    """Create gaussian kernel.

    The optimization option could lead to longer computation of the PDF.

    :param array_like data: output sample to draw a PDF from
      (n_samples, n_features).
    :param bool optimize: use global optimization of grid search.
    :return: gaussian kernel.
    :rtype: :class:`sklearn.neighbors.KernelDensity`.
    """
    n_samples, dim = data.shape
    cv = n_samples if n_samples < 50 else 50
    var = np.std(data, ddof=1)
    scott = n_samples ** (-1. / (dim + 4)) * var

    if optimize:
        def bw_score(bw):
            """Get the cross validation score for a given bandwidth."""
            score = cross_val_score(KernelDensity(bandwidth=bw),
                                    data, cv=cv, n_jobs=-1)
            return - score.mean()

        bounds = [(0.1 * var, 5. * var)]
        results = differential_evolution(bw_score, bounds, maxiter=5)
        bw = results.x
        ks_gaussian = KernelDensity(bandwidth=bw)
        ks_gaussian.fit(data)
    else:
        silverman = (n_samples * (dim + 2) / 4.) ** (-1. / (dim + 4)) * var
        bandwidth = np.hstack([np.logspace(-1, 1.0, 10) * var,
                               scott, silverman])
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': bandwidth},
                            cv=cv, n_jobs=-1)  # n-fold cross-validation
        grid.fit(data)
        ks_gaussian = grid.best_estimator_

    return ks_gaussian


def pdf(data, xdata=None, xlabel=None, flabel=None, moments=False,
        ticks_nbr=10, range_cbar=None, fname=None):
    """Plot PDF in 1D or 2D.

    :param nd_array/dict data: array of shape (n_samples, n_features)
        or a dictionary with the following:

        - **bounds** (array_like) -- first line is mins and
          second line is maxs (2, n_features).
        - **model** (:class:`batman.surrogate.SurrogateModel`/str) --
          path to the surrogate data.
        - **method** (str) -- surrogate model method.
        - **dist** (:class:`openturns.ComposedDistribution`) --
          joint distribution.

    :param array_like xdata: 1D discretization of the function (n_features,).
    :param str xlabel: label of the discretization parameter.
    :param str flabel: name of the quantity of interest.
    :param bool moments: whether to plot moments along with PDF if dim > 1.
    :param int ticks_nbr: number of color isolines for response surfaces.
    :param array_like range_cbar: Minimum and maximum values for output
      function (2 values).
    :param str fname: whether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    xlabel = 'x' if xlabel is None else xlabel
    flabel = 'F' if flabel is None else flabel

    dx = 100
    if isinstance(data, dict):
        try:
            f = bat.surrogate.SurrogateModel(data['method'], data['bounds'])
            f.read(data['model'])
        except (AttributeError, TypeError):
            f = data['model']
        sample = np.array(ot.LHSExperiment(data['dist'], 500).generate())
        z_array, _ = f(sample)
    else:
        z_array = np.asarray(data)

    # Compute PDF
    output_len = z_array.shape[1]
    if output_len > 1:
        z_array = z_array[:199]  # Reduce the number of sample to use
        pdf = []
        ydata = []
        for i in range(output_len):
            ks_gaussian = kernel_smoothing(z_array[:, i].reshape(-1, 1), False)
            xpdf = np.linspace(min(z_array[:, i]),
                               max(z_array[:, i]), dx).reshape(-1, 1)
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
    fig = plt.figure('PDF')
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    if output_len > 1:
        if range_cbar is None:
            max_pdf = np.percentile(pdf, 97) if np.max(pdf) < 1 else 1
            min_pdf = np.percentile(pdf, 3) if 0 < np.min(pdf) < max_pdf else 0
        else:
            min_pdf, max_pdf = range_cbar
        ticks = np.linspace(min_pdf, max_pdf, num=ticks_nbr)
        bound_pdf = np.linspace(min_pdf, max_pdf, 50, endpoint=True)
        cax = ax.contourf(xdata, ydata, pdf, bound_pdf,
                          cmap=cm.viridis, extend="max")
        cbar = fig.colorbar(cax, shrink=0.5, ticks=ticks)
        cbar.set_label(r"PDF")
        ax.set_xlabel(xlabel, fontsize=26)
        ax.set_ylabel(flabel, fontsize=26)
        if moments:
            ax.plot(xdata[0], sd_min, color='k', ls='-.',
                    linewidth=2, label="Standard Deviation")
            ax.plot(xdata[0], mean, color='k', ls='-', linewidth=2, label="Mean")
            ax.plot(xdata[0], sd_max, color='k', ls='-.', linewidth=2, label=None)
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    else:
        ax.plot(xdata, pdf, color='k', ls='-', linewidth=3, label=None)
        ax.hist(z_array, 30, fc='gray', histtype='stepfilled',
                alpha=0.2, density=True)
        z_delta = np.max(z_array) * 5e-4
        ax.plot(z_array[:199, 0],
                -z_delta - z_delta * np.random.random(z_array[:199].shape[0]),
                '+k', label=None)
        ax.set_xlabel(flabel, fontsize=26)
        ax.set_ylabel("PDF", fontsize=26)

    if fname is not None:
        # Write PDF to file
        xdata_flattened = xdata.flatten('C')
        pdf = pdf.flatten('F')
        names = ['output', 'PDF']
        if output_len > 1:
            ydata = np.array(ydata).flatten('C')
            names = ['x'] + names
            data = np.array([xdata_flattened, ydata, pdf])
        else:
            data = np.array([xdata_flattened, pdf])

        io = IOFormatSelector('fmt_tp_fortran')
        dataset = Dataset(names=names, data=data)
        io.write(fname.split('.')[0] + '.dat', dataset)

        # Write moments to file
        if moments:
            data = np.append([min_], [sd_min, mean, sd_max, max_])
            names = ['Min', 'SD_min', 'Mean', 'SD_max', 'Max']
            if output_len != 1:
                names = ['x'] + names
                data = np.append(xdata[0], data)

            dataset = Dataset(names=names, shape=[output_len, 1, 1], data=data)
            io.write(fname.split('.')[0] + '-moment.dat', dataset)

    bat.visualization.save_show(fname, [fig])

    return fig


def sobol(sobols, conf=None, plabels=None, xdata=None, xlabel='x', fname=None):
    """Plot total Sobol' indices.

    If `len(sobols)>2` map indices are also plotted along with aggregated
    indices.

    :param array_like sobols: `[first (n_params), total (n_params),
        first (xdata, n_params), total (xdata, n_params)]`.
    :param float/array_like conf: relative error around indices. If float,
        same error is applied for all parameters. Otherwise shape
        ([min, n_features], [max, n_features]).
    :param list(str) plabels: parameters' names.
    :param array_like xdata: 1D discretization of the function (n_features,).
    :param str xlabel: label of the discretization parameter.
    :param str fname: wether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    p_len = len(sobols[0])
    if plabels is None:
        plabels = ['x' + str(i) for i in range(p_len)]
    objects = [[r"$S_{" + p + r"}$", r"$S_{T_{" + p + r"}}$"]
               for i, p in enumerate(plabels)]
    color = [[cm.Pastel2(i), cm.Pastel2(i)]
             for i, p in enumerate(plabels)]

    s_lst = [item for sublist in objects for item in sublist]
    color = [item for sublist in color for item in sublist]
    y_pos = np.arange(2 * p_len)

    figures = []
    fig = plt.figure('Aggregated Indices')
    ax = fig.add_subplot(111)
    figures.append(fig)
    ax.bar(y_pos, np.array(sobols[:2]).flatten('F'),
           yerr=conf, align='center', alpha=0.5, color=color)
    ax.set_xticks(y_pos, s_lst)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel("Sobol' aggregated indices", fontsize=20)
    ax.set_xlabel("Input parameters", fontsize=20)

    if len(sobols) > 2:
        n_xdata = len(sobols[3])
        if xdata is None:
            xdata = np.linspace(0, 1, n_xdata)
        fig = plt.figure('Sensitivity Map')
        ax = fig.add_subplot(111)
        figures.append(fig)
        sobols = np.hstack(sobols[2:]).T
        s_lst = np.array(objects).T.flatten('C').tolist()
        for sobol, label in zip(sobols, s_lst):
            ax.plot(xdata, sobol, linewidth=3, label=label)
        ax.set_xlabel(xlabel, fontsize=26)
        ax.set_ylabel(r"Indices", fontsize=26)
        ax.set_ylim(-0.1, 1.1)
        ax.tick_params(axis='x', labelsize=23)
        ax.tick_params(axis='y', labelsize=23)
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    bat.visualization.save_show(fname, figures)

    return figures


def corr_cov(data, sample, xdata, xlabel='x', plabels=None, interpolation=None,
             fname=None):
    """Correlation and covariance matrices.

    Compute the covariance regarding YY and XY as well as the correlation
    regarding YY.

    :param array_like data: function evaluations (n_samples, n_features).
    :param array_like sample: sample (n_samples, n_featrues).
    :param array_like xdata: 1D discretization of the function (n_features,).
    :param str xlabel: label of the discretization parameter.
    :param list(str) plabels: parameters' labels.
    :param str interpolation: If None, does not interpolate correlation and
        covariance matrices (YY). Otherwize use Matplotlib methods from
        `imshow` such as `['bilinear', 'lanczos', 'spline16', 'hermite', ...]`.
    :param str fname: wether to export to filename or display the figures.
    :returns: figure.
    :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
    """
    p_len = np.asarray(sample).shape[1]
    data_len = np.asarray(data).shape[1]
    data = ot.Sample(data)

    corr_yy = np.array(data.computePearsonCorrelation())
    cov_yy = np.array(data.computeCovariance())
    cov_matrix_xy = np.dot((np.mean(sample) - sample).T,
                           np.mean(data, axis=0) - data) / (len(sample) - 1)

    x_2d_yy, y_2d_yy = np.meshgrid(xdata, xdata)
    x_2d_xy, y_2d_xy = np.meshgrid(xdata, np.arange(p_len))

    c_map = cm.viridis

    figures, axs = [], []

    # Covariance matrix YY
    fig, ax = plt.subplots()
    figures.append(fig)
    axs.append(ax)
    cax = ax.imshow(cov_yy, cmap=c_map, interpolation=interpolation, origin='lower')
    cbar = fig.colorbar(cax)
    cbar.set_label(r"Covariance", size=26)
    cbar.ax.tick_params(labelsize=23)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(xlabel, fontsize=26)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)

    # Correlation matrix YY
    fig, ax = plt.subplots()
    figures.append(fig)
    cax = ax.imshow(corr_yy, cmap=c_map, interpolation=interpolation, origin='lower')
    cbar = fig.colorbar(cax)
    cbar.set_label(r"Correlation", size=26)
    cbar.ax.tick_params(labelsize=23)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel(xlabel, fontsize=26)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)

    if plabels is None:
        plabels = ['x' + str(i) for i in range(p_len + 1)]
    else:
        plabels.insert(0, 0)

    # Covariance matrix XY
    fig, ax = plt.subplots()
    figures.append(fig)
    axs.append(ax)
    cax = ax.imshow(cov_matrix_xy, cmap=c_map, interpolation='nearest')
    ax.set_yticklabels(plabels, fontsize=6)
    cbar = fig.colorbar(cax)
    cbar.set_label(r"Covariance", size=26)
    cbar.ax.tick_params(labelsize=23)
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_ylabel('Input parameters', fontsize=26)
    ax.tick_params(axis='x', labelsize=23)
    ax.tick_params(axis='y', labelsize=23)

    if fname is not None:
        data = np.append(x_2d_yy, [y_2d_yy, corr_yy, cov_yy])
        dataset = Dataset(names=['x', 'y', 'Correlation-YY', 'Covariance'],
                          shape=[data_len, data_len, 1],
                          data=data)
        io = IOFormatSelector('fmt_tp_fortran')
        io.write(fname.split('.')[0] + '-correlation_covariance.dat', dataset)

        data = np.append(x_2d_xy, [y_2d_xy, cov_matrix_xy])
        dataset = Dataset(names=['x', 'y', 'Correlation-XY'],
                          shape=[p_len, data_len, 1],
                          data=data)
        io.write(fname.split('.')[0] + '-correlation_XY.dat', dataset)

    bat.visualization.save_show(fname, figures)

    return figures, axs
