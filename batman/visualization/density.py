"""
Density-based measures
----------------------

This module is based on code initially written by Irene Witte (University of
Hohenheim).

:Example:

::

    >> from batman.space import Space
    >> from batman.functions import Ishigami
    >> f = Ishigami()
    >> sample = Space(corners=[[-3.14, -3.14, -3.14], [3.14, 3.14, 3.14]],
    >>               sample=1000)
    >> sample.sampling()
    >> data = f(sample)
    >>
    >> cusunoro(sample, data)
    >> moment_independent(sample, data)
"""
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import matplotlib.pyplot as plt
import batman as bat


def cusunoro(sample, data, plabels=None, fname=None):
    """Cumulative sums of normalised reordered output.

    1. Data are normalized (mean=0, variance=1),
    2. Choose a feature and order its values,
    3. Order normalized data accordingly,
    4. Compute the cumulative sum vector.
    5. Plot and repeat for all features.

    :param array_like sample: Sample of parameters of Shape
      (n_samples, n_params).
    :param array_like data: Sample of realization which corresponds to the
      sample of parameters :attr:`sample` (n_samples, ).
    :param list(str) plabels: Names of each parameters (n_features).
    :param str fname: whether to export to filename or display the figures.
    :returns: figure, axis and sensitivity indices.
    :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instance,
      array_like.
    """
    sample = np.asarray(sample)
    data = np.asarray(data).flatten()

    ns, dim = sample.shape
    if plabels is None:
        plabels = ['x' + str(i) for i in range(dim)]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Normalization (mean=0, var=1) for first and second moment
    diff_y = (data - data.mean())
    w = diff_y ** 2
    ynorm = (diff_y) / (ns * np.std(data))
    ynorm2 = (w - w.mean()) / (ns * np.std(w))

    s_indices = np.zeros(dim)

    interval = int(ns / 500) if ns > 500 else 1
    n_bins = int(np.ceil(np.sqrt(ns)))  # from Plischke

    for i in range(dim):
        # Reordering and cumulative sum
        idx = sample[:, i].argsort()
        cumsum = np.cumsum(ynorm[idx])
        cumsum2 = np.cumsum(ynorm2[idx])

        # Estimation of the first order effect from the cusunoro curve
        s_indices[i] = sum(np.diff(cumsum[::n_bins]) ** 2) * n_bins
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


def ecdf(data):
    """Empirical Cumulative Distribution.

    :param array_like data: Sample of realization which corresponds to the
          sample of parameters :attr:`sample` (n_samples, n_features).
    :returns: xs (ordered data), ys (CDF(xs)).
    :rtypes: array_like (n_samples, n_features), (n_samples,)
    """
    xs = np.sort(data)
    ys = np.linspace(0, 1, num=len(data))
    return xs, ys


def moment_independent(sample, data, plabels=None, scale_plt=True, fname=None):
    """Moment independent measures.

    Use both PDF and ECDF to cumpute moment independent measures. The following
    algorithm describes the PDF method (ECDF works the same):

    1. Compute the unconditional PDF,
    2. Choose a feature and order its values and order the data accordingly,
    3. Create bins based on the feature ranges,
    4. Compute the PDF of the ordered data on all successive bins,
    5. Plot and repeat for all features.

    :param array_like sample: Sample of parameters of Shape
      (n_samples, n_params).
    :param array_like data: Sample of realization which corresponds to the
      sample of parameters :attr:`sample` (n_samples, ).
    :param list(str) plabels: Names of each parameters (n_features).
    :param bool scale_plt: Whether to scale y-axes between figures.
    :param str fname: Whether to export to filename or display the figures.
    :returns: Figure, axis and sensitivity indices.
    :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instances,
      dict(['Kolmogorov', 'Kuiper', 'Delta', 'Sobol'], n_features).
    """
    sample = np.asarray(sample)
    data = np.asarray(data).flatten()
    var_t = np.var(data)
    mean_t = np.mean(data)

    ns, dim = sample.shape
    ns = float(ns)
    if plabels is None:
        plabels = ['x' + str(i) for i in range(dim)]
    else:
        plabels = plabels

    s_indices = {'Kolmogorov': [], 'Kuiper': [], 'Delta': [], 'Cramer': [], 'Sobol': []}
    n_parts = int(min(np.ceil(ns ** (2 / (7 + np.tanh((1500 - ns) / 500)))), 48))
    len_part = ns / n_parts

    # Unconditional PDF
    xs = np.linspace(np.min(data), np.max(data), 100)
    pdf_u = gaussian_kde(data, bw_method="silverman")(xs)

    # Unconditional ECDF
    ecdf_u = ecdf(data)

    fig, axs = [], []
    pdf_max = 0

    for d in range(dim):
        fig_, axs_ = plt.subplots(2, 1)
        fig.append(fig_)
        axs.append(axs_)

        # Sensitivity indices
        ks = []
        kui = []
        delta = 0
        cramer = 0
        var_d = 0

        # Data reordering
        idx = sample[:, d].argsort()
        data_r = data[idx]

        for i in range(n_parts):
            # Conditional PDF
            data_ = data_r[int(i * len_part):int((i + 1) * len_part)]
            pdf_c = gaussian_kde(data_, bw_method="silverman")(xs)
            axs_[0].plot(xs, pdf_c, alpha=.3)

            # sup-ylim for plotting
            max_ = max(pdf_c)
            pdf_max = max_ if max_ > pdf_max else pdf_max

            # Conditional ECDF
            ecdf_c = ecdf(data_)
            axs_[1].plot(ecdf_c[0], ecdf_c[1], alpha=.3)

            # Metrics
            data_all = np.concatenate([ecdf_u[0], ecdf_c[0]])
            cdf1 = np.searchsorted(ecdf_u[0], data_all, side='right') / ns
            cdf2 = np.searchsorted(ecdf_c[0], data_all, side='right') / len_part
            cdf_diff = cdf1 - cdf2

            ks.append(np.max(np.absolute(cdf_diff)))
            kui.append(np.max(cdf_diff) - np.min(cdf_diff))
            delta += (len_part / (2 * ns)) * simps(np.abs(pdf_u - pdf_c), xs)
            var_d += (len_part / ns) * (data_.mean() - mean_t) ** 2

            xs_cdf = np.linspace(0, 1, len(cdf_diff))
            cramer += (len_part / ns) * simps((cdf_diff) ** 2, xs_cdf)\
                / np.trapz((cdf1 * (1 - cdf1)), xs_cdf)

        # Metrics
        s_indices['Kolmogorov'].append(np.mean(ks))
        s_indices['Kuiper'].append(np.mean(kui))
        s_indices['Delta'].append(delta)
        s_indices['Cramer'].append(cramer)
        s_indices['Sobol'].append(var_d / var_t)

        # Plots
        axs_[0].plot(xs, pdf_u, c='k', linewidth=2)
        axs_[0].set_ylabel('PDF')

        axs_[1].plot(ecdf_u[0], ecdf_u[1], c='k', linewidth=2)
        axs_[1].set_ylabel('CDF')

        axs_[1].set_xlabel('Y|' + plabels[d])

    if scale_plt:
        for i in range(dim):
            axs[i][0].set_ylim([0, 1.05 * pdf_max])
            axs[i][1].set_ylim([0, 1])

    bat.visualization.save_show(fname, fig)

    return fig, axs, s_indices
