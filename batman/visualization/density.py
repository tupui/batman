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


def moment_independent(sample, data, plabels=None, fname=None):
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
    :param str fname: whether to export to filename or display the figures.
    :returns: figure, axis and sensitivity indices.
    :rtype: Matplotlib figure instance, Matplotlib AxesSubplot instances,
      dict(['Kolmogorov', 'Kuiper', 'Delta', 'Sobol'], n_features).
    """
    sample = np.asarray(sample)
    data = np.asarray(data).flatten()
    var_t = np.var(data)
    mean_t = np.mean(data)

    ns, dim = sample.shape
    if plabels is None:
        plabels = ['x' + str(i) for i in range(dim)]
    else:
        plabels = plabels

    s_indices = {'Kolmogorov': [], 'Kuiper': [], 'Delta': [], 'Sobol': []}
    n_parts = int(min(np.ceil(ns ** (2 / (7 + np.tanh((1500 - ns) / 500)))), 48))
    len_part = ns / n_parts

    # Unconditional PDF
    xs = np.linspace(np.min(data), np.max(data), 100)
    pdf_u = gaussian_kde(data, bw_method="silverman")(xs)

    # Unconditional ECDF
    ecdf_u = ecdf(data)

    fig, axs = plt.subplots(2, dim)

    for d in range(dim):
        # Sensitivity indices
        ks = []
        kui = []
        d_hat = 0
        var_d = 0

        # Data reordering
        idx = sample[:, d].argsort()
        data_r = data[idx]

        for i in range(n_parts):
            # Conditional PDF
            data_ = data_r[int(i * len_part):int((i + 1) * len_part)]
            pdf_c = gaussian_kde(data_, bw_method="silverman")(xs)
            axs[0][d].plot(xs, pdf_c, alpha=.3)

            # Conditional ECDF
            ecdf_c = ecdf(data_)
            axs[1][d].plot(ecdf_c[0], ecdf_c[1], alpha=.3)

            # Metrics
            data_all = np.concatenate([ecdf_u[0], ecdf_c[0]])
            cdf1 = np.searchsorted(ecdf_u[0], data_all, side='right') / ns
            cdf2 = np.searchsorted(ecdf_c[0], data_all, side='right') / len_part

            ks_ = np.max(np.absolute(cdf1 - cdf2))
            kui_ = np.max(cdf1 - cdf2) - np.min(cdf1 - cdf2)

            ks.append(ks_)
            kui.append(kui_)
            d_hat += (len_part / (2 * ns)) * np.trapz(np.abs(pdf_u - pdf_c), xs)
            var_d += (len_part / ns) * (data_.mean() - mean_t) ** 2

        s_indices['Kolmogorov'].append(np.mean(ks))
        s_indices['Kuiper'].append(np.mean(kui))
        s_indices['Delta'].append(d_hat)
        s_indices['Sobol'].append(var_d / var_t)

        axs[0][d].plot(xs, pdf_u, c='k', linewidth=2)

        axs[1][d].plot(ecdf_u[0], ecdf_u[1], c='k', linewidth=2)
        axs[1][d].set_xlabel('Y|' + plabels[d])

        if d == 0:
            axs[0][d].set_ylabel('PDF')
            axs[1][d].set_ylabel('CDF')

    bat.visualization.save_show(fname, [fig])

    return fig, axs, s_indices
