"""
High Density Region Boxplot
---------------------------
"""
import logging
from itertools import (combinations_with_replacement, compress)
from multiprocessing import Pool
import numpy as np
from scipy.optimize import differential_evolution
from scipy.io import wavfile
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from .uncertainty import kernel_smoothing
from .doe import doe

import matplotlib.animation as manimation
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


class HdrBoxplot:

    """High Density Region boxplot.

    From a given dataset, it computes the HDR-boxplot. Results are accessibles
    directly through class attributes:

    - :attr:`median` : median curve,
    - :attr:`outliers` : outliers regarding a given threshold,
    - :attr:`hdr_90` : 90% quantile band,
    - :attr:`extra_quantiles` : other quantile bands,
    - :attr:`hdr_50` : 50% quantile band.

    The following methods are for convenience:

    - :func:`HdrBoxplot.plot`
    - :func:`HdrBoxplot.f_hops`
    - :func:`HdrBoxplot.sound`

    :Example:

    ::

        >> hdr = HdrBoxplot(data)
        >> hdr.plot()
        >> hdr.f_hops(generate=10)
        >> hdr.sound()

    """

    logger = logging.getLogger(__name__)

    def __init__(self, data, variance=0.8, alpha=None,
                 threshold=0.95, outliers_method='kde', optimize=False):
        """Compute HDR Boxplot on :attr:`data`.

        1. Compute a 2D kernel smoothing with a Gaussian kernel,
        2. Compute contour lines for quantiles 90, 50 and :attr:`alpha`,
        3. Compute mediane curve along with quantiles regions and outlier
        curves.

        :param array_like data: dataset (n_samples, n_features)
        :param float variance: percentage of total variance to conserve
        :param array_like alpha: extra quantile values (n_alpha)
        :param float threshold: threshold for outliers
        :param str outliers_method: detection method ['kde', 'forest']
        :param bool optimize: bandwidth global optimization or grid search
        :param int n_contours: discretization to compute contour
        """
        self.data = data
        self.threshold = threshold
        self.outliers_method = outliers_method
        self.optimize = optimize
        self.n_samples, self.dim = self.data.shape
        self.logger.info('Dataset with:\n-> {} samples\n-> {} features'
                         .format(self.n_samples, self.dim))
        # PCA and bivariate plot
        self.pca = PCA(n_components=variance, svd_solver='full')
        self.data_r = self.pca.fit_transform(self.data)
        self.n_components = len(self.pca.explained_variance_ratio_)

        self.logger.info('Explained variance ratio: {} -> {:0.3f}'
                         .format(self.pca.explained_variance_ratio_,
                                 np.sum(self.pca.explained_variance_ratio_)))

        # Create gaussian kernel
        self.ks_gaussian = kernel_smoothing(self.data_r, self.optimize)

        # Boundaries of the n-variate space
        self.bounds = np.array([self.data_r.min(axis=0), self.data_r.max(axis=0)]).T

        # Create list of quantile values
        if alpha is None:
            alpha = [threshold, 0.9, 0.5]
        else:
            alpha.extend([threshold, 0.9, 0.5])
            alpha = list(set(alpha))
        alpha.sort(reverse=True)
        self.alpha = alpha
        self.logger.debug('alpha: {}'.format(self.alpha))
        self.n_alpha = len(self.alpha)

        # Compute PDF values associated to each quantile
        self.pdf_r = np.exp(self.ks_gaussian.score_samples(self.data_r)).flatten()
        self.pvalues = np.array([np.percentile(self.pdf_r, (1 - self.alpha[i]) * 100,
                                               interpolation='linear')
                                 for i in range(self.n_alpha)])

        self.logger.debug('pvalues: {}'.format(self.pvalues))

        def pdf(x):
            """Compute -PDF given components."""
            return - np.exp(self.ks_gaussian.score_samples(x.reshape(1, -1)))

        # Find mean, quantiles and outliers curves
        median = differential_evolution(pdf, bounds=self.bounds, maxiter=5).x

        self.outliers = self.find_outliers(data=self.data, samples=self.pdf_r,
                                           method=self.outliers_method,
                                           threshold=self.threshold)

        extra_alpha = [i for i in self.alpha
                       if 0.5 != i and 0.9 != i and threshold != i]
        if extra_alpha != []:
            self.extra_quantiles = [y for x in extra_alpha
                                    for y in self.band_quantiles([x])]
        else:
            self.extra_quantiles = []

        # Inverse transform from n-variate plot to original dataset's shape
        self.median = self.pca.inverse_transform(median)
        self.hdr_90 = self.band_quantiles([0.9, 0.5])
        self.hdr_50 = self.band_quantiles([0.5])

    def band_quantiles(self, band):
        """Find extreme curves for a quantile band.

        From the :attr:`band` of quantiles, the associated PDF extrema values
        are computed. If `min_alpha` is not provided (single quantile value),
        `max_pdf` is set to `1E6` in order not to constrain the problem on high
        values.

        An optimization is performed per component in order to find the min and
        max curves. This is done by comparing the PDF value of a given curve
        with the band PDF.

        :param array_like band: alpha values `[max_alpha, min_alpha]` ex: [0.9, 0.5]
        :return: `[max_quantile, min_quantile]` (2, n_features)
        :rtype: list(array_like)
        """
        min_pdf = self.pvalues[self.alpha.index(band[0])]
        try:
            max_pdf = self.pvalues[self.alpha.index(band[1])]
        except IndexError:
            max_pdf = 1E6
        self.band = [min_pdf, max_pdf]

        with Pool() as pool:
            band_quantiles = pool.map(self._min_max_band, range(self.dim))

        band_quantiles = list(zip(*band_quantiles))

        return band_quantiles

    def _curve_constrain(self, x, idx, sign):
        """Find out if the curve is within the band.

        The curve value at :attr:`idx` for a given PDF is only returned if
        within bounds defined by the band. Otherwise, 1E6 is returned.

        :param float x: curve in reduced space
        :param int idx: index value of the components to compute
        :param int sign: return positive or negative value
        :return: Curve value at :attr:`idx`
        :rtype: float
        """
        x = x.reshape(1, -1)
        pdf = np.exp(self.ks_gaussian.score_samples(x))
        if self.band[0] < pdf < self.band[1]:
            value = sign * self.pca.inverse_transform(x)[0][idx]
        else:
            value = 1E6
        return value

    def _min_max_band(self, idx):
        """Min an max values at :attr:`idx`.

        Global optimization to find the extrema per component.

        :param int idx: curve index
        :returns: [max, min] curve values at :attr:`idx`
        :rtype: tuple(float)
        """
        max_ = differential_evolution(self._curve_constrain, bounds=self.bounds,
                                      args=(idx, -1),
                                      maxiter=7)
        min_ = differential_evolution(self._curve_constrain, bounds=self.bounds,
                                      args=(idx, 1),
                                      maxiter=7)
        return (self.pca.inverse_transform(max_.x)[idx],
                self.pca.inverse_transform(min_.x)[idx])

    def find_outliers(self, data, samples, method='kde', threshold=0.95):
        """Detect outliers.

        The *Isolation forrest* method requires additional computations to find
        the centroide. This operation is only performed once and stored in
        :attr:`self.detector`. Thus calling, several times the method will not
        cause any overhead.

        :param array_like, shape (n_samples, n_features) data: data from which
        to extract outliers
        :param array_like, shape (n_samples, n_features/n_components) samples:
        samples values to examine
        :param str method: detection method ['kde', 'forest']
        :param float threshold: detection sensitivity
        """
        if method == 'kde':
            outliers = np.where(samples < self.pvalues[self.alpha.index(threshold)])
            outliers = data[outliers]
        elif method == 'forest':
            try:
                try:
                    data_r = self.pca.transform(data)
                except ValueError:
                    data_r = data
                outliers = np.where(self.detector.predict(data_r) == -1)
            except AttributeError:
                forrest = IsolationForest(contamination=(1 - threshold),
                                          n_jobs=-1)
                self.detector = forrest.fit(self.data_r)
                outliers = np.where(self.detector.predict(data_r) == -1)
            outliers = data[outliers]
        else:
            self.logger.error('Unknown outlier method: no detection')
            outliers = None

        return outliers

    def plot(self, samples=None, fname=None, x_common=None, labels=None,
             xlabel='t', ylabel='y'):
        """Functional plot and n-variate space.

        If :attr:`self.n_components` is 2, an additional contour plot is done.
        If :attr:`samples` is `None`, the dataset is used for all plots ;
        otherwize the given sample is used.

        :param array_like, shape (n_samples, n_features): samples to plot
        :param str fname: wether to export to filename or display the figures
        :param array_like, shape (1, n_features) x_common: abscissa
        :param list(str) labels: labels for each curve
        :param str xlabel: label for x axis
        :param str ylabel: label for y axis
        :returns: figures and all axis
        :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances
        """
        figures = []
        axs = []

        if samples is None:
            data = self.data
            data_r = self.data_r
            n_samples = self.n_samples
        elif isinstance(samples, int):
            data_r = self.ks_gaussian.sample(n_samples=samples)
            data = self.pca.inverse_transform(data_r)
            n_samples = len(data)
        else:
            data = samples
            data_r = self.pca.transform(data)
            n_samples = len(data)

        if self.n_components == 2:
            n_contours = 50
            grid = np.meshgrid(*[np.linspace(*self.bounds[i], n_contours)
                                 for i in range(self.n_components)])
            stack = np.dstack(grid).reshape(-1, self.n_components)
            pdf = np.exp(self.ks_gaussian.score_samples(stack)).flatten()

            # 2D Kernel Smoothing with Gaussian kernel
            fig, ax = plt.subplots()
            figures.append(fig)
            axs.append(ax)
            contour = plt.contour(*grid,
                                  pdf.reshape((n_contours, n_contours)),
                                  self.pvalues)
            # Labels: probability instead of density
            fmt = {}
            for i in range(self.n_alpha):
                lev = contour.levels[i]
                fmt[lev] = "%.0f %%" % (self.alpha[i] * 100)
            plt.clabel(contour, contour.levels,
                       inline=True, fontsize=10, fmt=fmt)
            plt.tight_layout()

        # Bivariate space
        fig, sub_ax = doe(data_r,
                          p_lst=[str(i + 1) for i in range(self.n_components)])
        figures.append(fig)
        axs.append(sub_ax)

        # Time serie
        fig, ax = plt.subplots()
        figures.append(fig)
        axs.append(ax)
        if x_common is None:
            x_common = np.linspace(0, 1, self.dim)
        plt.plot(np.array([x_common] * n_samples).T, data.T,
                 c='c', alpha=.1, label='dataset')
        plt.plot(x_common, self.median, c='k', label='Median')
        plt.fill_between(x_common, *self.hdr_50,
                         color='gray', alpha=.4,  label='50% HDR')
        plt.fill_between(x_common, *self.hdr_90,
                         color='gray', alpha=.3, label='90% HDR')

        if len(self.extra_quantiles) != 0:
            plt.plot(np.array([x_common] * len(self.extra_quantiles)).T,
                     np.array(self.extra_quantiles).T,
                     c='y', ls='-.', alpha=.4, label='Extra quantiles')

        if len(self.outliers) != 0:
            if labels is not None:
                labels_pos = np.all(np.isin(self.data, self.outliers), axis=1)
                labels = list(compress(labels, labels_pos))

            for ii, outlier in enumerate(self.outliers):
                label = str(labels[ii]) if labels is not None else 'Outliers'
                plt.plot(x_common, outlier,
                         ls='--', alpha=0.7, label=label)
        else:
            self.logger.debug('It seems that there are no outliers...')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')

        plt.tight_layout()

        if fname is not None:
            pdf = matplotlib.backends.backend_pdf.PdfPages(fname)
            for fig in figures:
                pdf.savefig(fig, transparent=True, bbox_inches='tight')
            pdf.close()
        else:
            plt.show()
        plt.close('all')

        return figures, axs

    def f_hops(self, frame_rate=400, fname='f-HOPs.mp4', samples=None,
               x_common=None, labels=None, xlabel='t', ylabel='y', offset=0.05):
        """Functional Hypothetical Outcome Plots.

        Each frame consists in a HDR boxplot and an additional outcome.
        If it is an outlier, it is rendered as red dashed line.

        If :attr:`samples` is `None` it will use the dataset, if an `int>0`
        it will samples *n* new samples ; and if
        `array_like, shape (n_samples, n_features)` it will use this.

        :param int frame_rate: time between two outcomes (in milliseconds)
        :param str fname: export movie to filename
        :param False, int, list samples: Data selector
        :param array_like x_common: abscissa
        :param list(str) labels: labels for each curve
        :param str xlabel: label for x axis
        :param str ylabel: label for y axis
        :param float offset: Margin around the extreme values of the plot
        """
        movie_writer = manimation.writers['ffmpeg']
        metadata = {'title': 'f-HOPs',
                    'artist': 'batman',
                    'comment': "Functional Hypothetical Outcome Plots at {} ms"
                              .format(frame_rate)}

        writer = movie_writer(fps=1000 / frame_rate, metadata=metadata)

        fig = plt.figure()
        if x_common is None:
            x_common = np.linspace(0, 1, self.dim)
        plt.fill_between(x_common, *self.hdr_50,
                         color='gray', alpha=.4, label='50% HDR')
        plt.fill_between(x_common, *self.hdr_90,
                         color='gray', alpha=.3, label='90% HDR')
        plt.plot(x_common, self.median, c='k', label='Median')

        y_min = min(self.hdr_90[1])
        y_max = max(self.hdr_90[0])
        plt.ylim(y_min - abs(y_min) * offset, y_max + abs(y_max) * offset)

        frame, = plt.plot([], [], c='c', ls='-')
        frame_outliers, = plt.plot([], [], c='r', ls='--')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if samples is None:
            data_r = self.data_r
            pdf_r = self.pdf_r
            data = self.data
        elif isinstance(samples, int):
            data_r = self.ks_gaussian.sample(n_samples=samples)
            data = self.pca.inverse_transform(data_r)
            pdf_r = np.exp(self.ks_gaussian.score_samples(data_r))
        else:
            data = samples
            data_r = self.pca.transform(data)
            pdf_r = np.exp(self.ks_gaussian.score_samples(data_r))

        with writer.saving(fig, fname, dpi=200):
            for i, (pdf, curve_r, curve) in enumerate(zip(pdf_r, data_r, data)):
                curve_r = np.atleast_2d(curve_r)
                outliers = self.find_outliers(data=curve_r, samples=pdf,
                                              method=self.outliers_method,
                                              threshold=self.threshold)
                outliers = self.pca.inverse_transform(outliers)

                if len(outliers) == 0:
                    label = 'HOP: ' + str(labels[i]) \
                        if labels is not None else 'HOP'
                    frame.set_data(x_common, curve)
                    frame.set_label(label)
                    frame_outliers.set_data([], [])
                    frame_outliers.set_label(None)
                else:
                    label = 'HOP Outlier: ' + str(labels[i]) \
                        if labels is not None else 'HOP Outlier'
                    frame_outliers.set_data(x_common, outliers)
                    frame_outliers.set_label(label)
                    frame.set_data([], [])
                    frame.set_label(None)

                handles, labels_ = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels_, handles))
                plt.legend(by_label.values(), by_label.keys(), loc='best')

                writer.grab_frame()

    def sound(self, frame_rate=400, tone_range=[50, 1000], amplitude=1E3,
              distance=True, samples=False, fname='song-fHOPs.wav'):
        """Make sound from curves.

        Each curve is converted into a sum of tones. This sum is played during
        a given time before another serie starts.

        If :attr:`samples` is `False` it will use the dataset, if an `int>0`
        it will samples *n* new samples ; and if
        `array_like, shape (n_samples, n_features)` it will use this.

        :param int frame_rate: time between two outcomes (in milliseconds)
        :param list(int) tone_range: range of frequencies of a tone (in hertz)
        :param float amplitude: amplitude of the signal
        :param bool distance: use distance from median for tone generation
        :param False, int, list samples: Data selector
        :param str fname: export sound to filename
        """
        duration = frame_rate / 1000
        amp = amplitude
        rate = 44100
        t = np.linspace(0, duration, duration * rate)

        def note(freq):
            data = np.sin(2 * np.pi * freq * t) * amp
            return data

        scaler = MinMaxScaler(feature_range=tone_range)

        if isinstance(samples, bool):
            data = self.data
        elif isinstance(samples, int):
            data = self.sample(samples)
        else:
            data = samples

        if distance:
            centroide = self.pca.transform(self.median.reshape(1, -1))
            dists = cdist(centroide, self.pca.transform(data))[0]
            dists = scaler.fit_transform(dists.reshape(-1, 1))
            song = [np.array(note(d)) for d in dists]
        else:
            data = scaler.fit_transform(data)

            song = [np.sum([note(tone) for tone in curve],
                           axis=0) for curve in data]

        # two byte integers conversion
        wavfile.write(fname, rate,
                      np.array(song).astype(np.int16).flatten('C'))

    def sample(self, samples):
        """Sample new curves from KDE.

        If :attr:`samples` is an `int>0`, *n* new curves are randomly sampled
        taking into account the joined PDF ; and if
        `array_like, shape (n_samples, n_components)` curves are sampled
        from reduce coordinates of the n-variate space.

        :param int, array_like samples: Data selector
        :return: new curves
        :rtype: array_like, shape (n_samples, n_features)
        """
        if isinstance(samples, int):
            data = self.ks_gaussian.sample(n_samples=samples)
        else:
            data = samples

        curves = self.pca.inverse_transform(data)

        return curves