# coding: utf8
"""
Mixture Class
=============

Mixture of expert using clustering machine learning to form local surrogate
models.

:Example:

::

    >> from batman.surrogate import Mixture
    >> import numpy as np
    >> samples = np.array([[1., 5.], [2., 5.], [8., 5.], [9., 5.]])
    >> data = np.array([[50., 51., 52.], [49., 48., 47.], [10., 11., 12,],
       [9., 8., 7.]])
    >> sample_new = np.array([[0.5, 5.],[10., 5.],[8.5, 5.]])
    >> plabels = ['x1', 'x2']
    >> corners = np.array([[1., 5.], [9., 5.]])
    >> fsizes = 3
    >> algo = Mixture(samples, data, plabels, corners, fsizes)
    >> algo.evaluate(sample_new)
    (array([[30.196, 31.196, 32.196],
       [29.   , 28.   , 27.   ],
       [28.804, 27.804, 26.804]]), array([[19.999, 19.999, 19.999],
       [20.   , 20.   , 20.   ],
       [19.999, 19.999, 19.999]]))
    >> algo.estimate_quality()
     1.0, array([0., 0.])
"""
import logging
import numpy as np
from sklearn.decomposition import PCA
import sklearn.mixture
import sklearn.cluster
import sklearn.gaussian_process
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn import preprocessing
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import batman as bat
from batman.visualization import Kiviat3D
from ..space import Sample


class Mixture:
    """Mixture class.

    Unsupervised machine learning separate the DoE into clusters, supervised
    machine learning classify new sample to a cluster and local models
    predict the new sample."""

    logger = logging.getLogger(__name__)

    def __init__(self, samples, data, corners, fsizes=None, pod=None,
                 standard=True, local_method=None, pca_percentage=0.8,
                 clusterer='cluster.KMeans(n_clusters=2)',
                 classifier='gaussian_process.GaussianProcessClassifier()'):
        """Cluster data and fit local models.

        1. If :attr:`data` is not scalar, compute PCA on :attr:`data`.
        2. Cluster data.
        3. Each sample is affiliated to a cluster.
        4. Fit a classifier to handle new samples.
        5. A local model for each cluster is built.

        If :attr:`local_method` is not None, set as list of dict with options.
        Ex: `[{'kriging': {**args}}]`

        :param array_like sample: Sample of parameters of Shape
          (n_samples, n_params).
        :param array_like data: Sample of realization which corresponds to the
          sample of parameters :attr:`sample` (n_samples, n_features).
        :param array_like corners: Hypercube ([min, n_features],
          [max, n_features]).
        :param int fsizes: Number of components of output features.
        :param dict pod: Whether to compute POD or not in local models.

            - **tolerance** (float) -- Basis modes filtering criteria.
            - **dim_max** (int) -- Number of basis modes to keep.

        :param bool standard: Whether to standardize data before clustering.
        :param lst(dict) local_method: List of local surrrogate models
          for clusters or None for Kriging local surrogate models.
        :param float pca_percentage: Percentage of information kept for PCA.
        :param str clusterer: Clusterer from sklearn (unsupervised machine
          learning).
          http://scikit-learn.org/stable/modules/clustering.html#clustering
        :param str classifier: Classifier from sklearn (supervised machine
          learning).
          http://scikit-learn.org/stable/supervised_learning.html
        """
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        samples = self.scaler.transform(samples)
        corners = [[0 for i in range(samples.shape[1])],
                   [1 for i in range(samples.shape[1])]]

        # Only do the clustering on the sensor
        if fsizes is None:
            self.fsizes = data.shape[1]
        else:
            self.fsizes = fsizes

        if data.shape[1] > self.fsizes:
            clust = data[:, self.fsizes:]
        else:
            clust = data

        # Computation of PCA for vector output
        if clust.shape[1] > 1:
            pca = PCA(n_components=pca_percentage)
            pca.fit(clust.T)
            clust = pca.components_.T

        if standard is True:
            scaler = preprocessing.StandardScaler()
            clust = scaler.fit_transform(clust)

        # Acquisition of clusterer
        try:
            # Clusterer is already a sklearn object
            self.logger.debug('Clusterer info:\n{}'
                              .format(clusterer.get_params))
        except AttributeError:
            # Instanciate clusterer from str
            try:
                clusterer = eval("sklearn." + clusterer,
                                 {'__builtins__': None},
                                 {'sklearn': __import__('sklearn'),
                                  'sklearn.cluster': __import__('sklearn.cluster'),
                                  'sklearn.mixture': __import__('sklearn.mixture')})
            except (TypeError, AttributeError):
                raise AttributeError('Clusterer unknown from sklearn.')

            self.logger.debug('Clusterer info:\n{}'
                              .format(clusterer.get_params()))

        # Clustering
        try:
            labels = clusterer.fit_predict(clust)
        except AttributeError:
            clusterer.fit(clust)
            labels = clusterer.predict(clust)

        self.logger.debug('Cluster of data :{}'.format(samples, labels))
        self.clusters_id = np.unique(labels)

        # Acqusition of Classifier
        try:
            # Classifier is already a sklearn object
            self.logger.debug('Classifier info:\n{}'
                              .format(classifier.get_params))
        except AttributeError:
            # Instanciate classifier from str
            try:
                classifier = eval('ske.' + classifier,
                                  {'__builtins__': None},
                                  {'ske': __import__('sklearn'),
                                   'sklearn.svm': __import__('sklearn.svm'),
                                   'sklearn.naive_bayes': __import__('sklearn.naive_bayes'),
                                   'sklearn.gaussian_process': __import__('sklearn.gaussian_process'),
                                   'sklearn.neighbors': __import__('sklearn.neighbors'),
                                   'sklearn.ensemble': __import__('sklearn.ensemble')})
            except (TypeError, AttributeError):
                raise AttributeError('Classifier unknown from sklearn.')
            self.logger.debug('Classifier info:\n{}'
                              .format(classifier.get_params()))

        # Classification
        self.classifier = classifier
        self.classifier.fit(samples, labels)

        self.indice_clt = {}
        self.local_models = {}

        # Creation of local models
        for i, k in enumerate(self.clusters_id):
            idx = np.where(labels == k)[0]
            self.indice_clt[k] = list(idx)
            sample_ = [samples[j] for j in self.indice_clt[k]]
            data_ = [data[j, :self.fsizes] for j in self.indice_clt[k]]

            if pod is not None:
                from batman.pod import Pod
                local_pod = Pod(corners, **pod)
                snapshots = Sample(space=sample_, data=data_)
                local_pod.fit(snapshots)
                data_ = local_pod.VS
            else:
                local_pod = None

            from batman.surrogate import SurrogateModel
            if local_method is None:
                self.local_models[k] = SurrogateModel('kriging', corners, plabels=None)
            else:
                method = [lm for lm in local_method[i]][0]
                self.local_models[k] = SurrogateModel(method, corners, plabels=None,
                                                      **local_method[i][method])

            self.local_models[k].fit(np.asarray(sample_), np.asarray(data_), pod=local_pod)

    def boundaries(self, samples, plabels=None, fname=None):
        """Boundaries of clusters in the parameter space.

        Plot the boundaries for 2D and 3D hypercube or parallel coordinates
        plot for more than 3D to see the influence of sample variables on
        cluster affiliation.

        :param array_like samples: Sample features (n_samples, n_features).
        :param list(str) plabels: Names of each parameters (n_features).
        :param str fname: Whether to export to filename or display the figures.
        :returns: figure.
        :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
        """
        samples = np.asarray(samples)
        samples_ = self.scaler.transform(samples)
        classif_samples = self.classifier.predict(samples_)

        n_dim = samples.shape[1]
        plabels = ['x' + str(i) for i in range(n_dim)]\
            if plabels is None else plabels

        resolution = 20
        cmap = plt.cm.Set1
        color_clt = Normalize(vmin=min(self.clusters_id),
                              vmax=max(self.clusters_id), clip=True)
        markers = ['x', 'o', '+', 'h', '*', 's', 'p', '<', '>', '^', 'v']
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        fig, ax = plt.subplots()

        if n_dim == 1:
            xx = np.linspace(mins, maxs, resolution)[:, None]
            mesh = self.scaler.transform(xx)
            classif = self.classifier.predict(mesh)
            ax.scatter(xx, np.zeros(resolution),
                       alpha=0.3, c=cmap(color_clt(classif)))

            for i, k in enumerate(self.clusters_id):
                samples_ = samples[classif_samples == k]
                ax.scatter(samples_, np.zeros(len(samples_)),
                           c=cmap(color_clt(k)), marker=markers[i])
            ax.set_xlabel(plabels[0])
        elif n_dim == 2:
            xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                                 np.linspace(mins[1], maxs[1], resolution))
            mesh = np.stack((xx.ravel(), yy.ravel()), axis=-1)
            mesh = self.scaler.transform(mesh)
            classif = self.classifier.predict(mesh)

            classif = classif.reshape(xx.shape)
            ax.imshow(classif, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                      aspect='auto', origin='lower', interpolation='gaussian')

            for i, k in enumerate(self.clusters_id):
                ax.scatter(samples[:, 0][classif_samples == k],
                           samples[:, 1][classif_samples == k],
                           c=cmap(color_clt(k)), marker=markers[i])
            ax.set_xlabel(plabels[0])
            ax.set_ylabel(plabels[1])
        else:
            classif_samples = classif_samples.reshape(-1, 1)

            samples_ = np.concatenate((samples_, classif_samples), axis=-1)
            df = pd.DataFrame(samples_, columns=plabels + ["cluster"])
            ax = parallel_coordinates(df, "cluster")
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Parameters range')

            kiviat = Kiviat3D(samples, classif_samples)
            kiviat.plot(fname)

        bat.visualization.save_show(fname, [fig])

        return fig, ax

    def estimate_quality(self, multi_q2=False):
        """Estimate quality of the local models.

        Compute the Q2 for each cluster and return either the
        Q2 for each cluster or the lowest one with its cluster affiliation.

        :param float/bool multi_q2: Whether to return the minimal q2 or the q2
          of each cluster.
        :return: q2: Q2 quality for each cluster or the minimal value
        :rtype: array_like(n_cluster)/float.
        :return: point: Max MSE point for each cluster or the one corresponding
          to minimal Q2 value.
        :rtype: array_like(n_cluster)/float.
        """
        q2 = {}
        point = {}
        for k in self.clusters_id:
            q2[k], point[k] = self.local_models[k].estimate_quality()

        if multi_q2:
            output = q2, point
        else:
            ind = np.argmin(q2)
            output = q2[ind], point[ind]

        return output

    def evaluate(self, points, classification=False):
        """Predict new samples.

        Classify new samples then predict using the corresponding local model.

        :param array_like points: Samples to predict (n_samples,
          n_features).
        :param bool classification: Whether to output classification info.
        :return: predict, sigma: Prediction and sigma of new samples.
        :rtype: array_like (n_samples, n_features), array_like (n_samples,
          n_features).
        """
        points = self.scaler.transform(points)
        classif = self.classifier.predict(points)

        # Indices points for each cluster
        num_clust = np.unique(classif)
        indice_clf = {k: list(np.where(classif == k)[0]) for k in num_clust}

        # idx list of points sorted per cluster
        idx = [item for ind in indice_clf.values() for item in ind]

        # Prediction of new points
        result = np.array([]).reshape(0, self.fsizes)
        sigma = np.array([]).reshape(0, self.fsizes)
        for k in indice_clf:
            clf_ = points[indice_clf[k]]
            result_, sigma_ = self.local_models[k](clf_)

            result = np.concatenate([result, result_])
            try:
                sigma = np.concatenate([sigma, sigma_])
            except ValueError:
                sigma = None

        idx = np.argsort(idx)
        result = result[idx]
        try:
            sigma = sigma[idx]
        except TypeError:
            sigma = None

        return (result, sigma, classif) if classification else (result, sigma)
