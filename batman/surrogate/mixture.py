# coding: utf8
"""
Mixture Class
=============

Mixture of expert using clustering machine learning to form local surrogate models.

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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import batman as bat
from ..space import Sample


class Mixture(object):
    """Mixture of expert based on unsupervised machine learning and local
    Kriging Models to predict new samples and their affiliations."""

    logger = logging.getLogger(__name__)

    def __init__(self, samples, data, plabels, corners, fsizes, pod=None, tolerance=0.99,
                 dim_max=100, standard=True, local_method=None, pca_percentage=0.8,
                 clusterer='cluster.KMeans(n_clusters=2)', classifier='svm.SVC()'):
        """Compute PCA on datas to cluster and form local models used with
        classification of new points to predict.

        Uses the data given to compute PCA from scikit-learn tools.
        If the data is a scalar, no PCA is done.
        Then, set the parameters for the clusterer used from scikit-learn.
        The choosen clusterer and parameters will cluster datas and put a label
        on these. These clusters are then separated to decide the affiliation
        of each sample to its cluster.
        Datas from each cluster are taken to form a local Kriging model
        for each cluster.

        :param array_like samples: Sample features (n_samples, n_features).
        :param array_like data: Observed data (n_samples, n_features).
        :param list(str) plabels: Labels of sample points.
        :param array_like corners: Hypercube ([min, n_features], [max, n_features]).
        :param int fsizes: Number of components of output features.
        :param str/bool pod: Whether to compute POD or not in local models.
        :param float tolerance: Basis modes filtering criteria.
        :param int dim_max: Number of basis modes to keep.
        :param bool: Whether or not to standardize data before clustering phase.
        :param str/dict local_method: Dictionnary of local surrrogate models for clusters
          or None for Kriging local surrogate models.
        :param float pca_percentage: percentage of information kept for PCA.
        :param str clusterer: Clusterer from scikit-learn (unsupervised machine
          learning).
          http://scikit-learn.org/stable/modules/clustering.html#clustering
        :param str classifier: Classifier from Scikit-learn (supervised machine
          learning).
          http://scikit-learn.org/stable/supervised_learning.html
        """
        self.plabels = plabels
        self.fsizes = fsizes[0]
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        samples = self.scaler.transform(samples)
        corners = [[0 for i in range(samples.shape[1])],
                   [1 for i in range(samples.shape[1])]]
        if data.shape[1] > self.fsizes:
            clust = data[:, self.fsizes:].T
        else:
            clust = data.T
        # Computation of PCA
        if clust.shape[0] > 1:
            pca = PCA(n_components=pca_percentage)
            pca.fit(clust)
            clust = pca.components_.T
        else:
            clust = clust.T

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
                                  'sklearn.mixture': __import__('sklearn.mixture')
                                 })
            except (TypeError, AttributeError):
                raise AttributeError('Clusterer unknown from sklearn.')

            self.logger.debug('Clusterer info:\n{}'
                              .format(clusterer.get_params()))

        # Clustering
        if standard is True:
            scaler = preprocessing.StandardScaler()
            clust = scaler.fit_transform(clust)

        self.clust = clust
        try:
            self.label = clusterer.fit_predict(clust)
        except AttributeError:
            clusterer.fit(clust)
            self.label = clusterer.predict(clust)

        self.logger.debug('Cluster of data :{}'.format(samples, self.label))
        self.num_clust = np.unique(self.label)
        indices = []
        self.indice_clt = {}
        self.model = {}

        # Creation of local models
        for i, k in enumerate(self.num_clust):
            ii = np.where(self.label == k)[0]
            indices.append(list(ii))
            self.indice_clt[k] = indices[i]
            sample_i = [samples[j] for j in indices[i]]
            n_sample_i = len(sample_i)
            data_i = [data[j, :self.fsizes] for j in indices[i]]
            if pod is True:
                from batman.pod import Pod
                pod = Pod(corners, tolerance, dim_max)
                snapshots = Sample(space=sample_i, data=data_i)
                pod.fit(snapshots)
                data_i = pod.VS
            from batman.surrogate import SurrogateModel
            if local_method is None:
                self.model[k] = SurrogateModel('kriging', corners, plabels)
            else:
                method = [*local_method[i]][0]
                self.model[k] = SurrogateModel(method, corners, plabels,
                                               **local_method[i][method])

            self.model[k].fit(np.asarray(sample_i), np.asarray(data_i), pod=pod)

        #Acqusition of Classifier
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
                                   'sklearn.ensemble': __import__('sklearn.ensemble')
                                  })
            except (TypeError, AttributeError):
                raise AttributeError('Classifier unknown from sklearn.')
            self.logger.debug('Classifier info:\n{}'
                              .format(classifier.get_params()))

        self.classifier = classifier
        self.classifier.fit(samples, self.label)

    def boundaries(self, samples, fname=None):
        """Plot the boundaries for cluster visualization.

        Plot the boundaries for 2D and 3D hypercube or plot the parallel coordinates
        for more than 3D to see the influence of sample variables on cluster affiliation.

        :param array_like samples: Sample features (n_samples, n_features).
        :param str fname: whether to export to filename or display the figures.
        :returns: figure.
        :rtype: Matplotlib figure instances, Matplotlib AxesSubplot instances.
        """
        n_dim = samples.shape[1]
        resolution = 60
        cm = plt.cm.Set1
        scale = Normalize(clip=True)
        markers = ['x', 'o', '+', 'h', '*', 's', 'p', '<', '>', '^', 'v']
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        fig, ax = plt.subplots(1, 1)

        if n_dim == 3:
            xx, yy, zz = np.meshgrid(*[np.linspace(mins[i], maxs[i], resolution)
                                       for i in range(n_dim)])
            mesh = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=-1)
            mesh = self.scaler.transform(mesh)
            ax = Axes3D(fig)
            classif = self.classifier.predict(mesh)
            color_clf = cm(scale(classif))
            color_clt = cm(scale(self.label))
            ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2],
                       alpha=0.3, edgecolors='none', c=color_clf)
            for i, k in enumerate(self.num_clust):
                ax.scatter(samples[:, 0][self.label == k], samples[:, 1][self.label == k],
                           samples[:, 2][self.label == k], c=color_clt[self.label == k],
                           edgecolors='none', marker=markers[i])
            plt.xlabel(self.plabels[0])
            plt.ylabel(self.plabels[1])
            ax.set_zlabel(self.plabels[2])
        elif n_dim == 2:
            xx, yy = np.meshgrid(*[np.linspace(mins[i], maxs[i], resolution)
                                   for i in range(n_dim)])
            mesh = np.stack((xx.ravel(), yy.ravel()), axis=-1)
            mesh = self.scaler.transform(mesh)
            classif = self.classifier.predict(mesh)
            classif = classif.reshape(resolution, resolution)
            color_clt = cm(scale(self.label))
            ax.contourf(xx, yy, classif,
                        alpha=0.5, cmap=cm)
            for i, k in enumerate(self.num_clust):
                ax.scatter(samples[:, 0][self.label == k], samples[:, 1][self.label == k],
                           edgecolors='none', c=color_clt[self.label == k], marker=markers[i])
            plt.xlabel(self.plabels[0])
            plt.ylabel(self.plabels[1])
        elif n_dim == 1:
            xx = np.meshgrid(*[np.linspace(mins[i], maxs[i], resolution)
                               for i in range(n_dim)])
            yy = np.zeros(samples.shape[0])
            mesh = np.stack((xx.ravel(), yy.ravel()), axis=-1)
            mesh = self.scaler.transform(mesh)
            classif = self.classifier.predict(mesh)
            color_clf = cm(scale(classif))
            color_clt = cm(scale(self.label))
            ax.scatter(mesh[:, 0], mesh[:, 1],
                       alpha=0.3, edgecolors='none', c=color_clf)
            for i, k in enumerate(self.num_clust):
                ax.scatter(samples[self.label==k], c=color_clt[self.label == k],
                           edgecolors='none', marker=markers[i])
            plt.xlabel(self.plabels[0])
        else:
            self.label = self.label.reshape(-1, 1)
            samples = np.concatenate((samples, self.label), axis=-1)
            self.plabels.append("cluster")
            df = pd.DataFrame(samples, columns=self.plabels)
            ax = parallel_coordinates(df, "cluster")
            plt.xlabel('Parameters')
            plt.ylabel('Parameters range')

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
        for k in self.num_clust:
            q2[k], point[k] = self.model[k].estimate_quality()

        if multi_q2 is True:
            output = q2, point
        else:
            ind = np.argmin(q2)
            output = q2[ind], point[ind]

        return output

    def evaluate(self, points):
        """Classification of new samples based on Supervised Machine Learning
        and predictions of new points.

        Classify new samples given by using the samples already trained and
        a classifier from Scikit-learn tools then make predictions using
        local models with new sample points.

        :param array_like points: Samples to classify (n_samples,
          n_features).
        :return: predict, sigma: Prediction and sigma of new samples.
        :rtype: array_like (n_samples, n_features), array_like (n_samples,
          n_features).
        """
        points = self.scaler.transform(points)
        self.classif = self.classifier.predict(points)
        num_clust = np.unique(self.classif)
        indice = []
        indice_clf = {}
        ind = []
        result = np.array([]).reshape(0, self.fsizes)
        sigma = np.array([]).reshape(0, self.fsizes)
        for i, k in enumerate(num_clust):
            ii = np.where(self.classif == k)[0]
            indice.append(list(ii))
            indice_clf[k] = indice[i]

        # Prediction of new points
        for i, k in enumerate(indice_clf):
            clf_i = points[indice_clf[k]]
            result_i, sigma_i = self.model[k](clf_i)

            result = np.concatenate([result, result_i])
            sigma = np.concatenate([sigma, sigma_i])
            ind = np.concatenate((ind, indice_clf[k]))

        key = np.argsort(ind)
        result = result[key]
        sigma = sigma[key]

        return result, sigma
