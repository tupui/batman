# coding: utf8
"""
Mixture Class
=============

Mixture of expert using local Kriging models and clustering Machine Learning.

:Example:
    

::

    >> from batman.surrogate import Mixture
    >> import numpy as np
    >> samples = np.array([[1.,5.], [2.,5.], [8.,5.], [9.,5.]])
    >> data = np.array([[50.,51.,52.], [49.,48.,47.], [10.,11.,12,], [9.,8.,7.]])
    >> sample_new = np.array([[0.5,5.],[10.,5.],[8.5,5.]])
    >> algo = Mixture('GaussianMixture', 'SVC', samples, data, {'n_components':2},
    {'kernel':'linear'})
    >> algo.prediction(sample_new)
    (array([[30.196, 31.196, 32.196],
       [29.   , 28.   , 27.   ],
       [28.804, 27.804, 26.804]]), array([[19.999, 19.999, 19.999],
       [20.   , 20.   , 20.   ],
       [19.999, 19.999, 19.999]]), array([1, 0, 0]))
"""

import logging
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import (DBSCAN,KMeans,MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from batman.surrogate import Kriging


class Mixture:
    """Mixture of expert based on unsupervised machine learning and local 
    Kriging Models to predict new samples and their affiliations."""
    
    logger = logging.getLogger(__name__)

    def __init__(self, clusterer, classifier, samples,
                 data,  param_clt, param_clf, tolerance=0.8):
        """Compute PCA on datas to cluster and form local models for classification.

        Uses the data given to compute PCA from scikit-learn tools.
        If the data is a scalar, no PCA is done.
        Then, set the parameters for the clusterer used from scikit-learn.
        The choosen clusterer and parameters will cluster datas and put a label
        on these. These clusters are then separated to return the affiliation 
        of each sample to its cluster.
        Datas from each cluster are taken to form a local Kriging model
        for each cluster.
        
        :attr:`tolerance` is set at 0.8 for a 80% of the total information.

        :param str clusterer: Scikit-learn clusterer (unsupervised machine learning).
        :param str classifier: Classifier from Scikit-learn
        :param ndarray_like samples: Sample features (n_samples, n_features).
        :param ndarray_like data: Observed data (n_samples, n_features)
        :param float tolerance: % of information for PCA.
        :param dict param_clt,param_clf: Scikit-learn parameters for the clusterer
        and classifier. Check the API or documentation for more informations.
        """
        data = data.T
        
        if len(data[:, 0]) > 1:
            pca = PCA(n_components=tolerance)
            pca.fit(data)
            clust = pca.components_.T
        else:
            clust = data.T

        methods = {
            'DBSCAN': DBSCAN,
            'KMeans': KMeans,
            'MiniBatchKMeans': MiniBatchKMeans,
            'GaussianMixture': GaussianMixture,
            'GaussianNB': GaussianNB,
            'SVC': SVC,
            'KNeighborsClassifier': KNeighborsClassifier,
            'GaussianProcessClassifier': GaussianProcessClassifier}

        try:
            # Clusterer is already a sklearn object
            self.logger.debug('Clusterer info:\n{}'
                              .format(clusterer.get_params))
            method_clt = clusterer
        except AttributeError:
            # Instanciate clusterer from str
            try:
                method_clt = methods[clusterer]
            except (KeyError):
                raise ValueError('Clusterer unknown from sklearn.')

            self.logger.debug('Clusterer info:\n{}'
                              .format(method_clt().get_params()))

        self.clusterer = method_clt(**param_clt)
        
        if method_clt == GaussianMixture:
            self.clusterer.fit(clust)
            self.label = self.clusterer.predict(clust)
        else:
            self.label = self.clusterer.fit_predict(clust)

        self.num_clust = np.unique(self.label)
        indices = []
        self.indice_clt = {}
        self.model = {}
        
        for i, k in enumerate(self.num_clust):
            ii = np.where(self.label == k)[0]
            indices.append(list(ii))
            self.indice_clt[k] = indices[i]
            sample_i = [samples[j] for j in indices[i]]
            data_i = [data[:, j] for j in indices[i]]
            self.model[k] = Kriging(np.asarray(sample_i), np.asarray(data_i))
            
        try:
            # Classifier is already a sklearn object
            self.logger.debug('Classifier info:\n{}'
                              .format(classifier.get_params))
            method_clf = classifier
        except AttributeError:
            # Instanciate classifier from str
            try:
                method_clf = methods[classifier]
            except (KeyError):
                raise ValueError('Classifier unknown from sklearn.')
            self.logger.debug('Classifier info:\n{}'
                              .format(method_clf().get_params()))

        self.classifier = method_clf(**param_clf)
        self.classifier.fit(samples, self.label)        
            
    def prediction(self, sample_new):
        """Classification of new samples based on Supervised Machine Learning
        and make predictions.
        
        Classify new samples given by using the samples already trained and
        a classifier from Scikit-learn tools then make predictions using
        local models with new sample points.
        
        :param ndarray_like sample_new: Samples to classify (n_samples, n_features)
        :return: predict, sigma, classif: Prediction, sigma and classification of
        new samples
        :rtype: ndarray_like (n_samples, n_features), ndarray_like (n_samples, n_features),
        array_like (n_samples)
        """
        classif = self.classifier.predict(sample_new)
        indice = []
        self.indice_clf = {}

        for i, k in enumerate(self.num_clust):
            ii = np.where(classif == k)[0]
            indice.append(list(ii))
            self.indice_clf[k] = indice[i]  

        for i, k in enumerate(self.indice_clf):
            clf_i = sample_new[self.indice_clf[k]]
            predict_i, sigma_i = self.model[k].evaluate(clf_i)

            if i == 0:
                predict = predict_i
                sigma = sigma_i
                ind = self.indice_clf[k]
            else:
                predict = np.concatenate((predict, predict_i))
                sigma = np.concatenate((sigma, sigma_i))
                ind = np.concatenate((ind, self.indice_clf[k]))

        key = np.argsort(ind)
        predict = predict[key]
        sigma = sigma[key]

        return predict, sigma, classif
