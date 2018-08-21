# coding: utf8
"""
Mixture Class
=============

Mixture of expert using clustering machine learning and local Kriging models.

:Example:
    

::

    >> from batman.surrogate import Mixture
    >> import numpy as np
    >> samples = np.array([[1., 5.], [2., 5.], [8., 5.], [9., 5.]])
    >> data = np.array([[50., 51., 52.], [49., 48., 47.], [10., 11., 12,],
       [9., 8., 7.]])
    >> sample_new = np.array([[0.5, 5.],[10., 5.],[8.5, 5.]])
    >> algo = Mixture(samples, data, 'cluster.KMeans(n_clusters=2)', 'SVC()')
    >> algo.prediction(sample_new)
    (array([[30.196, 31.196, 32.196],
       [29.   , 28.   , 27.   ],
       [28.804, 27.804, 26.804]]), array([[19.999, 19.999, 19.999],
       [20.   , 20.   , 20.   ],
       [19.999, 19.999, 19.999]]), array([1, 0, 0]))
"""

import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sklearn.mixture
import sklearn.cluster
import sklearn.gaussian_process
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn import preprocessing
from ..space import Sample
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize


class Mixture(object):
    """Mixture of expert based on unsupervised machine learning and local 
    Kriging Models to predict new samples and their affiliations."""
    
    logger = logging.getLogger(__name__)

    def __init__(self, samples, data, plabels, corners, pod=None, tolerance=0.99,
                 dim_max=100, local_method=None, pca_percentage=0.8,
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

        :param str clusterer: Clusterer from scikit-learn (unsupervised machine 
          learning).
          http://scikit-learn.org/stable/modules/clustering.html#clustering
        :param str classifier: Classifier from Scikit-learn (supervised machine
          learning)
          http://scikit-learn.org/stable/supervised_learning.html
        :param array_like samples: Sample features (n_samples, n_features).
        :param array_like data: Observed data (n_samples, n_features)
        :param float tolerance: % of information for PCA.
        """
        self.plabels = plabels
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        samples = self.scaler.transform(samples)
        corners = [[0 for i in range(samples.shape[1])],
                   [1 for i in range(samples.shape[1])]]
        # Computation of PCA
        scaler = StandardScaler()
        data_clt = scaler.fit_transform(data).T
        if len(data_clt[:, 0]) > 1:
            pca = PCA(n_components = pca_percentage)
            pca.fit(data_clt)
            clust = pca.components_.T
        else:
            clust = data_clt.T
            
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

        try:
            self.label = clusterer.fit_predict(clust)
        except AttributeError:
            clusterer.fit(clust)
            self.label = clusterer.predict(clust)

        num_clust = np.unique(self.label)
        indices = []
        self.indice_clt = {}
        self.model = {}
        
        # Creation of local models
        try:
            index = list(local_method.keys())
        except:
            pass
        
        for i, k in enumerate(num_clust):
            ii = np.where(self.label == k)[0]
            indices.append(list(ii))
            self.indice_clt[k] = indices[i]
            sample_i = [samples[j] for j in indices[i]]
            data_i = [data[j, :] for j in indices[i]]

            if pod == True:
                from batman.pod import Pod
                pod = Pod(corners, self.plabels, sample_i, tolerance, dim_max)
                snapshots = Sample(space=sample_i, data=data_i)
                pod.decompose(snapshots)
                
            from batman.surrogate import SurrogateModel
            if local_method == None:          
                self.model[k] = SurrogateModel('kriging', corners, plabels)
            else: 
                self.model[k] = SurrogateModel(index[i], corners, plabels, **local_method[index[i]])

            self.model[k].fit(np.asarray(sample_i), np.asarray(data_i), pod=pod)

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
                                  'sklearn.ensemble': __import__('sklearn.ensemble')
                                 })
            except (TypeError, AttributeError):
                raise ValueError('Classifier unknown from sklearn.')
            self.logger.debug('Classifier info:\n{}'
                              .format(classifier.get_params()))
            
        self.classifier = classifier
        self.classifier.fit(samples, self.label)

    def boundaries(self, samples):
        """Plot the boundaries for cluster visualization.
        
        Plot the boundaries for 2D and 3D hypercube using viridis colar map
        or plot the parallel coordinates for 1D and more than 3D to see the
        influence of sample variables on cluster affiliation.
        
        :param array_like samples: Sample features (n_samples, n_features).
        """
        n_dim = samples.shape[1]
        resolution = 10
        cmap = cm.get_cmap('viridis')
        scale = Normalize(clip=True)
        
        if n_dim == 3:
            mins = samples.min(axis=0)
            maxs = samples.max(axis=0)
            xx, yy, zz = np.meshgrid(*[np.linspace(mins[i], maxs[i],
                                       resolution) for i in range(n_dim)])
            mesh = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis = -1)
            classif = self.classifier.predict(mesh)  
                 
            fig = plt.figure(figsize=(8,4))
            ax = Axes3D(fig) 
            color = cmap(scale(classif))
            ax.scatter(mesh[:,0], mesh[:,1], mesh[:,2], alpha=0.8, c=color,
                       edgecolors='none', s=30)
            plt.show()
        elif n_dim ==2:
            mins = samples.min(axis=0)
            maxs = samples.max(axis=0)
            xx, yy = np.meshgrid(*[np.linspace(mins[i], maxs[i],
                                   resolution) for i in range(n_dim)])
            mesh = np.stack((xx.ravel(), yy.ravel()), axis = -1)
            classif = self.classifier.predict(mesh)              
                         
            fig = plt.figure(figsize=(8,4))
            color = cmap(scale(classif))
            plt.scatter(mesh[:,0], mesh[:,1], alpha=0.8, c=color,
                        edgecolors='none', s=30)
            plt.show() 
        else:
            samples = np.stack((samples, self.label), axis = -1)
            self.plabels.append("cluster")
            df = pd.DataFrame(samples, columns = self.plabels)
            parallel_coordinates(df, "cluster")
            plt.show()    

    def estimate_quality(self, multiq2 = False):
        """Estimate quality of the local models.
        
        Compute the Q2 for each cluster and return either the
        Q2 for each cluster or the lowest one with its cluster affiliation.
        
        :param float/bool multiq2: Whether to return the minimal q2 or the q2
          of each cluster. 
        :return: q2: Q2 quality for each cluster or the minimal value
        :rtype: array_like(n_cluster) or float
        :return: point: Max MSE point for each cluster or the one corresponding
          to minimal Q2 value
        :rtype: array_like(n_cluster) or float
        """
        q2 = {}
        point = {}
        for k in self.num_clust:
            q2[k], point[k] = self.model[k].estimate_quality()
        
        if multiq2 == True:
            output = q2, point
        else: 
            ind = np.argmin(q2)
            output = q2[ind], point[ind]
            
        return output     
            
    def evaluate(self, samples_new):
        """Classification of new samples based on Supervised Machine Learning
        and predictions of new points.
        
        Classify new samples given by using the samples already trained and
        a classifier from Scikit-learn tools then make predictions using
        local models with new sample points.
        
        :param array_like sample_new: Samples to classify (n_samples,
          n_features)
        :return: predict, sigma and classif: Prediction, sigma and 
          classification of new samples
        :rtype: array_like (n_samples, n_features), array_like (n_samples,
          n_features), array_like (n_samples)
        """
        samples_new = self.scaler.transform(samples_new)
        classif = self.classifier.predict(samples_new)
        num_clust = np.unique(classif)
        indice = []
        indice_clf = {}
        
        for i, k in enumerate(num_clust):
            ii = np.where(classif == k)[0]
            indice.append(list(ii))
            indice_clf[k] = indice[i]  

        # Prediction of new points 
        for i, k in enumerate(indice_clf):
            clf_i = samples_new[indice_clf[k]]
            result_i, sigma_i = self.model[k](clf_i)

            if i == 0:
                result = result_i
                sigma = sigma_i
                ind = indice_clf[k]
            else:
                result = np.concatenate((result, result_i))
                sigma = np.concatenate((sigma, sigma_i))
                ind = np.concatenate((ind, indice_clf[k]))

        key = np.argsort(ind)
        result = result[key]
        sigma = sigma[key]

        return result, sigma, classif
