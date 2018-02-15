# coding: utf8
"""
SurrogateModel Class
====================

This class manages snapshot prediction.
It allows the creation of a surrogate model and making predictions.

:Example:

::

    >> from surrogate_model import SurrogateModel
    >> method = "kriging"
    >> predictor = SurrogateModel(method, space.corners, points_nb)
    >> predictor.fit(space, target_space)
    >> predictor.save('.')
    >> points = [(12.5, 56.8), (2.2, 5.3)]
    >> predictions = predictor(points)

"""

import os
import copy
import logging
import dill as pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from .kriging import Kriging
from .sk_interface import SklearnRegressor
from .polynomial_chaos import PC
from .RBFnet import RBFnet
from .multifidelity import Evofusion
from ..space import Space
from ..misc import (ProgressBar, NestedPool, cpu_system)


class SurrogateModel(object):
    """Surrogate model."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, corners, max_points_nb, **kwargs):
        r"""Init Surrogate model.

        :param str kind: name of prediction method, rbf or kriging.
        :param array_like corners: hypercube ([min, n_features], [max, n_features]).
        :param integer max_points_nb: number of sample points
        :param \**kwargs: See below

        :Keyword Arguments: For Polynomial Chaos the following keywords are
          available

            - **strategy** (str) -- Least square or Quadrature ['LS', 'Quad', 'SparseLS'].
            - **degree** (int) -- Polynomial degree.
            - **distributions** (lst(:class:`openturns.Distribution`)) --
              Distributions of each input parameter.
            - **n_samples** (int) -- Number of samples for least square.
            - **sparse_param**(array) -- ((int) Maximum Considered Terms, 
              (int) Most Siginificant number, (float) Significance Factor)

          For Kriging the following keywords are available

            - **kernel** (:class:`sklearn.gaussian_process.kernels`.*) --
              Kernel.
            - **noise** (float/bool) -- noise level.
        """
        self.kind = kind
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        self.space = Space(corners, max_points_nb)
        self.data = None
        self.pod = None
        self.update = False  # switch: update model if POD update
        self.dir = {
            'surrogate': 'surrogate.dat',
            'space': '../space/space.dat',
            'data': 'data.dat',
        }

        self.settings = kwargs

        if self.kind == 'pc':
            self.predictor = PC(**self.settings)
        elif self.kind == 'evofusion':
            self.space.multifidelity = [self.settings['cost_ratio'],
                                        self.settings['grand_cost']]

    def fit(self, sample, data, pod=None):
        """Construct the surrogate.

        :param array_like sample: sample of the sample (n_samples, n_features).
        :param array_like data: function evaluations (n_samples, n_features).
        :param pod: POD instance.
        :type pod: :class:`batman.pod.Pod`.
        """
        self.data = data
        sample = np.array(sample)
        try:
            sample_scaled = self.scaler.transform(sample)
        except ValueError:  # With multifidelity
            sample_scaled = self.scaler.transform(sample[:, 1:])
            sample_scaled = np.hstack((sample[:, 0].reshape(-1, 1), sample_scaled))

        # predictor object
        self.logger.info('Creating predictor of kind {}...'.format(self.kind))
        if self.kind == 'rbf':
            self.predictor = RBFnet(sample_scaled, data)
        elif self.kind == 'kriging':
            self.predictor = Kriging(sample_scaled, data, **self.settings)
        elif self.kind == 'pc':
            self.predictor.fit(sample, data)
        elif self.kind == 'evofusion':
            self.predictor = Evofusion(sample_scaled, data)
        else:
            self.predictor = SklearnRegressor(sample_scaled, data, self.kind)

        self.pod = pod
        self.space.empty()
        self.space += sample
        self.space.doe_init = len(sample)

        self.logger.info('Predictor created')
        self.update = False

    def __call__(self, points):
        """Predict snapshots.

        :param points: point(s) to predict.
        :type points: :class:`batman.space.Point` or array_like (n_samples, n_features).
        :param str path: if not set, will return a list of predicted snapshots
          instances, otherwise write them to disk.
        :return: Result.
        :rtype: array_like (n_samples, n_features).
        :return: Standard deviation.
        :rtype: array_like (n_samples, n_features).
        """
        if self.update:
            # pod has changed: update predictor
            self.fit(self.pod.space, self.pod.VS())

        try:
            points[0][0]
        except (TypeError, IndexError):
            points = [points]

        points = np.array(points)

        if self.kind != 'pc':
            points = self.scaler.transform(points)

        if self.kind in ['kriging', 'evofusion']:
            results, sigma = self.predictor.evaluate(points)
        else:
            results = self.predictor.evaluate(points)
            sigma = None

        results = np.atleast_2d(results)

        if self.pod is not None:
            pred = np.empty((len(results), len(self.pod.mean_snapshot)))
            for i, s in enumerate(results):
                pred[i] = self.pod.mean_snapshot + np.dot(self.pod.U, s)

            results = np.atleast_2d(pred)

        return results, sigma

    def estimate_quality(self, method='LOO'):
        """Estimate quality of the model.

        :param str method: method to compute quality ['LOO', 'ValidationSet'].
        :return: Q2 error.
        :rtype: float.
        :return: Max MSE point.
        :rtype: lst(float).
        """
        if self.pod is not None:
            return self.pod.estimate_quality()

        self.logger.info('Estimating Surrogate quality...')
        # Get rid of predictor creation messages
        level_init = copy.copy(self.logger.getEffectiveLevel())
        logging.getLogger().setLevel(logging.WARNING)

        loo = LeaveOneOut()
        loo_split = list(loo.split(self.space[:]))
        points_nb = len(self.space)

        # Multi-threading strategy
        n_cpu_system = cpu_system()
        n_cpu = n_cpu_system // 3
        if n_cpu < 1:
            n_cpu = 1
        elif n_cpu > points_nb:
            n_cpu = points_nb

        train_pred = copy.deepcopy(self)
        train_pred.space.empty()
        sample = np.array(self.space)

        def loo_quality(iteration):
            """Error at a point.

            :param int iteration: point iterator.
            :return: prediction.
            :rtype: array_like of shape (n_points, n_features).
            """
            train, test = loo_split[iteration]
            train_pred.fit(sample[train], self.data[train])
            pred, _ = train_pred(sample[test])

            return pred

        pool = NestedPool(n_cpu)
        progress = ProgressBar(points_nb)
        results = pool.imap(loo_quality, range(points_nb))

        y_pred = np.empty_like(self.data)
        for i in range(points_nb):
            y_pred[i] = results.next()
            progress()

        pool.terminate()

        q2_loo = r2_score(self.data, y_pred)
        index = ((self.data - y_pred) ** 2).sum(axis=1).argmax()

        logging.getLogger().setLevel(level_init)
        point = self.space[index]

        self.logger.info('Surrogate quality: {}, max error location at {}'
                         .format(q2_loo, point))

        return q2_loo, point

    def write(self, fname):
        """Save model and data to disk.

        :param str fname: path to a directory.
        """
        path = os.path.join(fname, self.dir['surrogate'])
        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.predictor)
        self.logger.debug('Model wrote to {}'.format(path))

        path = os.path.join(fname, self.dir['data'])
        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.data)
        self.logger.debug('Data wrote to {}'.format(path))

        self.logger.info('Model and data wrote.')

    def read(self, fname):
        """Load model, data and space from disk.

        :param str fname: path to a directory.
        """
        path = os.path.join(fname, self.dir['surrogate'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.predictor = unpickler.load()
        self.logger.debug('Model read from {}'.format(path))

        path = os.path.join(fname, self.dir['space'])
        self.space.read(path)

        path = os.path.join(fname, self.dir['data'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.data = unpickler.load()[:len(self.space)]
        self.logger.debug('Data read from {}'.format(path))

        self.logger.info('Model, data and space loaded.')
