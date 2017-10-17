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
    >> predictor = SurrogateModel(method, space.corners)
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
from pathos.multiprocessing import cpu_count
from .kriging import Kriging
from .polynomial_chaos import PC
from .RBFnet import RBFnet
from .multifidelity import Evofusion
from ..tasks import Snapshot
from ..space import Space
from ..misc import ProgressBar, NestedPool


class SurrogateModel(object):

    """Surrogate model."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, corners, **kwargs):
        r"""Init Surrogate model.

        :param np.array corners: space corners to normalize.
        :param str kind: name of prediction method, rbf or kriging.
        :param array_like corners: parameter space corners
          (2 points extrema, n_features).
        :param dict pc: configuration of polynomial chaos.
        :param \**kwargs: See below

        :Keyword Arguments: For Polynomial Chaos the following keywords are
          available

            - 'strategy', str. Least square or Quadrature ['LS', 'Quad'].
            - 'degree', int. Polynomial degree.
            - 'distributions', lst(:class:`openturns.Distribution`).
              Distributions of each input parameter.
            - 'n_sample', int. Number of samples for least square.
        """
        self.kind = kind
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        settings = {"space": {
            "corners": corners,
            "sampling": {"init_size": np.inf, "method": kind}}}
        self.space = Space(settings)
        self.pod = None
        self.update = False  # switch: update model if POD update
        self.dir = {
            'surrogate': 'surrogate.dat',
            'space': 'space.dat',
            'data': 'data.dat',
            'snapshot': 'Newsnap{}'
        }

        self.settings = kwargs

        if self.kind == 'pc':
            self.predictor = PC(strategy=self.settings['strategy'],
                                degree=self.settings['degree'],
                                distributions=self.settings['distributions'],
                                n_sample=self.settings['n_sample'])

    def fit(self, points, data, pod=None):
        """Construct the surrogate.

        :param array_like points: points of the sample (n_samples, n_features).
        :param array_like data: function evaluations (n_samples, n_features).
        :param pod: POD instance.
        :type pod: :class:`batman.pod.pod.Pod`
        """
        self.data = data
        points = np.array(points)
        try:
            points_scaled = self.scaler.transform(points)
        except ValueError:  # With multifidelity
            points_scaled = self.scaler.transform(points[:, 1:])
            points_scaled = np.hstack((points[:, 0].reshape(-1, 1), points_scaled))

        # predictor object
        self.logger.info('Creating predictor of kind {}...'.format(self.kind))
        if self.kind == 'rbf':
            self.predictor = RBFnet(points_scaled, data)
        elif self.kind == 'kriging':
            self.predictor = Kriging(points_scaled, data)
        elif self.kind == 'pc':
            self.predictor.fit(points, data)
        elif self.kind == 'evofusion':
            self.predictor = Evofusion(points_scaled, data)
            self.space.multifidelity = True

        self.pod = pod
        self.space.empty()
        self.space += points
        self.space.doe_init = len(points)

        self.logger.info('Predictor created')
        self.update = False

    def __call__(self, points, path=None):
        """Predict snapshots.

        :param :class:`space.point.Point` points: point(s) to predict
        :param str path: if not set, will return a list of predicted snapshots
        instances, otherwise write them to disk.
        :return: Result
        :rtype: lst(:class:`tasks.snapshot.Snapshot`) or np.array(n_points, n_features)
        :return: Standard deviation
        :rtype: lst(np.array)
        """
        if self.update:
            # pod has changed: update predictor
            self.fit(self.pod.points, self.pod.VS())

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

        if path is not None:
            points = self.scaler.inverse_transform(points)
            snapshots = [None] * len(points)
            for i, point in enumerate(points):
                snapshots[i] = Snapshot(point, results[i])
                s_path = os.path.join(path, self.dir['snapshot'].format(i))
                snapshots[i].write(s_path)
        else:
            return results, sigma

        return snapshots, sigma

    def estimate_quality(self, method='LOO'):
        """Estimate quality of the model.

        :param str method: method to compute quality ['LOO', 'ValidationSet']
        :return: Q2 error
        :rtype: float
        :return: Max MSE point
        :rtype: lst(float)
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
        n_cpu_system = cpu_count()
        n_cpu = n_cpu_system // 3
        if n_cpu < 1:
            n_cpu = 1
        elif n_cpu > points_nb:
            n_cpu = points_nb

        train_pred = copy.deepcopy(self)
        train_pred.space.empty()
        sample = np.array(self.space)

        def loo_quality(i):
            """Error at a point.

            :param int i: point iterator
            :return: prediction
            :rtype: np.array(n_points, n_features)
            """
            train, test = loo_split[i]
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

        q2_loo = r2_score(self.data, y_pred)
        index = ((self.data - y_pred) ** 2).sum(axis=1).argmax()

        logging.getLogger().setLevel(level_init)
        point = self.space[index]

        self.logger.info('Surrogate quality: {}, max error location at {}'
                         .format(q2_loo, point))

        return q2_loo, point

    def write(self, dir_path):
        """Save model, data and space to disk.

        :param str path: path to a directory.
        """
        path = os.path.join(dir_path, self.dir['surrogate'])
        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.predictor)
        self.logger.debug('Model wrote to {}'.format(path))

        path = os.path.join(dir_path, self.dir['space'])
        self.space.write(path)

        path = os.path.join(dir_path, self.dir['data'])
        with open(path, 'wb') as f:
            pickler = pickle.Pickler(f)
            pickler.dump(self.data)
        self.logger.debug('Data wrote to {}'.format(path))

        self.logger.info('Model, data and space wrote.')

    def read(self, dir_path):
        """Load model, data and space from disk.

        :param str path: path to a output/surrogate directory.
        """
        path = os.path.join(dir_path, self.dir['surrogate'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.predictor = unpickler.load()
        self.logger.debug('Model read from {}'.format(path))

        path = os.path.join(dir_path, self.dir['space'])
        self.space.read(path)

        path = os.path.join(dir_path, self.dir['data'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.data = unpickler.load()
        self.logger.debug('Data read from {}'.format(path))

        self.logger.info('Model, data and space loaded.')
