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
    >> predictions = SurrogateModel(points)

"""

import logging
from .kriging import Kriging
from .polynomial_chaos import PC
from .RBFnet import RBFnet
from .multifidelity import Evofusion
from ..tasks import Snapshot
from ..space import Space
from ..misc import ProgressBar, NestedPool
import dill as pickle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
import copy
from pathos.multiprocessing import cpu_count
import os


class SurrogateModel(object):

    """Surrogate model."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, corners):
        """Init Surrogate model.

        :param np.array corners: space corners to normalize
        :param str kind: name of prediction method, rbf or kriging
        :param np.array corners: parameter space corners (2 points extrema, n_features)
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

    def fit(self, points, data, pod=None):
        """Construct the surrogate."""
        self.data = data
        points = np.array(points)
        try:
            points_scaled = self.scaler.transform(points)
        except ValueError:
            points_scaled = self.scaler.transform(points[:, 1:])
            points_scaled = np.hstack((points[:, 0].reshape(-1, 1), points_scaled))
        # predictor object
        self.logger.info('Creating predictor of kind {}...'.format(self.kind))
        if self.kind == 'rbf':
            self.predictor = RBFnet(points_scaled, data)
        elif self.kind == 'kriging':
            self.predictor = Kriging(points_scaled, data)
        elif self.kind == 'pc':
            self.predictor = PC(input=points_scaled, output=data)
        elif self.kind == 'evofusion':
            self.predictor = Evofusion(points_scaled, data)
            self.space.multifidelity = True

        self.pod = pod
        self.space += points
        self.space.doe_init = len(points)

        self.logger.info('Predictor created')
        self.update = False

    def notify(self):
        """Notify the predictor that it requires an update."""
        self.update = True
        self.logger.info('got update notification')

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
        index = np.argmax(self.data - y_pred)

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
            self.logger.debug('Wrote model to {}'.format(path))

            path = os.path.join(dir_path, self.dir['space'])
            self.space.write(path)
            self.logger.debug('Wrote space to {}'.format(path))

            path = os.path.join(dir_path, self.dir['data'])
            with open(path, 'wb') as f:
                pickler = pickle.Pickler(f)
                pickler.dump(self.data)
            self.logger.debug('Wrote data to {}'.format(path))

            self.logger.info('Wrote model, data and space.')

    def read(self, dir_path):
        """Load model, data and space from disk.

        :param str path: path to a output/surrogate directory.
        """
        path = os.path.join(dir_path, self.dir['surrogate'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.predictor = unpickler.load()
        self.logger.debug('Read model from {}'.format(path))

        path = os.path.join(dir_path, self.dir['space'])
        self.space.read(path)
        self.logger.debug('Read space from {}'.format(path))

        path = os.path.join(dir_path, self.dir['data'])
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.data = unpickler.load()
        self.logger.debug('Read data from {}'.format(path))

        self.logger.info('Model, data and space loaded.')
