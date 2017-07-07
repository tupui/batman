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
import dill as pickle
import numpy as np
from sklearn import preprocessing
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
            'snapshot': 'Newsnap{}'
        }

    def fit(self, points, data, pod=None):
        """Construct the surrogate."""
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

    def write(self, path):
            """Save model to disk.

            Write a file containing information on the model.
            And write another one containing the associated space.

            :param str path: path to a directory.
            """
            path_model = os.path.join(path, self.dir['surrogate'])
            with open(path_model, 'wb') as f:
                pickler = pickle.Pickler(f)
                pickler.dump(self.predictor)
            self.logger.debug('Wrote model to {}'.format(path_model))

            path_space = os.path.join(path, self.dir['space'])
            self.space.write(path_space)
            self.logger.debug('Wrote space to {}'.format(path_space))

            self.logger.debug('Wrote model and space.')

    def read(self, path):
        """Load model and space from disk.

        :param str path: path to a output/surrogate directory.
        """
        path_model = os.path.join(path, self.dir['surrogate'])
        with open(path_model, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.predictor = unpickler.load()
        self.logger.debug('Read model from {}'.format(path_model))

        path_space = os.path.join(path, self.dir['space'])
        self.space.read(path_space)
        self.logger.debug('Read space from {}'.format(path_space))

        self.logger.info('Model and space loaded.')
