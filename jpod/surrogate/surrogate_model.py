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
        :param :class:`pod.pod.Pod` POD: a POD
        """
        self.kind = kind
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(np.array(corners))
        settings = {"space": {
        "corners": corners,
        "sampling": {"init_size": np.inf, "method": kind}}}
        self.space = Space(settings)
        self.pod = None
        self.update = False  # update switch: update model if POD update
        self.directories = {
            'surrogate': 'surrogate.dat',
            'space': 'space.dat',
            'snapshot': 'Newsnap%04d'
        }

    def fit(self, points, data, pod=None):
        """Construct the surrogate."""
        self.pod = pod
        self.space += points
        points = np.array(points)
        points = self.scaler.transform(points)
        # predictor object
        self.logger.info('Creating predictor of kind {}...'.format(self.kind))
        if self.kind == 'rbf':
            self.predictor = RBFnet(points, data)
        elif self.kind == 'kriging':
            self.predictor = Kriging(points, data)
        elif self.kind == 'pc':
            self.predictor = PC(input=points, output=data)

        self.logger.info('Predictor created')
        self.update = False

    def notify(self):
        """Notify the predictor that it requires an update."""
        self.update = True
        self.logger.info('got update notification')

    def __call__(self, points, path=None, snapshots=True):
        """Predict snapshots.

        :param :class:`space.point.Point` points: point(s) to predict
        :param str path: if not set, will return a list of predicted snapshots
        instances, otherwise write them to disk.
        :param bool snapshots: whether or not to return a Snapshot object
        :return: Result
        :rtype: lst(:class:`tasks.snapshot.Snapshot`) or np.array(n_points, n_features)
        :return: Standard deviation
        :rtype: lst(np.array)
        """
        if self.update:
            # pod has changed: update predictor
            self.fit(self.pod.points, self.pod.VS())

        if not isinstance(points, Space):
            points = [points]

        points = np.array(points)
        points = self.scaler.transform(points)
        if self.kind == 'kriging':
            results, sigma = self.predictor.evaluate(points)
        else:
            results = self.predictor.evaluate(points)
            sigma = None

        results = np.atleast_2d(results)

        if self.pod is not None:
            for i, s in enumerate(results):
                results[i] = self.pod.mean_snapshot + np.dot(self.pod.U, s)

        if snapshots:
            snapshots = [None] * len(points)
            for i, point in enumerate(points):
                snapshots[i] = Snapshot(point, results[i])

            if path is not None:
                for i, s in enumerate(snapshots):
                    s_path = os.path.join(path, self.directories['snapshot'] % i)
                    s.write(s_path)
        else:
            return results, sigma

        return snapshots, sigma

    def write(self, path):
            """Save model to disk.

            Write a file containing information on the model.
            And write another one containing the associated space.

            :param str path: path to a directory.
            """
            path_model = os.path.join(path, self.directories['surrogate'])
            with open(path_model, 'wb') as f:
                pickler = pickle.Pickler(f)
                pickler.dump(self.predictor)
            self.logger.info('Wrote model to {}'.format(path_model))

            path_space = os.path.join(path, self.directories['space'])
            with open(path_space, 'wb') as f:
                pickler = pickle.Pickler(f)
                pickler.dump(self.space)
            self.logger.info('Wrote space to {}'.format(path))

    def read(self, path):
        """Load model and space from disk.

        :param str path: path to a output/surrogate directory.
        """
        path_model = os.path.join(path, self.directories['surrogate'])
        with open(path_model, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.predictor = unpickler.load()
        self.logger.info('Model loaded.')

        path_space = os.path.join(path, self.directories['space'])
        with open(path_space, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            self.space = unpickler.load()
        self.logger.info('Space loaded.')
