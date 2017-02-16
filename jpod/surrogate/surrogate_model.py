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
    >> predictor = SurrogateModel(method, space, data)
    >> predictor.save()
    >> points = [(12.5, 56.8), (2.2, 5.3)]
    >> predictions = SurrogateModel(point)

"""

import logging
from .kriging import Kriging
from .polynomial_chaos import PC
from .RBFnet import RBFnet
from ..tasks import Snapshot
import dill as pickle
import numpy as np
import os


class SurrogateModel(object):

    """Surrogate model."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, space=None, data=None, pod=None):
        """Init POD predictor.

        :param :class:`jpod.Pod` pod: a pod
        :param lst points:
        :param np.array data:
        :param str kind : name of prediction method, rbf or kriging
        """
        self.kind = kind
        self.pod = pod
        points = space
        bounds = np.array(space.corners)

        if self.pod is not None:
            data = self.pod.VS().T  # SV.T a snapshot per column
            self.__call__ = self._call_pod
            self.update = False  # update switch: update model if POD update
        else:
            data = data
            self.__call__ = self._call

        # adimentionalize corners
        axis = len(bounds.shape) - 1
        self.bounds_min = np.array(
            (np.amin(bounds, axis=axis).reshape(2, -1)[0, :]))
        self.bounds_max = np.array(
            (np.amax(bounds, axis=axis).reshape(2, -1)[1, :]))
        points = np.array(points)
        points = np.divide(np.subtract(points, self.bounds_min),
                           self.bounds_max - self.bounds_min)

        # predictor object
        self.logger.info('Creating predictor of kind {}...'.format(self.kind))
        if kind == 'rbf':
            self.predictor = RBFnet(points, data)
        elif kind == 'kriging':
            self.predictor = Kriging(points, data)
        elif kind == 'pc':
            self.predictor = PC(input=points, output=data)

        self.logger.info('Predictor created')

    def notify(self):
        """Notify the predictor that it requires an update."""
        self.update = True
        self.logger.info('got update notification')

    def predict(self, point):
        """Compute a prediction.

        :param tuple(float) point: point at which prediction will be done
        :return: Result and standard deviation
        :rtype: np.arrays
        """
        point = np.divide(np.subtract(point, self.bounds_min),
                          self.bounds_max - self.bounds_min)
        try:
            result, sigma = self.predictor.evaluate(point)
        except ValueError:
            result = self.predictor.evaluate(point)
            sigma = 0

        return result, sigma

    def _call(self, points):
        """Compute predictions.

        :param points: list of points in the parameter space
        :return: Result
        :rtype: lst(:class:`pod.snapshot.Snapshot`)
        :return: Standard deviation
        :rtype: lst(np.array)
        """
        results = []
        sigmas = []
        for p in points:
            result, sigma = self.predict(p)
            results += [Snapshot(p, result)]
            sigmas += [sigma]

        return results, sigmas

    def _call_pod(self, points):
        """Compute predictions.

        :param points: list of points in the parameter space
        :return: Result
        :rtype: lst(:class:`pod.snapshot.Snapshot`)
        :return: Standard deviation
        :rtype: lst(np.array)
        """
        if self.update:
            # pod has changed : update predictor
            self.__init__(self.kind, self.pod)

        results = []
        sigmas = []
        for p in points:
            v, sigma = self.predict(p)
            result = self.pod.mean_snapshot + np.dot(self.pod.U, v)
            results += [Snapshot(p, result)]
            sigmas += [sigma]

        return results, sigmas

    def save(self, path):
            """Save model to disk.

            Write a file containing information on the model

            :param str path: path to a directory.
            """
            # Write the model
            file_name = os.path.join(path, 'model.dat')
            with open(file_name, 'wb') as fichier:
                mon_pickler = pickle.Pickler(fichier)
                mon_pickler.dump(self.predictor)
            self.logger.info('Wrote model to {}'.format(path))

    def load(self, path):
        """Load model from disk.

        :param str path: path to a output/surrogate directory.
        """
        file_name = os.path.join(path, 'model.dat')
        with open(file_name, 'rb') as fichier:
            mon_depickler = pickle.Unpickler(fichier)
            model_recupere = mon_depickler.load()
        self.logger.info('Model loaded.')
        return model_recupere
