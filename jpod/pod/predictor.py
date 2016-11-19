# coding: utf8
"""
Predictor Class
===============

This class manages snapshot prediction.
It allows the creation of a surrogate model and making predictions.

:Example:

::

    >> from predictor import Predictor
    >> method = "kriging"
    >> predictor = Predictor(method, pod)
    >> point = [(12.5, 56.8)]
    >> prediction = predictor(point)

"""

import logging
from ..surrogate import RBFnet, Kriging
import numpy as np
from .snapshot import Snapshot


class Predictor():

    """Predictor."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, pod):
        """Init POD predictor.

        :param :class:`jpod.Pod` pod: a pod
        :param str kind : name of prediction method, rbf or kriging
        """
        self.kind = kind

        self.pod = pod
        '''Pod used for predictions.'''

        self.update = False
        '''Switch to update or not predictor _preprocessing,
        used when the pod decomposition is updated.'''

        data = self.pod.VS()
        '''Output data at each point.'''
        points = self.pod.points
        '''List of points coordinate.'''

        self.pod.register_observer(self)

        # adimentionalize corners
        bounds = np.array(self.pod.corners)
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
        else:
            raise ValueError('kind must be either "rbf" or "kriging"')

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
        result, sigma = self.predictor.evaluate(point)

        return result, sigma

    def __call__(self, points):
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
