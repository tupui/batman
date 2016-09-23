import logging

from ..surrogate import RBFnet, Kriging
import numpy as np
from .snapshot import Snapshot


class Predictor(object):
    """Manages snapshot prediction."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, points, data, corners):
        """
        :param str kind: name of prediction method, rbf or kriging
        :param np.array points : list of points coordinate
        :param np.array data : output data at each point
        """
        # adimentionalize corners
        bounds = np.array(corners)
        axis = len(bounds.shape)-1
        self.bounds_min = np.array((np.amin(bounds, axis=axis).reshape(2,-1)[0, :]))
        self.bounds_max = np.array((np.amax(bounds, axis=axis).reshape(2,-1)[1, :]))
        points = np.array(points)
        points = np.divide(np.subtract(points, self.bounds_min), self.bounds_max - self.bounds_min)

        # predictor object
        if kind == 'rbf':
            self.predictor = RBFnet(points, data)
        elif kind == 'kriging':
            self.predictor = Kriging(points, data)
        else:
            raise ValueError('kind must be either "rbf" or "kriging"')

        self.logger.info('Created predictor of kind %s', kind)

    def __call__(self, point):
        """Compute a prediction.

        :param tuple(float) point: point at which prediction will be done
        :return: Result and standard deviation
        :rtype: np.arrays
        """
        point = np.divide(np.subtract(point, self.bounds_min), self.bounds_max - self.bounds_min)
        result, sigma = self.predictor.evaluate(point)
 
        return result, sigma


class PodPredictor(Predictor):
    """Manages snapshot prediction."""

    logger = logging.getLogger(__name__)

    def __init__(self, kind, pod):
        """
        :param :class:`jpod.Pod` pod: a pod
        :param str kind : name of prediction method, rbf or kriging
        """
        self.kind = kind

        self.pod = pod
        '''Pod used for predictions.'''

        self.update = False
        '''Switch to update or not predictor _preprocessing, used when the pod decomposition is updated.'''

        super(
            PodPredictor,
            self).__init__(
            self.kind,
            self.pod.points,
            self.pod.VS(),
            self.pod.corners)
        self.pod.register_observer(self)

    def notify(self):
        """Notify the predictor that it requires an update."""
        self.update = True
        self.logger.info('got update notification')

    def __call__(self, points):
        """Compute predictions.

        :param points: list of points in the parameter space
        :return: Result and standard deviation
        :rtype: 
        """
        if self.update:
            # pod has changed : update predictor
            super(PodPredictor, self).__init__(
                self.kind,
                self.pod.points,
                self.pod.VS(),
                self.pod.corners)
            self.update = False

        results = []
        sigma = []
        for p in points:
            v, sigma = super(PodPredictor, self).__call__(p)
            result = self.pod.mean_snapshot + np.dot(self.pod.U, v)
            results += [Snapshot(p, result)]

        return results, sigma
