import logging
from snapshot import Snapshot
from algebra import RBFnet, Kriging

import numpy as np

class Predictor(object):
    """Manages snapshot prediction."""

    logger = logging.getLogger(__name__)


    def __init__(self, kind, points, data):
        """
        kind   : name of prediction method, rbf or kriging
        points : numpy array of points, one per row
        data   : numpy array of data at each point, one per row
        """
        points = np.array(points)

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

        point: point at which prediction will be done

        Returns a numpy array with result.
        """
        result = self.predictor.evaluate(point)
        self.logger.debug('Computed prediction at point %s', point)
        return result




class PodPredictor(Predictor):
    """Manages snapshot prediction."""


    logger = logging.getLogger(__name__)


    def __init__(self, kind, pod):
        """
        pod  : a pod
        kind : name of prediction method, rbf or kriging
        """
        self.pod = pod
        '''Pod used for predictions.'''

        self.update = False
        '''Switch to update or not predictor _preprocessing, used when the pod decomposition is updated.'''

        super(PodPredictor, self).__init__(kind, self.pod.points, self.pod.VS())
        self.pod.register_observer(self)


    def notify(self):
        """Notify the predictor that it requires an update."""
        self.update = True
        self.logger.info('got update notification')


    def __call__(self, points):
        """Compute predictions.

        points: list of points in the parameter space

        Returns a list of snapshots.
        """
        if self.update:
            # pod has changed : update predictor
            1/0
            # TEST ME
            # super(PodPredictor, self).__init__(kind, self.pod.VS())
            # self.update = False

        results = []
        for p in points:
            v = super(PodPredictor, self).__call__(p)
            result = self.pod.mean_snapshot + np.dot(self.pod.U, v)
            results += [Snapshot(p, result)]

        return results
