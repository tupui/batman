# coding: utf8
"""
POD Class
=========

This class wraps the core of POD computations and manages high level IO.

:Example:

::

    >> from pod import Pod
    >> pod = Pod(tol, max, coners)
    >> pod.decompose(snapshots)
    >> pod.write(path)
    >> pod.estimate_quality()

"""
import logging
import os
import copy

from .core import Core
import numpy as np
from ..tasks import Snapshot
from ..space import Space


class Pod(Core):

    """POD class."""

    logger = logging.getLogger(__name__)

    directories = {
        'mean_snapshot': 'Mean',
        'modes': 'Mod_%04d',
        'snapshot': 'Newsnap%04d',
    }
    '''Directory structure to store a pod.'''

    pod_file_name = 'pod.npz'
    '''File name for storing the MPI independent pod data.'''

    points_file_name = 'points.dat'
    '''File name for storing the points.'''

    def __init__(self, settings, snapshot_io=None):
        """Init POD with settings."""
        self.quality = None
        '''Quality of the pod, used to know when it needs to be recomputed.'''

        self.predictor = None
        '''Snapshot predictor.'''

        self.corners = settings['space']['corners']
        '''Space corners.'''

        self.points = Space(settings)
        '''A space to record the points.'''

        # for external pod
        if snapshot_io is not None:
            Snapshot.initialize(snapshot_io)

        super(Pod, self).__init__(settings['pod']['tolerance'],
                                  settings['pod']['dim_max'])

    def __str__(self):
        s = ("\nPOD summary:\n"
             "modes filtering tolerance: {}\n"
             "dimension of parameter space: {}\n"
             "number of snapshots: {}\n"
             "number of data per snapshot: {}\n"
             "maximum number of modes: {}\n"
             "number of modes: {}\n"
             "modes: {}\n"
             .format(self.tolerance, self.points.dim, self.points.size,
                     self.mean_snapshot.shape[0], self.dim_max,
                     self.S.shape[0], self.S))
        return s

    def decompose(self, snapshots):
        """Create a pod from a set of snapshots.

        :param lst(array) snapshots: snapshots matrix
        """
        if len(snapshots) == 0:
            self.logger.info(
                'Empty snapshot list, no decomposition to compute')
            return

        snapshots = [Snapshot.convert(s) for s in snapshots]

        self.logger.info('Decomposing pod basis...')

        matrix = np.column_stack(tuple([s.data for s in snapshots]))
        super(Pod, self).decompose(matrix)

        for s in snapshots:
            self.points += s.point

        self.logger.info('Computed pod basis with %g modes', self.S.shape[0])

    def update(self, snapshot):
        """Update pod with a new snapshot.

        :param snapshot: new snapshot to update the pod with
        """
        self.logger.info('Updating pod basis...')
        snapshot = Snapshot.convert(snapshot)
        super(Pod, self).update(snapshot.data)
        self.points += snapshot.point
        self.logger.info('Updated pod basis with snapshot at point %s',
                         snapshot.point)

    def estimate_quality(self):
        """Quality estimator.

        Estimate the quality of the pod by the leave-one-out method.

        :return: Q2
        :rtype: float
        """
        self.logger.info('Estimating pod quality...')

        # Get rid of predictor creation messages
        level_init = copy.copy(self.logger.getEffectiveLevel())
        logging.getLogger().setLevel(logging.WARNING)

        quality, point = super(Pod, self).estimate_quality(self.points)

        logging.getLogger().setLevel(level_init)

        self.quality = quality
        self.logger.info('pod quality = %g, max error location = %s', quality,
                         point)
        return self.quality, point

    def write(self, path):
        """Save a pod to disk.

        :param str path: path to a directory.
        """
        # create output directory if necessary
        try:
            os.makedirs(path)
        except OSError:
            pass

        # points
        self.points.write(os.path.join(path, self.points_file_name))

        # mean snapshot
        p = os.path.join(path, self.directories['mean_snapshot'])
        Snapshot.write_data(self.mean_snapshot, p)

        # basis
        path_pattern = os.path.join(path, self.directories['modes'])
        for i, u in enumerate(self.U.T):
            Snapshot.write_data(u, path_pattern % i)

        points = np.vstack(tuple(self.points))
        np.savez(os.path.join(path, self.pod_file_name),
                 parameters=points,
                 # TODO: remove, only here for checking vs batman 1
                 values=self.S,
                 vectors=self.V)

        self.logger.info('Wrote pod to %s', path)

    def read(self, path):
        """Read a pod from disk.

        :param str path: path to a directory.
        """
        # points
        self.points.read(os.path.join(path, self.points_file_name))

        # mean snapshot
        p = os.path.join(path, self.directories['mean_snapshot'])
        self.mean_snapshot = Snapshot.read_data(p)

        lazy_data = np.load(os.path.join(path, self.pod_file_name))
        self.S = lazy_data['values']
        self.V = lazy_data['vectors']

        # basis
        size = self.S.shape[0]
        self.U = np.zeros([self.mean_snapshot.shape[0], size])

        path_pattern = os.path.join(path, self.directories['modes'])
        for i in range(size):
            self.U[:, i] = Snapshot.read_data(path_pattern % i)

        self.logger.info('Read pod from %s', path)
