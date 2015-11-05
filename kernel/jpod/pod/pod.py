import os
import logging
import numpy as N
from snapshot import Snapshot
from core import Core
from predictor import PodPredictor
from space import SpaceBase
import mpi


class Pod(Core):
    """This class wraps the core of pod computations and manages high level IO."""

    logger = logging.getLogger(__name__)

    directories = {
        'mean_snapshot' : 'Mean',
        'modes'         : 'Mod_%04d',
        'snapshot'      : 'Newsnap%04d',
    }
    '''Directory structure to store a pod.'''

    pod_file_name = 'pod.npz'
    '''File name for storing the MPI independent pod data.'''

    points_file_name = 'points.pickle'
    '''File name for storing the points.'''


    def __init__(self, tolerance, dim_max, snapshot_io=None):
        self.quality = None
        '''Quality of the pod, used to know when it needs to be recomputed.'''

        self.observers = []
        '''Objects to update on new pod decomposition.'''

        # TODO: refactor observers?
        self.predictor = None
        '''Snapshot predictor.'''

        self.points = SpaceBase()
        '''A space to record the points.'''

        # for external pod
        # TODO: find a better alternative
        if snapshot_io is not None:
            Snapshot.initialize(snapshot_io)

        super(Pod, self).__init__(tolerance, dim_max)


    def __str__(self):
        format = '%-28s : %s'
        s  = ['POD summary:']
        s += [format%('modes filtering tolerance', self.tolerance)]
        s += [format%('dimension of parameter space', self.points.dim)]
        s += [format%('number of snapshots', self.points.size)]
        s += [format%('number of data per snapshot', self.mean_snapshot.shape[0])]
        s += [format%('maximum number of modes', self.dim_max)]
        s += [format%('number of modes', self.S.shape[0])]
        s += [format%('modes', self.S)]
        return '\n\t'.join(s)


    def register_observer(self, obj):
        """Register an observer for pod decomposition update."""
        self.observers += [obj]


    def _notify_observers(self):
        """Notify observers that depend on pod decomposition update."""
        for o in self.observers:
            o.notify()


    def decompose(self, snapshots):
        """Create a pod from a set of snapshots.

        snapshots : list of snapshots
        """
        if len(snapshots) == 0:
            self.logger.info('Empty snapshot list, no decomposition to compute')
            return

        # TODO: manage memory here: make each snapshot a slice of the matrix if its stable, delete otherwise?
        # TODO: delete data, optional, default true
        snapshots = [Snapshot.convert(s) for s in snapshots]

        self.logger.info('Decomposing pod basis...')

        matrix = N.column_stack(tuple([s.data for s in snapshots]))
        super(Pod, self).decompose(matrix)

        for s in snapshots:
            self.points.add(s.point)

        self._post_processing()
        self.logger.info('Computed pod basis with %g modes', self.S.shape[0])


    def update(self, snapshot):
        """Update pod with a new snapshot.

        :param snapshot: new snapshot to update the pod with
        """
        self.logger.info('Updating pod basis...')
        snapshot = Snapshot.convert(snapshot)
        super(Pod, self).update(snapshot.data)
        self.points.add(snapshot.point)
        self._post_processing()
        self.logger.info('Updated pod basis with snapshot at point %s',
                         snapshot.point)


    def _post_processing(self):
        # reset quality
        self.quality = None
        # poking
        self._notify_observers()


    def estimate_quality(self):
        """Quality estimator.

        Estimate the quality of the pod by the leave-one-out method.
        This part is sequential.
        """
        if self.quality is None:
            self.logger.info('Estimating pod quality...')
            # get rid of the potential tons of predictor creation messages
            logging.getLogger('pod.predictor').setLevel(logging.WARNING)
            self.quality = super(Pod, self).estimate_quality(self.points)
            logging.getLogger('pod.predictor').setLevel(logging.INFO)

        (quality, point) = self.quality
        self.logger.info('pod quality = %g, max error location = %s', quality,
                         point)
        return self.quality


    def predict(self, method, points, path=None):
        """Predict snapshots.

        path : if not set, will return a list of predicted snapshots instances, otherwise write them to disk.
        """
        if self.predictor is None:
            self.predictor = PodPredictor(method, self)

        snapshots = self.predictor(points)

        if path is not None:
            s_list = []
            for i,s in enumerate(snapshots):
                s_path = os.path.join(path, self.directories['snapshot']%i)
                s_list += [s_path]
                s.write(s_path)
            snapshots = s_list
        return snapshots


    def write(self, path):
        """Save a pod to disk.

        :param path: path to a directory.
        """
        # create output directory if necessary
        mpi.makedirs(path)

        # points
        if mpi.myid == 0:
            self.points.write(os.path.join(path, self.points_file_name))

        # mean snapshot
        p = os.path.join(path, self.directories['mean_snapshot'])
        Snapshot.write_data(self.mean_snapshot, p)

        # basis
        path_pattern = os.path.join(path, self.directories['modes'])
        i = 0
        for u in self.U.T:
            Snapshot.write_data(u, path_pattern%i)
            i += 1

        if mpi.myid == 0:
            points = N.vstack(tuple(self.points))
            N.savez(os.path.join(path, self.pod_file_name),
                    parameters = points, # TODO: remove, only here for checking vs jpod 1
                    values  = self.S,
                    vectors = self.V)

        self.logger.info('Wrote pod to %s', path)


    def read(self, path):
        """Read a pod from disk.

        :param path: path to a directory.
        """
        # points
        self.points.read(os.path.join(path, self.points_file_name))

        # mean snapshot
        p = os.path.join(path, self.directories['mean_snapshot'])
        self.mean_snapshot = Snapshot.read_data(p)

        # TODO: MPI com instead of reading from all cpus?
        lazy_data = N.load(os.path.join(path, self.pod_file_name))            
        self.S    = lazy_data['values']
        self.V    = lazy_data['vectors']

        # basis
        size = self.S.shape[0]
        self.U = N.zeros([self.mean_snapshot.shape[0], size])

        path_pattern = os.path.join(path, self.directories['modes'])
        for i in range(size):
            self.U[:,i] = Snapshot.read_data(path_pattern%i)

        self.logger.info('Read pod from %s',path)
