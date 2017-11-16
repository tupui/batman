# coding: utf8
"""
POD Class
=========

Defines the methods to compute the POD.

:Example:

::

    >> from pod import Pod
    >> pod = Pod(tol, max, coners)
    >> pod.decompose(snapshots)
    >> pod.write(path)
    >> pod.estimate_quality()

References
----------
[1] M. Brand: Fast low-rank modifications of the thin singular value decomposition.
2006. DOI:10.1016/j.laa.2005.07.021

[2] T. Braconnier: Towards an adaptive POD/SVD surrogate model for aeronautic design.
Computers & Fluids. 2011. DOI:10.1016/j.compfluid.2010.09.002

"""
import logging
import os
import copy
import numpy as np
from ..surrogate import SurrogateModel
from ..misc import ProgressBar, NestedPool, cpu_system
from ..tasks import Snapshot
from ..space import Space


class Pod(object):

    """POD class."""

    logger = logging.getLogger(__name__)

    # Directory structure to store a pod
    directories = {
        'mean_snapshot': 'Mean',
        'modes': 'Mod_%04d',
        'snapshot': 'Newsnap%04d',
    }

    # File name for storing the MPI independent POD data
    pod_file_name = 'pod.npz'

    # File name for storing the points
    points_file_name = 'points.dat'

    def __init__(self, corners, nsample, tolerance, dim_max, nrefine=0):
        """Initialize POD components.

        The decomposition of the snapshot matrix is stored as attributes:

        - U: Singular vectors matrix, array_like (n_features, n_snapshots),
          after filtering array_like(n_features, n_modes),
        - S: Singular values matrix, array_like (n_modes, n_snapshots),
          only the diagonal is stored, of length (n_modes),
        - V: array_like(n_snapshots, n_snapshots),
          after filtering (n_snapshots, n_modes).

        :param array_like corners: hypercube ([min, n_features], [max, n_features]).
        :param int/array_like sample: number of sample or list of sample of
          shape (n_samples, n_features).
        :param int nrefine: number of point to use for refinement.
        :param float tolerance: basis modes filtering criteria.
        :param int dim_max: number of basis modes to keep.
        """
        self.quality = None
        self.predictor = None
        self.leave_one_out_predictor = 'kriging'
        self.corners = corners
        self.points = Space(self.corners, nsample, nrefine)

        # POD computation related
        self.tolerance = tolerance
        self.dim_max = dim_max

        self.mean_snapshot = None

        self.U = None
        self.S = None
        self.V = None

    def __str__(self):
        """POD summary."""
        s = ("\nPOD summary:\n"
             "modes filtering tolerance: {}\n"
             "dimension of parameter space: {}\n"
             "number of snapshots: {}\n"
             "number of data per snapshot: {}\n"
             "maximum number of modes: {}\n"
             "number of modes: {}\n"
             "modes: {}\n"
             .format(self.tolerance, self.points.dim, len(self.points),
                     self.mean_snapshot.shape[0], self.dim_max,
                     self.S.shape[0], self.S))
        return s

    def decompose(self, snapshots):
        """Create a POD from a set of snapshots.

        :param lst(array) snapshots: snapshots matrix.
        """
        snapshots = [Snapshot.convert(s) for s in snapshots]

        self.logger.info('Decomposing POD basis...')

        matrix = np.column_stack(tuple([s.data for s in snapshots]))
        self._decompose(matrix)

        for s in snapshots:
            self.points += s.point

        self.logger.info('Computed POD basis with %g modes', self.S.shape[0])

    def update(self, snapshot):
        """Update POD with a new snapshot.

        :param snapshot: new snapshot to update the POD with.
        """
        self.logger.info('Updating POD basis...')
        snapshot = Snapshot.convert(snapshot)
        self._update(snapshot.data)
        self.points += snapshot.point
        self.logger.info('Updated POD basis with snapshot at point {}'
                         .format(snapshot.point))

    def estimate_quality(self):
        """Quality estimator.

        Estimate the quality of the POD by the leave-one-out method.

        :return: Q2.
        :rtype: float.
        """
        self.logger.info('Estimating POD quality...')

        # Get rid of predictor creation messages
        level_init = copy.copy(self.logger.getEffectiveLevel())
        logging.getLogger().setLevel(logging.WARNING)

        quality, point = self._estimate_quality(self.points)

        logging.getLogger().setLevel(level_init)

        self.quality = quality
        self.logger.info('POD quality: {}, max error location at {}'
                         .format(quality, point))
        return self.quality, point

    def write(self, path):
        """Save a POD to disk.

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
                 values=self.S,
                 vectors=self.V)

        self.logger.info('Wrote POD to %s', path)

    def read(self, path):
        """Read a POD from disk.

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

        self.logger.info('Read POD from %s', path)

    def VS(self):
        """Compute V*S matrix product.

        S is diagonal and stored as vector thus (V*S).T = SV.T
        """
        return self.V * self.S

    def _decompose(self, snapshots):
        """Perform the POD.

        The snapshot matrix consists in snapshots arranged in column.
        Snapshots are centered with the mean snapshot then the matrix is
        decomposed using a reduce SVD from numpy.

        `S` is not stored as the conjugate but as `S`.

        :param array_like snapshots: Snapshot matrix (nb of data per snapshot,
            nb of samples)
        """
        # compute mean snapshot
        self.mean_snapshot = np.average(snapshots, 1)

        # center snapshots
        for i in range(snapshots.shape[1]):
            snapshots[:, i] -= self.mean_snapshot

        self.U, self.S, self.V = np.linalg.svd(snapshots, full_matrices=False)
        self.V = self.V.T
        self.U, self.S, self.V = self.filtering(self.U, self.S, self.V,
                                                self.tolerance, self.dim_max)

    @staticmethod
    def filtering(U, S, V, tolerance, dim_max):
        """Remove lowest modes in U, S and V.

        :param array_like U: (nb of data, nb of snapshots).
        :param array_like S: (nb of modes).
        :param array_like V: (nb of snapshots, nb of snapshots).
        :param float tolerance: basis modes filtering criteria.
        :param int dim_max: number of basis modes to keep.
        :return: U (nb of data, nb of modes).
        :rtype: array_like.
        :return: S (nb of modes).
        :rtype: array_like.
        :return: V (nb of snapshots, nb of modes).
        :rtype: array_like.
        """
        total_sum = np.sum(S)

        for i in range(S.shape[0]):
            dim = i+1

            with np.errstate(divide='ignore', invalid='ignore'):
                if np.sum(S[:i + 1]) / total_sum > tolerance:
                    break

        dim = min(dim, dim_max)

        # copy ensures an array is not a slice of a bigger memory zone
        if U.shape[1] != dim:
            U = U[:, :dim].copy()
        if S.shape[0] != dim:
            S = S[:dim].copy()
        if V.shape[1] != dim:
            V = V[:, :dim].copy()

        return U, S, V

    def _update(self, snapshot):
        """Update POD with a new snapshot.

        :param array_like snapshot: a snapshot, (n_features,).
        """
        if self.mean_snapshot is None:
            # start off with a mode that will be thrown away
            # by filtering: 0. singular value
            self.mean_snapshot = snapshot
            self.U = np.zeros([snapshot.shape[0], 1])
            self.U[0, 0] = 1.
            self.S = np.zeros([1])
            self.V = np.ones([1, 1])

        else:
            # backup and update mean snapshot
            mean_snapshot_copy = self.mean_snapshot.copy()
            s_nb = self.V.shape[0]
            self.mean_snapshot = (s_nb * self.mean_snapshot + snapshot)\
                / (s_nb + 1)

            # move to POD origin and project the snapshot on the POD basis
            snapshot -= mean_snapshot_copy
            s_proj = np.dot(self.U.T, snapshot)

            # mpi.Allreduce(sendbuf=s_proj.copy(), recvbuf=s_proj, op=mpi.sum)

            h = snapshot - np.dot(self.U, s_proj)
            h_norm = np.linalg.norm(h)

            h_norm *= h_norm
            h_norm = np.sum(h_norm)
            # h_norm = mpi.allreduce(h_norm, op=mpi.sum)
            h_norm = np.sqrt(h_norm)

            # St = |S   U^T s_proj|
            #      |0      norm(h)|
            self.S = np.column_stack([np.diag(self.S), s_proj])
            self.S = np.vstack([self.S, np.zeros_like(self.S[0])])
            self.S[-1, -1] = h_norm

            # Ut = |U  q/norm(q)|
            if h_norm == 0.:
                h_norm = 1.  # fix for h = 0
            self.U = np.column_stack([self.U, h / h_norm])

            # Vt = |V  0|
            #      |0  1|
            self.V = np.vstack([self.V, np.zeros_like(self.V[0])])
            self.V = np.column_stack([self.V, np.zeros_like(self.V[:, 0])])
            self.V[-1, -1] = 1.

            Ud, self.S, Vd_T = np.linalg.svd(self.S)
            self.V = np.dot(self.V, Vd_T.T)
            Un, self.S, self.V = self.downgrade(self.S, self.V)
            self.U = np.dot(self.U, np.dot(Ud, Un))

        self.U, self.S, self.V = self.filtering(self.U, self.S, self.V,
                                                self.tolerance, self.dim_max)

    @staticmethod
    def downgrade(S, Vt):
        r"""Downgrade by removing the kth row of V.

        .. math:: S^{-k} &= U\Sigma R^T Q^T\\
            S^{-k} &= UU'\Sigma'V'^TQ^T \\
            S^{-k} &= U^{-k}\Sigma'V^{(-k)^T}

        :param S: Singular vector, array_like (n_modes,).
        :param Vt: V.T without one row, array_like (n_snapshots - 1, n_modes).
        :return: U', S', V(-k).T
        :rtype: array_like.
        """
        v = np.average(Vt, 0)
        for row in Vt:
            row -= v
        Q, R = np.linalg.qr(Vt)
        R = (S * R).T
        Urot, S, V = np.linalg.svd(R, full_matrices=False)
        V = np.dot(Q, V.T)
        return Urot, S, V

    def _estimate_quality(self, points):
        r"""Quality estimation of the model.

        The quality estimation is done using the leave-one-out method.
        A parallel computation is performed by iterating over the
        points of the DOE.
        Q2 is computed and the point with max MSE is looked up.

        A multithreading strategy is used:

        1. Create a N threads with :math:`N=\frac{n_{cpu}}{n_{restart} \times n_{modes}}`,
        2. If :math:`N > n_{cpu}` restrict the threads to 1.

        :param lst points: Points in the parameter space.
        :return: Q2 error.
        :rtype: float.
        :return: Max MSE point.
        :rtype: array_like (n_features,).
        """
        points_nb = len(points)
        data_len = self.U.shape[0]
        error = np.empty(points_nb)
        mean = np.empty((points_nb, data_len))
        surrogate = SurrogateModel(self.leave_one_out_predictor,
                                   self.corners)

        def quality(i):
            """Error at a point.

            :param int i: point iterator.
            :return: mean and error.
            :rtype: array_like.
            """
            # Remove point from matrix
            V_1 = np.delete(self.V, i, 0)

            (Urot, S_1, V_1) = self.downgrade(self.S, V_1)
            (Urot, S_1, V_1) = self.filtering(Urot, S_1, V_1,
                                              1.,
                                              len(self.S))

            points_1 = points[:]
            points_1.pop(i)

            new_pod = copy.deepcopy(self)
            new_pod.points = points_1
            new_pod.V = V_1
            new_pod.S = S_1

            # New prediction with points_nb - 1
            surrogate.fit(new_pod.points, new_pod.V * new_pod.S)

            prediction, _ = surrogate(points[i])

            # MSE on the missing point
            error = np.sum((np.dot(Urot, prediction[0]) - float(points_nb)
                            / float(points_nb - 1) * self.V[i] * self.S)
                           ** 2)

            # Because V = V.T -> V[i] is a column so V[i]S = SV.T
            mean = np.dot(self.U, self.V[i] * self.S)

            return mean, error

        # Multi-threading strategy
        n_cpu_system = cpu_system()
        n_cpu = n_cpu_system // (len(self.S) * 3)
        if n_cpu < 1:
            n_cpu = 1
        elif n_cpu > points_nb:
            n_cpu = points_nb

        pool = NestedPool(n_cpu)
        progress = ProgressBar(points_nb)
        results = pool.imap(quality, range(points_nb))

        for i in range(points_nb):
            mean[i], error[i] = results.next()
            progress()

        pool.terminate()

        mean = np.sum(mean)
        mean = mean / points_nb
        var = 0.
        for i in range(points_nb):
            var += np.sum((mean - np.dot(self.U, self.V[i] * self.S)) ** 2)

        # Compute Q2
        err_q2 = 1 - np.sum(error) / var

        index = error.argmax()

        return err_q2, points[index]
