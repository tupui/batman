# coding: utf8
"""
Core Class
==========

Defines the methods to compute the POD.

References
----------

M. Brand: Fast low-rank modifications of the thin singular value decomposition. 2006. DOI:10.1016/j.laa.2005.07.021

"""
import numpy as np
import copy
from ..surrogate import SurrogateModel
from .. import mpi
from ..misc import ProgressBar, NestedPool
from pathos.multiprocessing import cpu_count


class Core(object):

    """A class for doing pod with raw arrays.

    It works with in-memory numpy arrays and deals with math only.
    """

    leave_one_out_predictor = 'kriging'
    '''Predictor kind for the leave one out method.'''

    def __init__(self, tolerance, dim_max):
        """Initialize POD components.

        The decomposition of the snapshot matrix is stored as attributes

        - U: Singular vectors matrix, ndarray(nb of data, nb of snapshots),
          after filtering ndarray(nb of data, nb of modes),
        - S: Singular values matrix, ndarray(nb of modes, nb of snapshots),
          only the diagonal is stored, of length (nb of modes),
        - V: ndarray(nb of snapshots, nb of snapshots),
          after filtering (nb of snapshots, nb of modes).

        :param float tolerance: basis modes filtering criteria
        :param int dim_max: number of basis modes to keep
        """
        self.tolerance = tolerance
        self.dim_max = dim_max

        self.mean_snapshot = None

        self.U = None
        self.S = None
        self.V = None

    def VS(self):
        """Compute V*S matrix product.

        S is diagonal and stored as vector thus (V*S).T = SV.T
        """
        return self.V * self.S

    def decompose(self, snapshots):
        """Perform the POD.

        The snapshot matrix consists in snapshots arranged in column.
        Snapshots are centered with the mean snapshot then the matrix is
        decomposed using a reduce SVD from numpy.

        `S` is not stored as the conjugate but as `S`.

        :param np.array snapshots: Snapshot matrix (nb of data per snapshot,
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

    def filtering(self, U, S, V, tolerance, dim_max):
        """Remove lowest modes in U, S and V.

        :param np.array U: (nb of data, nb of snapshots)
        :param np.array S: (nb of modes)
        :param np.array V: (nb of snapshots, nb of snapshots)
        :param float tolerance: basis modes filtering criteria
        :param int dim_max: number of basis modes to keep
        :return: U (nb of data, nb of modes)
        :rtype: np.array
        :return: S (nb of modes)
        :rtype: np.array
        :return: V (nb of snapshots, nb of modes)
        :rtype: np.array
        """
        total_sum = np.sum(S)
        # if total_sum == 0. and S.size == 1:
        #     total_sum = 1.

        for i in range(S.shape[0]):
            dim = i+1

            if total_sum == 0.:
                total_sum = 0.0001

            if np.sum(S[:i+1]) / total_sum > tolerance:
                break

        dim = min(dim, dim_max)

        # copy ensures an array is not a slice of a bigger memory zone
        if U.shape[1] != dim:
            U = U[:, :dim].copy()
        if S.shape[0] != dim:
            S = S[:dim].copy()
        if V.shape[1] != dim:
            V = V[:, :dim].copy()

        return (U, S, V)

    def update(self, snapshot):
        """Update POD with a new snapshot.

        :param np.array snapshot: a snapshot
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

            # move to pod origin and project the snapshot on the pod basis
            snapshot -= mean_snapshot_copy
            s_proj = np.dot(self.U.T, snapshot)

            mpi.Allreduce(sendbuf=s_proj.copy(), recvbuf=s_proj, op=mpi.sum)

            h = snapshot - np.dot(self.U, s_proj)
            h_norm = np.linalg.norm(h)

            h_norm *= h_norm
            h_norm = mpi.allreduce(h_norm, op=mpi.sum)
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

            (Ud, self.S, Vd_T) = np.linalg.svd(self.S)
            self.V = np.dot(self.V, Vd_T.T)
            Un, self.S, self.V = self.downgrade(self.S, self.V)
            self.U = np.dot(self.U, np.dot(Ud, Un))

        self.U, self.S, self.V = self.filtering(self.U, self.S, self.V,
                                                self.tolerance, self.dim_max)

    def downgrade(self, S0, Vt):
        """Downgrade."""
        v = np.average(Vt, 0)
        for row in Vt:
            row -= v
        (Q, R) = np.linalg.qr(Vt)
        R = (S0*R).T  # R = S0[:,np.newaxis] * R.T
        (Urot, S, V) = np.linalg.svd(R)
        V = np.dot(Q, V.T)
        return (Urot, S, V)

    def estimate_quality(self, points):
        r"""Quality estimation of the model.

        The quality estimation is done using the leave-one-out method.
        A parallel computation is performed by iterating over the
        points of the DOE.
        Q2 is computed and the point with max MSE is looked up.

        A multithreading strategy is used:

        1. Create a N threads with :math:`N=\frac{n_{cpu}}{n_{restart} \times n_{modes}}`,
        2. If :math:`N > n_{cpu}` restrict the threads to 1.

        :param lst points: Points in the parameter space
        :return: Q2 error
        :rtype: float
        :return: Max MSE point
        :rtype: lst(float)
        """
        points_nb = len(points)
        data_len = self.U.shape[0]
        error = np.empty(points_nb)
        mean = np.empty((points_nb, data_len))
        surrogate = SurrogateModel(self.leave_one_out_predictor,
                                   self.corners)

        def quality(i):
            """Error at a point.

            :param int i: point iterator
            :return: mean, error
            :rtype: np.array, float
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
        n_cpu_system = cpu_count()
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
