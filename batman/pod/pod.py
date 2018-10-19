# coding: utf8
"""
POD Class
=========

Defines the methods to compute the POD.

:Example:

::

    >> from pod import Pod
    >> pod = Pod(corners, tol, max)
    >> pod.fit(snapshots)
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


class Pod:
    """POD class."""

    logger = logging.getLogger(__name__)

    # Directory structure to store a pod
    directories = {
        'mean_snapshot': 'Mean.txt',
        'modes': 'Mods.npz',
    }

    # File name for storing the MPI independent POD data
    pod_file_name = 'pod.npz'

    # File name for storing the points
    points_file_name = 'points.dat'

    def __init__(self, corners, tolerance=0.99, dim_max=100):
        """Initialize POD components.

        The decomposition of the snapshot matrix is stored as attributes:

        - U: Singular vectors matrix, array_like (n_features, n_snapshots),
          after filtering array_like(n_features, n_modes),
        - S: Singular values matrix, array_like (n_modes, n_snapshots),
          only the diagonal is stored, of length (n_modes),
        - V: array_like(n_snapshots, n_snapshots),
          after filtering (n_snapshots, n_modes).

        :param array_like corners: Hypercube ([min, n_features],
          [max, n_features]).
        :param float tolerance: Basis modes filtering criteria.
        :param int dim_max: Number of basis modes to keep.
        """
        self.quality = None
        self.space = []
        self.corners = corners

        # POD computation related
        self.tolerance = tolerance
        self.dim_max = dim_max

        self.mean_snapshot = None
        self.U = None
        self.S = None
        self.V = None

    def __repr__(self):
        """POD summary."""
        s = ("\nPOD summary:\n"
             "-> modes filtering tolerance: {}\n"
             "-> number of snapshots: {}\n"
             "-> number of data per snapshot: {}\n"
             "-> maximum number of modes: {}\n"
             "-> number of modes: {}\n"
             "-> modes: {}\n"
             .format(self.tolerance, len(self.space),
                     self.mean_snapshot.shape[0], self.dim_max,
                     self.S.shape[0], self.S))
        return s

    def fit(self, samples):
        """Create a POD from a set of samples.

        :param samples: Samples.
        :type samples: :class:`batman.space.Sample`.
        """
        self.logger.info('Decomposing POD basis...')

        # Samples' shape is (n_samples, n_features)
        # but SVD requires (n_features, n_samples)
        matrix = np.transpose(samples.data)
        self._fit(matrix)

        self.space.extend(samples.space)

        self.logger.info('Computed POD basis with %g modes', self.S.shape[0])

    def update(self, samples):
        """Update POD with a new snapshot.

        :param samples: new samples to update the POD with.
        """
        self.logger.info('Updating POD basis...')
        for snapshot in samples.data:
            self._update(snapshot)
        self.space.extend(samples.space)
        self.logger.info('Updated POD basis with snapshot at points {}'
                         .format(samples.space))

    def estimate_quality(self):
        """Quality estimator.

        Estimate the quality of the POD by the leave-one-out method.

        :return: Q2.
        :rtype: float.
        """
        self.logger.info('Estimating POD quality...')

        # Get rid of 'kriging' creation messages
        level_init = copy.copy(self.logger.getEffectiveLevel())
        logging.getLogger().setLevel(logging.WARNING)

        quality, point = self._estimate_quality(self.space)

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

        # mean snapshot
        path_snapshot = os.path.join(path, self.directories['mean_snapshot'])
        np.savetxt(path_snapshot, self.mean_snapshot)

        # basis
        np.savez(os.path.join(path, self.pod_file_name),
                 space=self.space,
                 modes=self.S,
                 vectors_v=self.V,
                 vectors_u=self.U)

        self.logger.info('Wrote POD to %s', path)

    def read(self, path):
        """Read a POD from disk.

        :param str path: path to a directory.
        """
        # mean snapshot
        path_snapshot = os.path.join(path, self.directories['mean_snapshot'])
        self.mean_snapshot = np.atleast_1d(np.loadtxt(path_snapshot))

        # basis
        lazy_data = np.load(os.path.join(path, self.pod_file_name))
        self.space = list(lazy_data['space'])
        self.S = lazy_data['modes']
        self.V = lazy_data['vectors_v']
        self.U = lazy_data['vectors_u']

        self.logger.info('Read POD from %s', path)

    @property
    def VS(self):
        """Compute V*S matrix product.

        S is diagonal and stored as vector thus (V*S).T = SV.T
        """
        return self.V * self.S

    def _fit(self, snapshots):
        """Perform the POD.

        The snapshot matrix consists in snapshots arranged in column.
        Snapshots are centered with the mean snapshot then the matrix is
        decomposed using a reduce SVD from numpy.

        `S` is not stored as the conjugate but as `S`.

        :param array_like snapshots: Snapshot matrix (nb of data per snapshot,
            nb of samples)
        """
        # center snapshots using the mean snapshot
        self.mean_snapshot = np.average(snapshots, 1)
        snapshots -= self.mean_snapshot[:, None]

        self.U, self.S, self.V = np.linalg.svd(snapshots, full_matrices=False)
        self.V = self.V.T
        self.U, self.S, self.V = self.filtering(self.U, self.S, self.V,
                                                self.tolerance, self.dim_max)

    def inverse_transform(self, samples):
        """Convert VS back into the original space.

        :param samples: Samples VS to convert (n_samples, n_components).
        :return: Samples in the original space.
        :rtype: array_like (n_samples, n_features)
        """
        samples = np.asarray(samples)
        pred = self.mean_snapshot + np.dot(self.U, samples.T).T
        return np.atleast_2d(pred)

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
        s_cum_sum = np.cumsum(S)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_cum_cum = s_cum_sum / total_sum
        dim = np.searchsorted(ratio_cum_cum, tolerance) + 1
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

            h = snapshot - np.dot(self.U, s_proj)
            h_norm = np.linalg.norm(h)

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
        error_l_two = np.empty(points_nb)
        snapshot_value = np.empty((points_nb, data_len))
        error_matrix = np.empty((points_nb, data_len))
        var_matrix = np.empty((points_nb, data_len))
        surrogate = SurrogateModel('kriging', self.corners, plabels=None)

        def quality(i):
            """Error at a point.

            :param int i: point iterator.
            :return: mean and error.
            :rtype: array_like.
            """
            # Remove point from matrix
            V_1 = np.delete(self.V, i, axis=0)

            (Urot, S_1, V_1) = self.downgrade(self.S, V_1)
            (Urot, S_1, V_1) = self.filtering(Urot, S_1, V_1,
                                              1.,
                                              len(self.S))

            new_pod = copy.deepcopy(self)
            new_pod.space = np.delete(points, i, axis=0)
            new_pod.V = V_1
            new_pod.S = S_1

            # New prediction with points_nb - 1
            surrogate.fit(new_pod.space, new_pod.V * new_pod.S)
            prediction, _ = surrogate(points[i])

            # MSE on the missing point
            error_no_mod = np.dot(Urot, prediction[0]) - float(points_nb) /\
                float(points_nb - 1) * self.V[i] * self.S
            error_vector_ = np.dot(self.U, error_no_mod)
            error_l_two_ = np.sqrt(np.sum(error_no_mod ** 2))

            # Because V = V.T -> V[i] is a column so V[i]S = SV.T
            snapshot_value_ = np.dot(self.U, self.V[i] * self.S)

            return snapshot_value_, error_l_two_, error_vector_

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
            snapshot_value[i], error_l_two[i], error_matrix[i] = results.next()
            progress()

        pool.terminate()

        mean = np.mean(snapshot_value, axis=0)
        for i in range(points_nb):
            var_matrix[i] = (mean - np.dot(self.U, self.V[i] * self.S)) ** 2

        # Compute Q2
        # Use a part of the code of the r2_score function
        # From scikit-learn library
        numerator = (error_matrix ** 2).sum(axis=0, dtype=np.float64)
        denominator = np.sum(var_matrix, axis=0, dtype=np.float64)

        nonzero_denominator = denominator != 0
        nonzero_numerator = numerator != 0
        valid_score = nonzero_denominator & nonzero_numerator
        output_scores = np.ones([data_len])

        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                          denominator[valid_score])
        # arbitrary set to zero to avoid -inf scores, having a constant
        # y_true is not interesting for scoring a regression anyway
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

        q2 = output_scores
        index = error_l_two.argmax()
        err_q2 = np.mean(q2)

        return err_q2, points[index]
