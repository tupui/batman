"""
Core Class
==========

Defines the methods to compute the POD.

"""
import numpy as np
import sys
from .predictor import Predictor
from . import svd
from ..misc import progress_bar


class Core(object):

    """A class for doing pod with raw arrays.

    It works with in-memory numpy arrays and deals with math only.
    """

    refined_points = []
    '''For the original jpod 1 resampling strategy.'''

    leave_one_out_predictor = 'kriging'
    '''Predictor kind for the leave one out method.'''

    def __init__(self, tolerance, dim_max):
        """Initialize POD components.

        :param float tolerance: filtering criteria
        :param int dim_max: number of modes to keep
        """
        self.tolerance = None
        '''Tolerance for basis modes filtering'''

        self.dim_max = None
        '''Maximum number of basis modes.'''

        self.mean_snapshot = None
        '''Mean snapshot.'''

        self.U = None
        '''Singular vectors matrix, ndarray(nb of data, nb of modes).'''

        self.S = None
        '''Singular values matrix, ndarray(nb of modes, nb of snapshots),
        only the diagonal is stored, of length (nb of modes).'''

        self.V = None
        '''Matrix V, ndarray(nb of snapshots, nb of snapshots),
        after filtering (nb of snapshots, nb of modes)'''

        self.tolerance = tolerance
        self.dim_max = dim_max

    def VS(self):
        """Compute V*S matrix product when S is diagonal stored as vector."""
        return self.V * self.S

    def decompose(self, snapshots):
        """Perform the POD.

        Snapshots are modified (zero averaged) and the matrix is
        decomposed using SVD from numpy.

        :param array snapshots: Snapshot matrix (nb of data per snapshot,
        nb of samples)
        """
        # compute mean snapshot
        self.mean_snapshot = np.average(snapshots, 1)

        # center snapshots
        for i in range(snapshots.shape[1]):
            snapshots[:, i] -= self.mean_snapshot

        self.U, self.S, self.V = np.linalg.svd(snapshots, full_matrices=False)
        self.V = self.V.T
        self.U, self.S, self.V = svd.filtering(self.U, self.S, self.V,
                                               self.tolerance, self.dim_max)

    def update(self, snapshot):
        """Update POD with a new snapshot.

        :param np.array snapshot: snapshot data
        """
        self.U, self.S, self.V, self.mean_snapshot = \
            svd.update(self.U, self.S, self.V, self.mean_snapshot, snapshot)
        self.U, self.S, self.V = svd.filtering(self.U, self.S, self.V,
                                               self.tolerance, self.dim_max)

    def estimate_quality(self, points):
        """Quality estimation of the model.

        The quality estimation is done using the leave-one-out method.
        Q2 is computed and the point with max MSE is looked up.

        :param lst points: Points in the parameter space
        :return: Q2 error
        :rtype: float
        :return: Max MSE point
        :rtype: lst(float)
        """
        points_nb = len(points)
        error = np.empty(points_nb)
        mean = np.zeros(self.mean_snapshot.shape[0])

        for i in range(points_nb):
            progress_bar(i, points_nb)

            # Remove point from matrix
            V_1 = np.delete(self.V, i, 0)

            (Urot, S_1, V_1) = svd.downgrade(self.S, V_1)
            (Urot, S_1, V_1) = svd.filtering(Urot, S_1, V_1,
                                             self.tolerance,
                                             self.dim_max)

            points_1 = points[:]
            points_1.pop(i)

            # New prediction with points_nb - 1
            predictor = Predictor(self.leave_one_out_predictor,
                                  points_1,
                                  V_1 * S_1,
                                  self.corners)
            prediction, _ = predictor(points[i])

            # MSE on the missing point
            error[i] = np.sum((np.dot(Urot, prediction) - float(points_nb)
                              / float(points_nb - 1) * self.V[i] * self.S)
                              ** 2)

            mean += np.dot(self.U, self.V[i] * self.S)

        mean = mean / points_nb
        var = 0.
        for i in range(points_nb):
            var += np.sum((mean - np.dot(self.U, self.V[i] * self.S)) ** 2)

        # Compute Q2
        err_q2 = 1 - np.sum(error) / var

        error = error.reshape(-1)
        index = error.argmax()

        return err_q2, points[index]
