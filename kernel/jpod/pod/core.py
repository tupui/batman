import logging

import mpi
import numpy as N
from predictor import Predictor
from space import SpaceBase
import svd


class Core(object):
    """A class for doing pod with raw arrays.

    It works with in-memory numpy arrays and deals with math only.
    """

    refined_points = []
    '''For the original jpod 1 resampling strategy.'''

    leave_one_out_predictor = 'kriging'
    '''Predictor kind for the leave one out method.'''

    def __init__(self, tolerance, dim_max):
        self.tolerance = None
        '''Tolerance for basis modes filtering'''

        self.dim_max = None
        '''Maximum number of basis modes.'''

        self.mean_snapshot = None
        '''Mean snapshot.'''

        self.U = None
        '''Singular vectors matrix, ndarray(nb of data, nb of modes).'''

        self.S = None
        '''Singular values matrix, ndarray(nb of modes, nb of snapshots), only the diagonal is stored, of length (nb of modes).'''

        self.V = None
        '''Matrix V, ndarray(nb of snapshots, nb of snapshots), after filtering (nb of snapshots, nb of modes)'''

        if not 0 < tolerance <= 1:
            raise ValueError('tolerance must be in ]0,1[ : ' + str(tolerance))
        else:
            self.tolerance = tolerance

        if dim_max < 0:
            raise ValueError('pod basis maximum dimension must be positive')
        else:
            self.dim_max = dim_max

    def VS(self):
        """Compute V*S matrix product when S is diagonal stored as vector"""
        # TODO: move to pod.py?
        return self.V * self.S

    def decompose(self, snapshots):
        """Do the POD, snapshots are modified! (zero averaged)

        snapshots : array(nb of data per snapshot, nb of samples)
        """
        # compute mean snapshot
        self.mean_snapshot = N.average(snapshots, 1)

        # center snapshots
        for i in range(snapshots.shape[1]):
            snapshots[:, i] -= self.mean_snapshot

        if mpi.size > 1:
            raise NotImplemented("use dynamic pod in parallel")

        # TODO: play with svd optional arguments
        self.U, self.S, self.V = N.linalg.svd(snapshots, full_matrices=True)
        self.V = self.V.T
        self.U, self.S, self.V = svd.filtering(self.U, self.S, self.V,
                                               self.tolerance, self.dim_max)

    def update(self, snapshot):
        """Update pod with a new snapshot.

        :param snapshot : numpy array of the snapshot data
        """
        self.U, self.S, self.V, self.mean_snapshot = \
            svd.update(self.U, self.S, self.V, self.mean_snapshot, snapshot)
        self.U, self.S, self.V = svd.filtering(self.U, self.S, self.V,
                                               self.tolerance, self.dim_max)

    def estimate_quality(self, points):
        """Return the quality estimation and the corresponding point.

        :param points: list of points in the parameter space.

        The quality estimation is done with the leave-one-out method.
        """
        points_nb = len(points)
        error = N.empty(points_nb)

        for i in range(error.shape[0]):
            V_1 = N.delete(self.V, i, 0)

            (Urot, S_1, V_1) = svd.downgrade(self.S, V_1)
            (Urot, S_1, V_1) = svd.filtering(Urot, S_1, V_1, self.tolerance,
                                             self.dim_max)

            points_1 = points[:]
            points_1.pop(i)

            predictor = Predictor(
                self.leave_one_out_predictor,
                points_1,
                V_1 * S_1)
            prediction, _ = predictor(points[i])
            alphakpred = N.dot(Urot, prediction) - \
                float(points_nb) / float(points_nb - 1) * self.V[i] * self.S

            error[i] = N.linalg.norm(alphakpred)

        quality = N.linalg.norm(error)**2 / error.shape[0]

        error = error.reshape(-1)
        index = error.argmax()

        if True:  # orignal jpod 1 strategy
            error_max = 0.
            for i in range(len(error)):  # TODO: enumerate
                if i not in self.refined_points:
                    if error[i] > error_max:
                        index = i
                        error_max = error[i]
            self.refined_points += [index]

        return (float(quality), points[index])
