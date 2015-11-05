import logging
import numpy as N
import resampling


class QuadTreeRefiner(object):
    """Interface to quad-tree resampling"""

    logger = logging.getLogger(__name__)

    # allowed error when comparing two points
    comparison_error = 1.e-11


    def __init__(self, points):
        points = N.asarray(points)
        self.dim = points.shape[1]
        p = resampling.init_space_part(points)
        self.parameter = p[:, :-4]


    def refine(self, point):
        """docstring for refine"""
        # trick to remove spurious DeprecationWarning from rpyc
        point = tuple(point)

        # find point index
        for i in range(self.parameter.shape[0]):
            if N.amax(N.fabs(self.parameter[i,:self.dim] - N.array(point))) <= \
               self.comparison_error:
                index = i
                break

        if 'index' not in locals():
            raise ValueError('point not in quad tree')

        S_froze  = self.parameter[index, 0 * self.dim:1 * self.dim].reshape(1,-1)
        S0_froze = self.parameter[index, 1 * self.dim:2 * self.dim].reshape(1,-1)
        DS_froze = self.parameter[index, 2 * self.dim:3 * self.dim].reshape(1,-1)
        (NewS, NewS0, NewDS) = resampling.splitelement(S_froze, S0_froze, DS_froze)
        new = N.column_stack((NewS, NewS0, NewDS))
        self.parameter = N.vstack([self.parameter, new])
        return NewS.tolist()
