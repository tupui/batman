import sys
import logging
import pickle
import numpy as N
import sampling
from point import Point
from refiner import MSE


class UnicityError(Exception):
    """Exception for unicity error."""
    pass


class AlienPointError(Exception):
    """Exception when a point is from outer space."""
    pass


class FullSpaceError(Exception):
    """Exception when the maximum number of points is reached."""
    pass


class DimensionError(Exception):
    """Exception for bad data dimension."""
    pass


class SpaceBase(list):
    """Base class to manage a list of points."""

    def __init__(self):
        self.dimension = None


    def add(self, point):
        """Add a points to the space, raise if it already exists."""
        # the following test uses Point class approximative comparison
        # if point is a Point instance
        if point not in self:
            self += [point]
        else:
            raise UnicityError('point %s already exists in the space'% \
                               str(point))

        if not self.dimension:
            # set dimension when adding the first point
            self.dimension = len(point)
        elif len(point) != self.dimension:
            msg = 'coordinates dimensions mismatch, should be %i'%self.dimension
            raise DimensionError(msg)


    @property
    def size(self):
        """Return the number of points in space."""
        return len(self)


    @property
    def dim(self):
        """Return the dimension of the space."""
        return len(self[0])


    def write(self, path):
        """Write space in the file `path`."""
        pickle.dump(self, open(path, 'wb'))


    def read(self, path):
        """Read space from the file `path`."""
        self.empty()
        self += pickle.load(open(path, 'rb'))


    def empty(self):
        """Remove all points."""
        [self.pop() for i in self]



class Space(SpaceBase):
    """Manages the space of parameters."""

    logger = logging.getLogger(__name__)


    def __init__(self, corners_user, max_points_nb, delta_space):
        """
        :param corners: lower and upper corner points that define the space
        :param max_points_nb: maximum number of points allowed in the space
        """
        super(Space, self).__init__()

        self.max_points_nb = int(max_points_nb)
        '''Maximum number of points.'''

        # Extension of space

        c1 = []
        c2 = []

        for i in range(len(corners_user[0])):
            c1.append(corners_user[0][i] - delta_space * (corners_user[1][i]-corners_user[0][i]))  
            c2.append(corners_user[1][i] + delta_space * (corners_user[1][i]-corners_user[0][i]))
        
        corners = (tuple(c1),tuple(c2),) 

        # corner points
   
        try:
            self.corners = [Point(p) for p in corners]
        except Exception, e:
            e.args = ('bad corner points: ' + e.args[0],)
            raise

        for i in range(len(corners[0])):
            if corners[0][i] == corners[1][i]:
                raise ValueError('%dth corners coordinate are equal'%(i+1))


    def __str__(self):
        s  = 'end points           : '
        s += ' '.join([str(c) for c in self.corners]) + '\n'
        s += 'number of points     : ' + str(self.size) + '\n'
        s += 'max number of points : ' + str(self.max_points_nb)
        return s


    def __repr__(self):
        s  = str(self) + '\n'
        s += 'points :' + '\n'
        s += super(Space, self).__str__()
        return s


    def __del__(self):
        pass

    def is_full(self):
        """Returns whether the maximum number of points is reached."""
        return len(self) >= self.max_points_nb


    def add(self, points):
        """Add `points` to the space, raise if space is over full."""
        for point in points:
            # check point is inside
            for i in range(len(self.corners[0])):
                if not self.corners[0][i] <= point[i] <= self.corners[1][i]:
                    raise AlienPointError('point %s is outside the space'% \
                                          str(point))

            super(Space, self).add(Point(point))

            # check space is full
            if len(self) > self.max_points_nb:
                raise FullSpaceError('Maximum number of points reached: %d points'% \
                                     self.max_points_nb)


    def sampling(self, kind, n):
        """Create point samples in the parameter space.

        kind: method of sampling
        n   : number of samples

        Minimum number of samples for halton and sobol : 4
        For uniform sampling, the number of points is per dimensions.
        The points are registered into the space and replace existing ones.

        Returns the list of points.
        """
        if kind == 'halton':
            sampler = sampling.halton
        elif kind == 'lhcc':
            sampler = sampling.clhc
        elif kind == 'lhcr':
            sampler = sampling.rlhc
        elif kind == 'sobol':
            sampler = sampling.sobol
        elif kind == 'sobolscramble':
            sampler = sampling.sobol_scramble
        elif kind == 'faure':
            sampler = sampling.faure
        elif kind == 'uniform':
            sampler = sampling.uniform
            n = [n] * len(self.corners[1])
        else:
            raise ValueError('Bad sampling method: ' + kind)

        bounds  = N.array(self.corners)
        samples = sampler(bounds.shape[1], n, bounds)
        self.empty()
        self.add([s.tolist() for s in samples])

        self.logger.info('Created %d samples with the %s method', len(self),
                         kind)
        self.logger.debug("Points are: \n {}".format(samples))
        return self


    def refine(self, pod, refiner):
        """Refine the sample, update space points and return the new point."""
        # Refinement strategy
        if refiner == 'MSE':
            error = MSE(pod)
            point = error.get_point()

        point = [Point(point)]
        self.add(point)

        self.logger.info('Refined sampling with new point: ' + str(point))

        return point
