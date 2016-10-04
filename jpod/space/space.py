# -*- coding: utf-8 -*-
import logging
import numpy as np
import sampling
from .point import Point
from .refiner import Refiner


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
            raise UnicityError('point %s already exists in the space' % str(point))

        if not self.dimension:
            # set dimension when adding the first point
            self.dimension = len(point)
        elif len(point) != self.dimension:
            msg = 'coordinates dimensions mismatch, should be %i' % self.dimension
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
        np.savetxt(path, self)

    def read(self, path):
        """Read space from the file `path`."""
        self.empty()
        self += np.loadtxt(path)

    def empty(self):
        """Remove all points."""
        [self.pop() for i in self]


class Space(SpaceBase):

    """Manages the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings):
        """Generate a Space.

        Dilate the space using the delta space.

        :param dict settings: JPOD settings
        """
        super(Space, self).__init__()

        self.settings = settings
        corners_user = settings.space['corners']
        delta_space = settings.space['delta_space']
        self.max_points_nb = int(settings.space['size_max'])

        # Extension of space
        c1 = []
        c2 = []
        for i in range(len(corners_user[0])):
            c1.append(corners_user[0][i] - delta_space * (corners_user[1][i]-corners_user[0][i]))
            c2.append(corners_user[1][i] + delta_space * (corners_user[1][i]-corners_user[0][i]))

        corners = (tuple(c1), tuple(c2),)

        # corner points
        try:
            self.corners = [Point(p) for p in corners]
        except Exception as e:
            e.args = ('bad corner points: ' + e.args[0],)
            raise

        # Point of the sample resampled around
        self.refined_pod_points = []

        for i in range(len(corners[0])):
            if corners[0][i] == corners[1][i]:
                raise ValueError('%dth corners coordinate are equal' % (i+1))

    def __str__(self):
        s = 'end points           : '
        s += ' '.join([str(c) for c in self.corners]) + '\n'
        s += 'number of points     : ' + str(self.size) + '\n'
        s += 'max number of points : ' + str(self.max_points_nb)
        return s

    def __repr__(self):
        s = str(self) + '\n'
        s += 'points :' + '\n'
        s += super(Space, self).__str__()
        return s

    def is_full(self):
        """Return whether the maximum number of points is reached."""
        return len(self) >= self.max_points_nb

    def add(self, points):
        """Add `points` to the space, raise if space is over full."""
        for point in points:
            # check point is inside
            for i in range(len(self.corners[0])):
                if not self.corners[0][i] <= point[i] <= self.corners[1][i]:
                    raise AlienPointError('point %s is outside the space' % str(point))

            super(Space, self).add(Point(point))

            # check space is full
            if len(self) > self.max_points_nb:
                raise FullSpaceError('Maximum number of points reached: %d points' % self.max_points_nb)

    def sampling(self, kind, n):
        """Create point samples in the parameter space.

        Minimum number of samples for halton and sobol : 4
        For uniform sampling, the number of points is per dimensions.
        The points are registered into the space and replace existing ones.

        :param str kind: method of sampling
        :param int n: number of samples
        :return: List of points
        :rtype: self
        """

        bounds = np.array(self.corners)
        samples = sampling.doe(bounds.shape[1], n, bounds, kind)

        self.empty()
        self.add([s.tolist() for s in samples])

        self.logger.info('Created %d samples with the %s method', len(self),
                         kind)
        self.logger.debug("Points are: \n {}".format(samples))
        return self

    def refine(self, pod, point_loo):
        """Refine the sample, update space points and return the new point(s).

        :param pod: POD
        :return: List of points to add
        :rtype: space.point.Point -> lst(tuple(float))
        """
        refiner = Refiner(pod, self.settings)
        # Refinement strategy
        method = self.settings.pod['resample']
        if method == 'MSE':
            new_point = refiner.mse()
        elif method == 'loo_mse':
            new_point = refiner.leave_one_out_mse(point_loo)
        elif method == 'loo_sobol':
            new_point = refiner.leave_one_out_sobol(point_loo)
        elif method == 'extrema':
            new_point, self.refined_pod_points = \
                refiner.extrema(self.refined_pod_points)
        elif method == 'hybrid':
            new_point, self.refined_pod_points = \
                refiner.hybrid(self.refined_pod_points, point_loo)

        try:
            point = [Point(point) for point in [new_point]]
        except TypeError:
            point = [Point(point) for point in new_point]

        self.add(point)

        self.logger.info('Refined sampling with new point: {}'
                         .format(str(point)))

        return point
