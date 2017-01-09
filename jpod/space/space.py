# coding: utf8
"""
Space class
===========

Derives from :class:`list` and constitutes a groupment for points.
The space can be filled using low discrepancy sequences from :class:`openturns.LowDiscrepancySequence`,
it can be resampled or points can be added manually.

:Example:

::

    >> from jpod.space import Space
    >> from jpod.point import Point
    >> space = Space(settings)
    >> point = Point([12.3, 18.0])
    >> space.add(point)

"""
import logging
import numpy as np
from . import sampling
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


class Space(list):

    """Manages the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings):
        """Generate a Space.

        Dilate the space using the delta space.

        :param dict settings: JPOD settings
        """
        self.dimension = None

        self.settings = settings
        corners = settings['space']['corners']
        self.max_points_nb = settings['space']['size_max']

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
        s = ("Hypercube points: {}\n"
             "Number of points: {}\n"
             "Max number of points: {}").format([str(c) for c in self.corners],
                                                str(self.size),
                                                str(self.max_points_nb))
        return s

    def __repr__(self):
        s = ("{}\n"
             "Points:\n"
             "{}").format(str(self), super(Space, self).__str__())
        return s

    def is_full(self):
        """Return whether the maximum number of points is reached."""
        return len(self) >= self.max_points_nb

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
        space = np.loadtxt(path)
        for p in space:
            self.add(p.flatten().tolist())

    def empty(self):
        """Remove all points."""
        del self[:]

    def add(self, points):
        """Add `points` to the space.

        Raise if point already exists or if space is over full.

        :param lst(float) or lst(lst(float)): point(s) to add to space
        """
        # determine if adding one or multiple points
        try:
            points[0][0]
        except TypeError:
            points = [points]

        for point in points:
            # set dimension when adding the first point
            if not self.dimension:
                self.dimension = len(point)
            elif len(point) != self.dimension:
                self.logger.exception("Coordinates dimensions mismatch, should be {}"
                                      .format(self.dimension))
                raise SystemExit

            # check space is full
            if self.is_full():
                raise FullSpaceError("Space is full"
                                     .format(point))

            point = Point(point)

            # verify point is inside
            not_alien = (self.corners[0] <= np.array(point)).all()\
                & (np.array(point) <= self.corners[1]).all()
            if not not_alien:
                raise AlienPointError("Point {} is out of space"
                                      .format(str(point)))                    

            # verify point is not already in space
            if point not in self:
                self += [point]
            else:
                raise UnicityError("Point {} already exists in the space"
                                    .format(point))

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
        samples = sampling.doe(n, bounds, kind)

        self.empty()
        self.add(samples)

        self.logger.info('Created %d samples with the %s method', len(self),
                         kind)
        self.logger.debug("Points are: \n {}".format(samples))
        return self

    def refine(self, pod, point_loo):
        """Refine the sample, update space points and return the new point(s).

        :param jpod.pod.pod.Pod pod: POD
        :return: List of points to add
        :rtype: space.point.Point -> lst(tuple(float))
        """
        refiner = Refiner(pod, self.settings)
        # Refinement strategy
        method = self.settings['pod']['resample']
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
