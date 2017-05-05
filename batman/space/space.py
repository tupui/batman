# coding: utf8
"""
Space class
===========

Derives from :py:class:`list` and constitutes a groupment for points.
The space can be filled using low discrepancy sequences from :class:`openturns.LowDiscrepancySequence`,
it can be resampled or points can be added manually.

:Example:

::

    >> from batman.space import Space
    >> from batman.point import Point
    >> space = Space(settings)
    >> point = Point([12.3, 18.0])
    >> space += point

"""
import logging
import os
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sampling import Doe
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

        :param dict settings: space settings
        """
        self.settings = settings
        self.doe_init = settings['space']['sampling']['init_size']
        self.doe_method = settings['space']['sampling']['method']
        if 'resampling' in settings['space']:
            self.refiner = None
            self.max_points_nb = settings['space']['resampling']['resamp_size'] + self.doe_init
        else:
            self.max_points_nb = self.doe_init
        self.size = 0
        corners = settings['space']['corners']
        self.dim = len(corners[0])

        try:
            self.p_lst = settings['snapshot']['io']['parameter_names']
        except KeyError:
            self.p_lst = ["x" + str(i) for i in range(self.dim)]

        # corner points
        try:
            self.corners = [Point(p) for p in corners]
        except Exception as e:
            e.args = ('bad corner points: ' + e.args[0],)
            raise

        # Point of the sample resampled around
        self.refined_pod_points = []

        # corner points validation
        for i in range(self.dim):
            if corners[0][i] == corners[1][i]:
                raise ValueError('%dth corners coordinate are equal' % (i + 1))

    def __str__(self):
        s = ("Hypercube points: {}\n"
             "Number of points: {}\n"
             "Max number of points: {}").format([c for c in self.corners],
                                                self.size,
                                                self.max_points_nb)
        return s

    def __repr__(self):
        s = ("{}\n"
             "Points:\n"
             "{}").format(str(self), super(Space, self).__repr__())
        return s

    def is_full(self):
        """Return whether the maximum number of points is reached."""
        return len(self) >= self.max_points_nb

    def write(self, path):
        """Write space in file.

        After writting points, it plots them with :func:`Space.plot_space`

        :param str path: folder to save the points in
        """
        np.savetxt(path, self)
        self.plot_space(path)

    def plot_space(self, path):
        """Plot the space of parameters 2d-by-2d.

        :param str path: folder to save the fig in
        """
        sample = np.array(self)
        fig = plt.figure('Design of Experiment')

        if self.dim < 2:
            plt.scatter(sample[0:self.doe_init],
                        [0] * self.doe_init, c='k', marker='o')
            plt.scatter(sample[self.doe_init:],
                        [0] * (len(self) - self.doe_init), c='r', marker='^')
            plt.xlabel(self.p_lst[0])
            plt.tick_params(axis='y', which='both',
                            labelleft='off', left='off')

        else:
            # num figs = ((n-1)**2+(n-1))/2
            fig = plt.figure('Design of Experiment')
            plt.tick_params(axis='both', labelsize=8)

            for i in range(0, self.dim - 1):
                for j in range(i + 1, self.dim):
                    ax = plt.subplot2grid((self.dim, self.dim), (j, i))
                    ax.scatter(sample[0:self.doe_init, i], sample[
                        0:self.doe_init, j], s=5, c='k', marker='o')
                    ax.scatter(sample[self.doe_init:, i], sample[
                        self.doe_init:, j], s=5, c='r', marker='^')
                    ax.tick_params(axis='both', labelsize=(10 - self.dim))
                    if i == 0:
                        ax.set_ylabel(self.p_lst[j])
                    if j == (self.dim - 1):
                        ax.set_xlabel(self.p_lst[i])

        fig.tight_layout()
        path = os.path.join(os.path.dirname(os.path.abspath(path)), 'DOE.pdf')
        fig.savefig(path, transparent=True, bbox_inches='tight')
        plt.close('all')

    def read(self, path):
        """Read space from the file `path`."""
        self.empty()
        space = np.loadtxt(path)
        for p in space:
            self += p.flatten().tolist()

    def empty(self):
        """Remove all points."""
        del self[:]

    def __iadd__(self, points):
        """Add `points` to the space.

        Raise if point already exists or if space is over full.

        :param lst(float) or lst(lst(float)): point(s) to add to space
        """
        # determine if adding one or multiple points
        try:
            points[0][0]
        except (TypeError, IndexError):
            points = [points]

        for point in points:
            # check point dimension is correct
            if len(point) != self.dim:
                self.logger.exception("Coordinates dimensions mismatch, should be {}"
                                      .format(self.dim))
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
                self.append(point)
                self.size += 1
            else:
                raise UnicityError("Point {} already exists in the space"
                                   .format(point))
        return self

    def sampling(self, n=None, kind=None):
        """Create point samples in the parameter space.

        Minimum number of samples for halton and sobol : 4
        For uniform sampling, the number of points is per dimensions.
        The points are registered into the space and replace existing ones.

        :param str kind: method of sampling
        :param int n: number of samples
        :return: List of points
        :rtype: self
        """
        if kind is None:
            kind = self.doe_method
        if n is None:
            n = self.doe_init

        bounds = np.array(self.corners)
        doe = Doe(n, bounds, kind)
        samples = doe.generate()

        self.empty()
        self += samples

        self.logger.info("Created {} samples with the {} method"
                         .format(len(self), kind))
        self.logger.debug("Points are:\n{}".format(samples))
        return self

    def refine(self, surrogate, point_loo):
        """Refine the sample, update space points and return the new point(s).

        :param :class:`surrogate.surrogate_model.SurrogateModel` surrogate: surrogate
        :return: List of points to add
        :rtype: space.point.Point -> lst(tuple(float))
        """
        # Refinement strategy
        if self.refiner is None:
            self.refiner = Refiner(surrogate, self.settings)
            if self.settings['space']['resampling']['method'] == 'hybrid':
                strategy = []
                for method in self.settings['space']['resampling']['hybrid']:
                    strategy.append([method[0]] * method[1])
                self.hybrid = itertools.cycle(itertools.chain.from_iterable(strategy))

        method = self.settings['space']['resampling']['method']
        if method == 'sigma':
            new_point = self.refiner.sigma()
        elif method == 'loo_sigma':
            new_point = self.refiner.leave_one_out_sigma(point_loo)
        elif method == 'loo_sobol':
            new_point = self.refiner.leave_one_out_sobol(point_loo)
        elif method == 'extrema':
            new_point, self.refined_pod_points = self.refiner.extrema(self.refined_pod_points)
        elif method == 'hybrid':
            new_point, self.refined_pod_points = self.refiner.hybrid(self.refined_pod_points,
                                                                point_loo,
                                                                next(self.hybrid))

        try:
            point = [Point(point) for point in [new_point]]
        except TypeError:
            point = [Point(point) for point in new_point]

        self += point

        self.logger.info('Refined sampling with new point: {}'
                         .format(str(point)))

        return point
