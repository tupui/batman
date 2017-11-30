# coding: utf8
"""
Space class
===========

Derives from :py:class:`list` and constitutes a groupment for points.
The space can be filled using low discrepancy sequences from
:class:`openturns.LowDiscrepancySequence`, it can be resampled or points can be
added manually.

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
import itertools
import numpy as np
from scipy.optimize import differential_evolution
from sklearn import preprocessing
from .sampling import Doe
from .point import Point
from .refiner import Refiner
from .. import visualization


class Space(list):
    """Manages the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, corners, sample=np.inf, nrefine=0, plabels=None,
                 multifidelity=None, duplicate=False):
        """Generate a Space.

        :param array_like corners: hypercube ([min, n_features], [max, n_features]).
        :param int/array_like sample: number of sample or list of sample of
          shape (n_samples, n_features).
        :param int nrefine: number of point to use for refinement.
        :param list(str) plabels: parameters' names.
        :param list(float) multifidelity: Whether to consider the first
          parameter as the fidelity level. It is a list of ['cost_ratio',
          'grand_cost'].
        :param bool duplicate: Whether to allow duplicate points in space.
        """
        if isinstance(sample, (int, float)):
            self.doe_init = sample
        else:
            self.doe_init = len(sample)

        if nrefine > 0:
            self.refiner = None
            self.max_points_nb = nrefine + self.doe_init
        else:
            self.max_points_nb = self.doe_init

        self.dim = len(corners[0])
        self.multifidelity = multifidelity
        self.duplicate = duplicate

        # Multifidelity configuration
        if multifidelity is not None:
            self.doe_cheap = self._cheap_doe_from_expensive(self.doe_init)
            self.logger.info('Multifidelity with Ne: {} and Nc: {}'
                             .format(self.doe_init, self.doe_cheap))

        # create parameter list and omit fidelity if relevent
        if plabels is not None:
            self.plabels = plabels
            try:
                self.plabels.remove('fidelity')
            except ValueError:
                pass
        else:
            self.plabels = ["x" + str(i) for i in range(self.dim)]

        # corner points
        self.corners = [Point(p) for p in corners]

        # Point of the sample resampled around
        self.refined_pod_points = []

        # corner points validation
        for i in range(self.dim):
            if corners[0][i] == corners[1][i]:
                raise ValueError('{}th corners coordinate are equal'
                                 .format(i + 1))

    def __str__(self):
        s = ("Space summary:\n"
             "Hypercube points: {}\n"
             "Number of points: {}\n"
             "Max number of points: {}").format([c for c in self.corners],
                                                len(self),
                                                self.max_points_nb)
        return s

    def __repr__(self):
        s = ("{}\n"
             "Points:\n"
             "{}").format(str(self), super(Space, self).__repr__())
        return s

    def __iadd__(self, points):
        """Add `points` to the space.

        Raise if point already exists or if space is over full.

        :param array_like: point(s) to add to space (n_samples, n_features).
        """
        # determine if adding one or multiple points
        try:
            points[0][0]
        except (TypeError, IndexError):
            points = [points]

        points_set = set(self)
        for point in points:
            # check point dimension is correct
            if (len(point) - 1 if self.multifidelity else len(point)) != self.dim:
                self.logger.warning("Ignoring Point - Coordinates dimensions"
                                    "mismatch - is {}, should be {}"
                                    .format(len(point), self.dim))
                continue

            # check space is full
            if self.is_full():
                self.logger.warning("Ignoring Point - Full Space - {}"
                                    .format(point))
                continue

            point = Point(point)

            test_point = np.array(point)[1:] if self.multifidelity else np.array(point)
            # verify point is inside
            not_alien = (self.corners[0] <= test_point).all()\
                & (test_point <= self.corners[1]).all()
            if not not_alien:
                self.logger.warning("Ignoring Point - Out of Space - {}"
                                    .format(point))
                continue

            # verify point is not already in space
            if (point not in points_set) or self.duplicate:
                self.append(point)
                points_set.add(point)
            else:
                self.logger.warning("Ignoring Point - Duplicate - {}"
                                    .format(point))
                continue
        return self

    def sampling(self, n_sample=None, kind='halton', dists=None, discrete=None):
        """Create point samples in the parameter space.

        Minimum number of samples for halton and sobol: 4
        For uniform sampling, the number of points is per dimensions.
        The points are registered into the space and replace existing ones.

        :param int n_sample: number of samples.
        :param str kind: method of sampling.
        :param lst(str) dists: List of valid openturns distributions as string.
        :param int discrete: index of the discrete variable
        :return: List of points.
        :rtype: self.
        """
        if self.multifidelity and n_sample is None:
            n_sample = self._cheap_doe_from_expensive(self.doe_init)
        elif self.multifidelity and n_sample is not None:
            n_sample = self._cheap_doe_from_expensive(n_sample)
        elif not self.multifidelity and n_sample is None:
            n_sample = self.doe_init

        bounds = np.array(self.corners)
        doe = Doe(n_sample, bounds, kind, dists, discrete)
        samples = doe.generate()

        # concatenate cheap and expensive space and add identifier 0 or 1
        if self.multifidelity:
            levels = np.vstack((np.zeros((self.doe_init, 1)),
                                np.ones((self.doe_cheap, 1))))
            samples = np.vstack((samples[0:self.doe_init, :], samples))
            samples = np.hstack((levels, samples))

        if kind == 'saltelli':
            self.duplicate = True

        self.empty()
        self += samples

        self.logger.info("Created {} samples with the {} method"
                         .format(len(self), kind))
        self.logger.debug("Points are:\n{}".format(samples))
        self.logger.info("Discrepancy is {}".format(self.discrepancy()))
        return self

    def refine(self, surrogate, method, point_loo=None, delta_space=0.08,
               dists=None, hybrid=None, discrete=None):
        """Refine the sample, update space points and return the new point(s).

        :param surrogate: Surrogate.
        :type surrogate: :class:`batman.surrogate.SurrogateModel`.
        :param str method: Refinement method.
        :param array_like point_loo: Leave-one-out worst point (n_features,).
        :param float delta_space: Shrinking factor for the parameter space.
        :param lst(str) dists: List of valid openturns distributions as string.
        :param lst(lst(str, int)) hybrid: Navigator as list of [Method, n].
        :param int discrete: index of the discrete variable.
        :return: List of points to add.
        :rtype: element or list of :class:`batman.space.Point`.
        """
        # Refinement strategy
        if (self.refiner is None) and (method == 'hybrid'):
            strategy = [[m[0]] * m[1] for m in hybrid]
            self.hybrid = itertools.cycle(itertools.chain.from_iterable(strategy))

        self.refiner = Refiner(surrogate, self.corners, delta_space, discrete)

        if method == 'sigma':
            new_point = self.refiner.sigma()
        elif method == 'discrepancy':
            new_point = self.refiner.discrepancy()
        elif method == 'loo_sigma':
            new_point = self.refiner.leave_one_out_sigma(point_loo)
        elif method == 'loo_sobol':
            new_point = self.refiner.leave_one_out_sobol(point_loo, dists)
        elif method == 'extrema':
            new_point, self.refined_pod_points = self.refiner.extrema(self.refined_pod_points)
        elif method == 'hybrid':
            new_point, self.refined_pod_points = self.refiner.hybrid(self.refined_pod_points,
                                                                     point_loo,
                                                                     next(self.hybrid),
                                                                     dists)
        elif method == 'optimization':
            new_point = self.refiner.optimization()
        elif method == 'sigma_discrepancy':
            new_point = self.refiner.sigma_discrepancy()

        try:
            point = [Point(point) for point in [new_point]]
        except TypeError:
            point = [Point(point) for point in new_point]

        # Check if points are added to space so only added points are returned
        points_set = set(self)
        new_point = []
        for p in point:
            try:
                points_set.add(point)
                self += point
            except TypeError:
                # Empty list
                continue

            if point in points_set:
                new_point.append(point)

        self.logger.info('Refined sampling with new point: {}'.format(point))
        self.logger.info("New discrepancy is {}".format(self.discrepancy()))

        return point

    def empty(self):
        """Remove all points."""
        del self[:]

    def is_full(self):
        """Return whether the maximum number of points is reached."""
        return len(self) >= self.max_points_nb

    def _cheap_doe_from_expensive(self, n):
        """Compute the number of points required for the cheap DOE.

        :param int n: size of the expensive design.
        :return: size of the cheap design.
        :rtype: int.
        """
        doe_cheap = (self.multifidelity[1] - n) * self.multifidelity[0]
        doe_cheap = int(doe_cheap)
        if doe_cheap / float(n) <= 1:
            self.logger.error('Nc/Ne must be positive')
            raise SystemExit
        self.max_points_nb = n + doe_cheap
        return doe_cheap

    def optimization_results(self):
        """Compute the optimal value."""
        gen = [self.refiner.func(x) for x in self]
        arg_min = np.argmin(gen)
        min_value = gen[arg_min]
        min_x = self[arg_min]
        self.logger.info('New minimal value is: f(x)={} for x={}'
                         .format(min_value, min_x))

        bounds = np.array(self.corners).T
        results = differential_evolution(self.refiner.func, bounds)
        min_value = results.fun
        min_x = results.x
        self.logger.info('Optimization with surrogate: f(x)={} for x={}'
                         .format(min_value, min_x))

    def discrepancy(self, sample=None):
        """Compute the centered discrepancy.

        :return: Centered discrepancy.
        :rtype: float.
        """
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(self.corners)
        if sample is None:
            sample = scaler.transform(self)
        else:
            sample = scaler.transform(sample)

        n_s = len(sample)

        abs_ = abs(sample - 0.5)
        disc1 = np.sum(np.prod(1 + 0.5 * abs_ - 0.5 * abs_ ** 2, axis=1))

        prod_arr = 1
        for i in range(self.dim):
            s0 = sample[:, i]
            prod_arr *= (1 +
                         0.5 * abs(s0[:, None] - 0.5) + 0.5 * abs(s0 - 0.5) -
                         0.5 * abs(s0[:, None] - s0))
        disc2 = prod_arr.sum()

        c2 = (13.0 / 12.0) ** self.dim - 2.0 / n_s * disc1 + 1.0 / (n_s ** 2) * disc2

        return c2

    def read(self, path):
        """Read space from the file `path`."""
        self.empty()
        space = np.loadtxt(path)
        for p in space:
            self += p.flatten().tolist()
        self.logger.debug('Space read from {}'.format(path))

    def write(self, path):
        """Write space in file.

        After writting points, it plots them.

        :param str path: folder to save the points in.
        """
        np.savetxt(path, self)
        resampling = len(self) - self.doe_init
        path = os.path.join(os.path.dirname(os.path.abspath(path)), 'DOE.pdf')
        visualization.doe(self, plabels=self.plabels, resampling=resampling,
                          multifidelity=self.multifidelity, fname=path)
        self.logger.debug('Space wrote to {}'.format(path))
