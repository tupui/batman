# coding: utf8
"""
Space class
===========

Derives from :class:`Sample` and constitutes a groupment for points.
The space can be filled using low discrepancy sequences from
:class:`openturns.LowDiscrepancySequence`, it can be resampled or points can be
added manually.

:Example:

::

    >> from batman.space import Space
    >> space = Space(settings)
    >> point = [12.3, 18.0]
    >> space += point

"""
import logging
import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import preprocessing
from .sampling import Doe
from .sample import Sample
from .refiner import Refiner
from .gp_sampler import GpSampler
from .. import visualization


class Space(Sample):
    """Manages the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, corners, sample=np.inf, nrefine=0, plabels=None, psizes=None,
                 multifidelity=None, duplicate=False, threshold=0., gp_samplers=None):
        """Generate a Space.

        :param array_like corners: hypercube ([min, n_features], [max, n_features]).
        :param int/array_like sample: number of sample or list of sample of
          shape (n_samples, n_features).
        :param int nrefine: number of point to use for refinement.
        :param list(str) plabels: parameters' names.
        :param list(int) psizes: number of components of each parameters.
        :param list(float) multifidelity: Whether to consider the first
          parameter as the fidelity level. It is a list of ['cost_ratio',
          'grand_cost'].
        :param bool duplicate: Whether to allow duplicate points in space,
        :param float threshold: minimal distance between 2 disctinct points,
        :param dict gp_samplers: Gaussian process samplers for functional inputs
        """
        try:
            self.doe_init = len(sample)
        except TypeError:
            self.doe_init = sample

        self.max_points_nb = self.doe_init
        if nrefine > 0:
            self.refiner = None
            self.hybrid = None
            self.max_points_nb += nrefine
        self.refined_points = []

        self.dim = len(corners[0])
        self.multifidelity = multifidelity
        self.duplicate = duplicate
        self.threshold = threshold

        # Gaussian process samplers for functional inputs
        if gp_samplers:
            self.nb_gp_samplers = len(gp_samplers['index'])
            if 'thresholds' in gp_samplers.keys():
                thresholds = [1 - gp_samplers['thresholds'][i]
                              for i in range(self.nb_gp_samplers)]
            else:
                thresholds = [0.01] * self.nb_gp_samplers
            self.gp_samplers = [GpSampler(gp_samplers['reference'][i],
                                          gp_samplers['kernel'][i],
                                          gp_samplers['add'][i],
                                          thresholds[i])
                                for i in range(self.nb_gp_samplers)]
        else:
            self.nb_gp_samplers = 0

        # Parameter names list
        if plabels is None:
            plabels = ['x{}'.format(i) for i in range(self.dim)]
        if psizes is None:
            psizes = [1] * self.dim
            if gp_samplers:
                for i in range(self.nb_gp_samplers):
                    index_i = gp_samplers['index'][i]
                    n_modes_i = gp_samplers['n_nodes'][i]
                    psizes[index_i] = n_modes_i

        # Multifidelity configuration
        psizes = [1] + psizes if multifidelity else psizes
        if multifidelity and not isinstance(multifidelity, bool):
            self.doe_cheap = self._cheap_doe_from_expensive(self.doe_init)
            self.logger.info("Multifidelity with Ne: {} and Nc: {}"
                             .format(self.doe_init, self.doe_cheap))

        # Corner points
        self.corners = np.asarray(corners)
        if np.any(self.corners[0] == self.corners[1]):
            raise ValueError('corners coordinates at positions {} are equal'
                             .format(np.flatnonzero(self.corners[0] == self.corners[1])))

        # Initialize Sample container with empty space dataframe
        super(Space, self).__init__(plabels=plabels, psizes=psizes)

        try:
            self += sample
        except IndexError:
            pass

    def sampling(self, n_samples=None, kind='halton', dists=None, discrete=None):
        """Create point samples in the parameter space.

        Minimum number of samples for halton and sobol: 4
        For uniform sampling, the number of points is per dimensions.
        The points are registered into the space and replace existing ones.

        :param int n_samples: number of samples.
        :param str kind: method of sampling.
        :param lst(str) dists: List of valid openturns distributions as string.
        :param int discrete: index of the discrete variable
        :return: List of points.
        :rtype: :class:`numpy.ndarray`
        """
        if n_samples is None:
            n_samples = self.doe_init
        if self.multifidelity:
            n_samples = self._cheap_doe_from_expensive(n_samples)

        # sample the inputs non GP-distributed
        if dists is None:
            dists_not_gp = None
            corners_not_gp = self.corners
        else:
            not_gp = [dist != 'GpSampler' for dist in dists]
            dists_not_gp = [dist for i, dist in enumerate(dists) if not_gp[i]]
            corners_not_gp = np.array(self.corners)[:, not_gp]

        doe = Doe(n_samples, corners_not_gp, kind, dists_not_gp, discrete)
        samples = doe.generate()

        # sample the GP-distributed inputs and concatenate
        if hasattr(self, 'gp_samplers'):
            for i in range(self.nb_gp_samplers):
                gp_samples_i = self.gp_samplers[i](n_samples, kind=kind)['Values']
                samples = np.append(samples, gp_samples_i, axis=1)

        # concatenate cheap and expensive space, prepend identifier 0 or 1
        if self.multifidelity:
            fidelity = np.append(np.zeros(self.doe_init), np.ones(self.doe_cheap)).reshape(-1, 1)
            samples = np.append(samples[:self.doe_init, :], samples, axis=0)
            samples = np.append(fidelity, samples, axis=1)

        if kind == 'saltelli':
            self.duplicate = True

        self.empty()
        self.append(samples)

        self.logger.info("Created {} samples with the {} method".format(len(self), kind))
        self.logger.debug("Points are:\n{}".format(samples))

        if not hasattr(self, 'gp_samplers'):
            s = int(bool(self.multifidelity))
            self.logger.info("Discrepancy is {}"
                             .format(self.discrepancy(self.values[:, s:], self.corners)))

        return self.values

    def refine(self, surrogate, method, point_loo=None, delta_space=0.08,
               dists=None, hybrid=None, discrete=None, extremum='min'):
        """Refine the sample, update space points and return the new point(s).

        :param surrogate: Surrogate.
        :type surrogate: :class:`batman.surrogate.SurrogateModel`.
        :param str method: Refinement method.
        :param array_like point_loo: Leave-one-out worst point (n_features,).
        :param float delta_space: Shrinking factor for the parameter space.
        :param lst(str) dists: List of valid openturns distributions as string.
        :param lst(lst(str, int)) hybrid: Navigator as list of [Method, n].
        :param int discrete: Index of the discrete variable.
        :param str extremum: Minimization or maximization objective
          ['min', 'max'].
        :return: List of points to add.
        :rtype: :class:`numpy.ndarray`
        """
        try:
            self.refiner
        except AttributeError:
            return np.empty((0, len(self.plabels)))

        # Refinement strategy
        if (self.refiner is None) and (method == 'hybrid'):
            strategy = [[m[0]] * m[1] for m in hybrid]
            self.hybrid = itertools.cycle(itertools.chain.from_iterable(strategy))

        pod = None if surrogate.pod is None else surrogate.pod
        self.refiner = Refiner(surrogate, self.corners, delta_space, discrete, pod)

        if method == 'sigma':
            new_point = self.refiner.sigma()
        elif method == 'discrepancy':
            new_point = self.refiner.discrepancy()
        elif method == 'loo_sigma':
            new_point = self.refiner.leave_one_out_sigma(point_loo)
        elif method == 'loo_sobol':
            new_point = self.refiner.leave_one_out_sobol(point_loo, dists)
        elif method == 'extrema':
            new_point, self.refined_points = self.refiner.extrema(self.refined_points)
        elif method == 'hybrid':
            new_point, self.refined_points = self.refiner.hybrid(self.refined_points,
                                                                 point_loo,
                                                                 next(self.hybrid),
                                                                 dists)
        elif method == 'optimization':
            new_point = self.refiner.optimization(extremum=extremum)
        elif method == 'sigma_discrepancy':
            new_point = self.refiner.sigma_discrepancy()

        # return added points
        points = np.atleast_2d(new_point)
        new_points = self.append(points)

        self.logger.info('Refined sampling with new point: {}'.format(new_points))
        self.logger.info('New discrepancy is {}'
                         .format(self.discrepancy(self.values, self.corners)))
        return new_points

    def optimization_results(self, extremum):
        """Compute the optimal value.

        :param str extremum: minimization or maximization objective
          ['min', 'max'].
        """
        sign = 1 if extremum == 'min' else -1
        gen = [self.refiner.func(x, sign=sign) for x in self.values]
        arg_extremum = np.argmin(gen)
        _extremum = sign * gen[arg_extremum]
        _x = self[arg_extremum]
        self.logger.info('New extremum value is: f(x)={} for x={}'.format(_extremum, _x))

        bounds = np.transpose(self.corners)
        results = differential_evolution(self.refiner.func, bounds, args=(sign,))
        _extremum = sign * results.fun
        _x = results.x
        self.logger.info('Optimization with surrogate: f(x)={} for x={}'.format(_extremum, _x))

    @staticmethod
    def discrepancy(sample, bounds=None, method='CD'):
        """Compute the discrepancy.

        Centered, wrap around or mixture discrepancy measures the uniformity
        of the parameter space. The lowest the value, the uniform the design.

        :param array_like sample: The sample to compute the discrepancy from
          (n_samples, k_vars).
        :param array_like bounds: Desired range of transformed data.
          The transformation apply the bounds on the sample and not the
          theoretical space, unit cube. Thus min and max values of the sample
          will coincide with the bounds. ([min, k_vars], [max, k_vars]).
        :param str method: Type of discrepancy. ['CD', 'WD', 'MD'].
        :return: Discrepancy.
        :rtype: float.
        """
        if bounds is not None:
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(bounds)
            sample = scaler.transform(sample)
        else:
            sample = np.asarray(sample)

        n_s, dim = sample.shape

        if method == 'CD':
            abs_ = abs(sample - 0.5)
            disc1 = np.sum(np.prod(1 + 0.5 * abs_ - 0.5 * abs_ ** 2, axis=1))

            prod_arr = 1
            for i in range(dim):
                s0 = sample[:, i]
                prod_arr *= (1 +
                             0.5 * abs(s0[:, None] - 0.5) + 0.5 * abs(s0 - 0.5) -
                             0.5 * abs(s0[:, None] - s0))
            disc2 = prod_arr.sum()

            disc = (13.0 / 12.0) ** dim - 2.0 / n_s * disc1 + 1.0 / (n_s ** 2) * disc2
        elif method == 'WD':
            prod_arr = 1
            for i in range(dim):
                s0 = sample[:, i]
                x_kikj = abs(s0[:, None] - s0)
                prod_arr *= 3.0 / 2.0 - x_kikj + x_kikj ** 2

            disc = - (4.0 / 3.0) ** dim + 1.0 / (n_s ** 2) * prod_arr.sum()
        elif method == 'MD':
            abs_ = abs(sample - 0.5)
            disc1 = np.sum(np.prod(5.0 / 3.0 - 0.25 * abs_ - 0.25 * abs_ ** 2, axis=1))

            prod_arr = 1
            for i in range(dim):
                s0 = sample[:, i]
                prod_arr *= (15.0 / 8.0 -
                             0.25 * abs(s0[:, None] - 0.5) - 0.25 * abs(s0 - 0.5) -
                             3.0 / 4.0 * abs(s0[:, None] - s0) +
                             0.5 * abs(s0[:, None] - s0) ** 2)
            disc2 = prod_arr.sum()

            disc = (19.0 / 12.0) ** dim - 2.0 / n_s * disc1 + 1.0 / (n_s ** 2) * disc2
        else:
            raise ValueError('Method {} is not valid. Options are CD or WD'
                             .format(method))

        return disc

    @staticmethod
    def mst(sample, fname=None, plot=True):
        """Minimum Spanning Tree.

        MST is used here as a discrepancy criterion.
        Comparing two different designs: the higher the mean,
        the better the design is in terms of space filling.

        :param array_like sample: The sample to compute the discrepancy from
          (n_samples, k_vars).
        :param str fname: whether to export to filename or display the figures.
        :return: Mean, standard deviation and edges of the MST.
        :rtypes: float, float, array_like (n_edges, 2 nodes indices).
        """
        sample = np.asarray(sample)

        dist = cdist(sample, sample)
        mst = minimum_spanning_tree(dist)

        edges = np.where(mst.toarray() > 0)
        edges = np.array(edges).T

        if (sample.shape[1] == 2) and plot:
            fig, ax = plt.subplots()
            for edge in edges:
                ax.plot(sample[edge, 0], sample[edge, 1], c='k')
            ax.scatter(sample[:, 0], sample[:, 1])

            visualization.save_show(fname, [fig])

        return mst.data.mean(), mst.data.std(), edges

    def _cheap_doe_from_expensive(self, n):
        """Compute the number of points required for the cheap DOE.

        :param int n: size of the expensive design.
        :return: size of the cheap design.
        :rtype: int.
        """
        doe_cheap = int((self.multifidelity[1] - n) * self.multifidelity[0])
        if doe_cheap / float(n) <= 1:
            self.logger.error('Nc/Ne must be positive')
            raise SystemExit
        self.max_points_nb = n + doe_cheap
        return doe_cheap

    # -----------------------------------------------------------
    # Method overwriting
    # -----------------------------------------------------------

    def append(self, points):
        """Add `points` to the space.

        Ignore any point that already exists or that would exceed space capacity.

        :param array_like points: Point(s) to add to space (n_samples, n_features)
        :return: Added points.
        :rtype: :class:`numpy.ndarray`
        """
        # avoid unnecessary operations
        if len(self) >= self.max_points_nb:
            self.logger.warning("Ignoring Points - Full Space - {}".format(points))
            return self

        if isinstance(points, Sample):
            # get values with columns in right order
            points = points.dataframe

        if isinstance(points, (pd.DataFrame, pd.Series)):
            try:
                points = points['space']
            except KeyError:
                pass
            # get values with columns in right order
            points = points[self.plabels].values

        # enforce 2D shape on points
        points = np.asarray(points)
        points = points.reshape(-1, points.shape[-1])

        # select only points in the space boundaries
        s = int(bool(self.multifidelity))  # drop 1st column during test if multifidelity
        dim = self.dim-self.nb_gp_samplers
        mask = np.logical_and(points[:, s:dim] >= self.corners[0][0:dim],
                              points[:, s:dim] <= self.corners[1][0:dim]).all(axis=1)
        if not np.all(mask):
            drop = np.logical_not(mask)
            self.logger.warning("Ignoring Points - Out of Space - {}".format(points[drop, :]))
        points = points[mask, :]

        # find new points to append
        if (not self.duplicate) and (len(self) > 0):
            existing_points = self.values[:, s:]
            test_points = points[:, s:]
            new_idx = []
            for i, point in enumerate(test_points):
                distances = np.linalg.norm(point - existing_points)
                if np.all(distances > self.threshold):
                    new_idx.append(i)
                    existing_points = np.append(existing_points, [point], axis=0)
            if not np.array_equal(new_idx, range(len(points))):
                drop = list(set(range(len(points))) - set(new_idx))
                self.logger.warning("Ignoring Points - Duplicate - {}".format(points[drop, :]))
            points = points[new_idx, :]

        # number of points that can be added
        if self.max_points_nb < np.inf:
            nbpoints = self.max_points_nb - len(self)
            if nbpoints < len(points):
                self.logger.warning("Ignoring Points - Full Space - {}"
                                    .format(points[nbpoints:, :]))
        else:
            nbpoints = None
        super(Space, self).append(points[:nbpoints, :])

        # return added points
        return points[:nbpoints, :]

    def read(self, path):
        """Read space from file `path`."""
        self.empty()
        super(Space, self).read(space_fname=path)
        self.logger.debug('Space read from {}'.format(path))

    def write(self, path='.', fname='space.dat', doe=True):
        """Write space to file `path`, then plot it."""
        space_file = os.path.join(path, fname)
        super(Space, self).write(space_fname=space_file)

        if doe:
            resampling = len(self) - self.doe_init
            visualization.doe(self, plabels=self.plabels, resampling=resampling,
                              multifidelity=self.multifidelity,
                              fname=os.path.join(path, 'DOE.pdf'))
        self.logger.debug('Space wrote to {}'.format(space_file))

    def __repr__(self):
        """Python Data Model. `str` function. Space string representation."""
        msg = ('Space summary:\n'
               'Hypercube points: {}\n'
               'Number of points: {}\n'
               'Max number of points: {}\n').format(self.corners, len(self), self.max_points_nb)
        return msg + super(Space, self).__repr__()
