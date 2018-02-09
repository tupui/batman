# coding: utf8
"""
[TODO]
"""
from copy import copy
import logging
import os
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn import preprocessing
from .sampling import Doe
from .sample import Sample
from .refiner import Refiner
from .. import visualization


class Space(Sample):
    """[TODO]
    """

    logger = logging.getLogger(__name__)

    def __init__(self, corners, sample=np.inf, nrefine=0, plabels=None, psizes=None,
                 multifidelity=None, duplicate=False, threshold=0.):
        """[TODO]
        """
        try:
            self.doe_init = len(sample)
        except TypeError:
            self.doe_init = sample

        self.max_points_nb = self.doe_init
        if nrefine > 0:
            self.refiner = None
            self.max_points_nb += nrefine
        self.refined_pod_points = []

        self.dim = len(corners[0])
        self.multifidelity = multifidelity
        self.duplicate = duplicate
        self.threshold = threshold

        # Parameter names list
        if plabels is None:
            plabels = ['x{}'.format(i) for i in range(self.dim)]
        if psizes is None:
            psizes = [1] * self.dim
        try:
            pos = plabels.index('fidelity')
        except ValueError:
            pass
        else:
            plabels = list(np.delete(plabels, pos))
            psizes = list(np.delete(psizes, pos))

        # Multifidelity configuration
        if multifidelity is not None:
            self.doe_cheap = self._cheap_doe_from_expensive(self.doe_init)
            plabels = ['fidelity'] + plabels
            psizes = ['fidelity'] + psizes
            self.logger.info('Multifidelity with Ne: {} and Nc: {}'
                             .format(self.doe_init, self.doe_cheap))

        # Corner points
        self.corners = np.array(corners)
        if np.any(self.corners[0] == self.corners[1]):
            raise ValueError('corners coordinates at positions {} are equal'
                             .format(np.flatnonzero(self.corners[0] == self.corners[1])))

        # Initialize Sample container with empty space dataframe
        super().__init__(plabels=plabels)

    def sampling(self, n_samples=None, kind='halton', dists=None, discrete=None):
        """[TODO]
        """
        if n_samples is None:
            n_samples = self.doe_init
        if self.multifidelity:
            n_samples = self._cheap_doe_from_expensive(n_samples)
        doe = Doe(n_samples, self.corners, kind, dists, discrete)
        samples = doe.generate()

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
        self.logger.info("Discrepancy is {}".format(self.discrepancy()))
        return self.values

    def refine(self, surrogate, method, point_loo=None, delta_space=0.08,
               dists=None, hybrid=None, discrete=None, extremum='min'):
        """[TODO]
        """
        try:
            self.refiner
        except AttributeError:
            return np.empty((0, len(self.plabels)))

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
            new_point = self.refiner.optimization(extremum=extremum)
        elif method == 'sigma_discrepancy':
            new_point = self.refiner.sigma_discrepancy()

        # return added points
        points = np.atleast_2d(new_point)
        new_points = self.append(points)

        self.logger.info('Refined sampling with new point: {}'.format(new_points))
        self.logger.info("New discrepancy is {}".format(self.discrepancy()))
        return new_points
        
    def optimization_results(self, extremum):
        """[TODO]
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

    def discrepancy(self, sample=None):
        """[TODO]
        """
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(self.corners)
        if sample is None:
            sample = scaler.transform(self.values)
        else:
            sample = scaler.transform(sample)
        
        abs_ = abs(sample - 0.5)
        disc1 = np.sum(np.prod(1 + 0.5 * abs_ - 0.5 * abs_ ** 2, axis=1))

        prod_arr = 1
        for i in range(self.dim):
            s0 = sample[:, i]
            prod_arr *= (1 +
                         0.5 * abs(s0[:, None] - 0.5) + 0.5 * abs(s0 - 0.5) -
                         0.5 * abs(s0[:, None] - s0))
        disc2 = prod_arr.sum()

        n_s = len(sample)
        c2 = (13.0 / 12.0) ** self.dim - 2.0 / n_s * disc1 + 1.0 / (n_s ** 2) * disc2
        return c2

    def _cheap_doe_from_expensive(self, n):
        """[TODO]
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
        """[TODO]
        """
        # avoid unnecessary operations
        if len(self) >= self.max_points_nb:
            self.logger.warning("Ignoring Points - Full Space - {}".format(points))
            return self
        
        if isinstance(points, Sample):
            # get values with columns in right order
            points = points.dataframe

        if isinstance(points, pd.DataFrame) or isinstance(points, pd.Series):
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
        mask = np.logical_and(points >= self.corners[0], points <= self.corners[1]).all(axis=1)
        if not np.all(mask):
            drop = np.logical_not(mask)
            self.logger.warning("Ignoring Points - Out of Space - {}".format(points[drop, :]))
        points = points[mask, :]

        # find new points to append
        if not self.duplicate and len(self) > 0:
            s = int(bool(self.multifidelity))  # drop 1st column during test if multifidelity
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
                self.logger.warning("Ignoring Points - Full Space - {}".format(points[nbpoints:, :]))
        else:
            nbpoints = None
        super().append(points[:nbpoints, :])

        # return added points
        return points[:nbpoints, :]

    def read(self, path):
        """[TODO]"""
        self.empty()
        super().read(space_fname=path)
        self.logger.debug('Space read from {}'.format(path))

    def write(self, path='.'):
        """[TODO]"""
        super().write(space_fname=os.path.join(path, 'space.dat'))
        resampling = len(self) - self.doe_init
        path = os.path.join(path, 'DOE.pdf')
        visualization.doe(self, plabels=self.plabels, resampling=resampling,
                          multifidelity=self.multifidelity, fname=path)
        self.logger.debug('Space wrote to {}'.format(path))

    def __str__(self):
        """[TODO]"""
        msg = ('Space summary:\n'
               'Hypercube points: {}\n'
               'Number of points: {}\n'
               'Max number of points: {}\n\n'
              ).format(self.corners, len(self), self.max_points_nb)
        return msg + super().__str__()
