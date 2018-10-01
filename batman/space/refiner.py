# coding: utf8
"""
Refinement Class
================

This class defines all resampling strategies that can be used.

It implements the following methods:

- :func:`Refiner.func`
- :func:`Refiner.func_sigma`
- :func:`Refiner.pred_sigma`
- :func:`Refiner.distance_min`
- :func:`Refiner.hypercube`
- :func:`Refiner.sigma`
- :func:`Refiner.leave_one_out_sigma`
- :func:`Refiner.leave_one_out_sobol`
- :func:`Refiner.extrema`
- :func:`Refiner.hybrid`
- :func:`Refiner.optimization`

:Example::

    >> corners = ((10, 400), (18, 450))
    >> resample = Refiner(pod, corners)
    >> new_point = resample.sigma()

"""
import logging
import copy
import warnings
from scipy.optimize import (differential_evolution, basinhopping)
from scipy.stats import norm
from scipy.spatial.distance import cdist
import numpy as np
from sklearn import preprocessing
import batman as bat
from .sampling import Doe
from ..misc import optimization


class Refiner:
    """Resampling the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, data, corners, delta_space=0.08, discrete=None, pod=None):
        """Initialize the refiner with the Surrogate and space corners.

        Points data are scaled between ``[0, 1]`` based on the size of the
        corners taking into account a :param:``delta_space`` factor.

        :param data: Surrogate or space
        :type data: :class:`batman.surrogate.SurrogateModel` or
          :class:`batman.space.Space`.
        :param array_like corners: hypercube ([min, n_features], [max, n_features]).
        :param float delta_space: Shrinking factor for the parameter space.
        :param int discrete: index of the discrete variable.
        :param pod: POD instance.
        :type pod: :class:`batman.pod.Pod`.
        """
        if isinstance(data, bat.surrogate.SurrogateModel):
            self.surrogate = data
        else:
            max_points_nb = data.shape[0]
            self.surrogate = bat.surrogate.SurrogateModel('kriging', data.corners, max_points_nb)
            self.space = data
            self.logger.debug("Using Space instance instead of SurrogateModel "
                              "-> restricted to discrepancy refiner")

        self.pod_S = 1 if pod is None else pod.S

        self.space = self.surrogate.space
        self.points = copy.deepcopy(self.space[:])

        self.discrete = discrete

        self.corners = np.array(corners).T
        self.dim = len(self.corners)

        # Prevent delta space contraction on discrete
        _dim = list(range(self.dim))
        if self.discrete is not None:
            _dim.pop(self.discrete)

        # Inner delta space contraction: delta_space * 2 factor
        for i in _dim:
            self.corners[i, 0] = self.corners[i, 0] + delta_space\
                * (self.corners[i, 1]-self.corners[i, 0])
            self.corners[i, 1] = self.corners[i, 1] - delta_space\
                * (self.corners[i, 1]-self.corners[i, 0])

        # Data scaling
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.corners.T)
        self.points = self.scaler.transform(self.points)

    def func(self, coords, sign=1):
        r"""Get the prediction for a given point.

        Retrieve Gaussian Process estimation. The function returns plus or
        minus the function depending on the sign.
        `-1` if we want to find the max and `1` if we want the min.

        :param lst(float) coords: coordinate of the point
        :param float sign: -1. or 1.
        :return: L2 norm of the function at the point
        :rtype: float
        """
        f, _ = self.surrogate(coords)
        modes_weights = np.array(self.pod_S ** 2).reshape(-1, 1)
        sum_f = np.average(f, weights=modes_weights)

        return sign * sum_f

    def pred_sigma(self, coords):
        """Prediction and sigma.

        Same as :func:`Refiner.func` and :func:`Refiner.func_sigma`.
        Function prediction and sigma are weighted using POD modes.

        :param lst(float) coords: coordinate of the point
        :returns: sum_f and sum_sigma
        :rtype: floats
        """
        f, sigma = self.surrogate(coords)
        modes_weights = np.array(self.pod_S ** 2).reshape(-1, 1)
        sum_f = np.average(f, weights=modes_weights)
        sum_sigma = np.average(sigma, weights=modes_weights)

        return sum_f, sum_sigma

    def distance_min(self, point):
        """Get the distance of influence.

        Compute the Chebyshev distance, max Linf norm between the anchor point
        and every sampling points. Linf allows to add this lenght to all
        coordinates and ensure that no points will be within this hypercube.
        It returns the minimal distance. :attr:`point` will be scaled by
        :attr:`self.corners` so the returned distance is scaled.

        :param array_like point: Anchor point.
        :return: The distance to the nearest point.
        :rtype: float.
        """
        point = self.scaler.transform(point.reshape(1, -1))[0]
        distances = cdist([point], self.points, 'chebyshev')[0]

        # Do not get itself
        distances = distances[np.nonzero(distances)]
        distance = min(distances)

        self.logger.debug("Distance min: {}".format(distance))

        return distance

    def hypercube_distance(self, point, distance):
        """Get the hypercube to add a point in.

        Propagate the distance around the anchor.
        :attr:`point` will be scaled by :attr:`self.corners` and input distance
        has to be already scalled. Ensure that new values are bounded by
        corners.

        :param array_like point: Anchor point.
        :param float distance: The distance of influence.
        :return: The hypercube around the point.
        :rtype: array_like.
        """
        point = self.scaler.transform(point.reshape(1, -1))[0]
        hypercube = np.array([point - distance, point + distance])
        hypercube = self.scaler.inverse_transform(hypercube)
        hypercube = hypercube.T
        self.logger.debug("Prior Hypercube:\n{}".format(hypercube))
        hypercube[:, 0] = np.minimum(hypercube[:, 0], self.corners[:, 1])
        hypercube[:, 0] = np.maximum(hypercube[:, 0], self.corners[:, 0])
        hypercube[:, 1] = np.minimum(hypercube[:, 1], self.corners[:, 1])
        hypercube[:, 1] = np.maximum(hypercube[:, 1], self.corners[:, 0])
        self.logger.debug("Post Hypercube:\n{}".format(hypercube))

        return hypercube

    def hypercube_optim(self, point):
        """Get the hypercube to add a point in.

        Compute the largest hypercube around the point based on the *L2-norm*.
        Ensure that only the *leave-one-out* point lies within it.
        Ensure that new values are bounded by corners.

        :param np.array point: Anchor point.
        :return: The hypercube around the point (a point per column).
        :rtype: array_like.
        """
        distance = self.distance_min(point) * 0.99
        x0 = self.hypercube_distance(point, distance).flatten('F')
        point = np.minimum(point, self.corners[:, 1])
        point = np.maximum(point, self.corners[:, 0])
        point = self.scaler.transform(point.reshape(1, -1))[0]

        gen = [p for p in self.points if not np.allclose(p, point)]

        def min_norm(hypercube):
            """Compute euclidean distance.

            :param np.array hypercube: [x1, y1, x2, y2, ...]
            :return: distance of between hypercube points
            :rtype: float
            """
            hypercube = hypercube.reshape(2, self.dim)
            try:
                hypercube = self.scaler.transform(hypercube)
            except ValueError:  # If the hypercube is nan
                return np.inf

            # Sort coordinates
            for i in range(self.dim):
                hypercube[:, i] = hypercube[hypercube[:, i].argsort()][:, i]

            hypercube = hypercube.T

            diff = hypercube[:, 1] - hypercube[:, 0]
            n = - np.linalg.norm(diff)

            # Check aspect ratio
            aspect = abs(diff)
            with np.errstate(divide='ignore', invalid='ignore'):
                aspect = abs(np.divide(np.power(np.max(aspect), self.dim),
                                       np.prod(aspect)))

            aspect = np.power(aspect, 1 / self.dim)
            if not aspect <= 1.5:
                return np.inf

            # Verify that LOO point is inside
            insiders = (hypercube[:, 0] <= point).all() & (point <= hypercube[:, 1]).all()
            if not insiders:
                return np.inf

            # Verify that no other point is inside
            insiders = np.array([True if (hypercube[:, 0] <= p).all() &
                                 (p <= hypercube[:, 1]).all() else False
                                 for p in gen]).any()
            if insiders:
                return np.inf

            return n

        bounds = np.reshape([self.corners] * 2, (self.dim * 2, 2))
        # results = differential_evolution(min_norm, bounds, popsize=100)
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        results = basinhopping(min_norm, x0,
                               niter=1000, minimizer_kwargs=minimizer_kwargs)
        hypercube = results.x.reshape(2, self.dim)
        for i in range(self.dim):
            hypercube[:, i] = hypercube[hypercube[:, i].argsort()][:, i]
        hypercube = hypercube.T

        self.logger.debug("Corners:\n{}".format(self.corners))
        self.logger.debug("Optimization Hypercube:\n{}".format(hypercube))

        return hypercube

    def discrepancy(self):
        """Find the point that minimize the discrepancy.

        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        self.logger.debug("Discrepancy strategy")
        init_discrepancy = bat.space.Space.discrepancy(self.space,
                                                       self.space.corners)

        @optimization(self.corners, self.discrete)
        def func_discrepancy(coords):
            """Discrepancy of the augmented space."""
            sample = np.vstack([self.space[:], coords])
            return bat.space.Space.discrepancy(sample)

        min_x, new_discrepancy = func_discrepancy()

        rel_change = (new_discrepancy - init_discrepancy) / init_discrepancy

        self.logger.debug("Relative change in discrepancy: {}%".format(rel_change))

        return min_x

    def sigma(self, hypercube=None):
        """Find the point at max Sigma.

        It returns the point where the variance (sigma) is maximum.
        To do so, it uses Gaussian Process information.
        A genetic algorithm get the global maximum of the function.

        :param array_like hypercube: Corners of the hypercube.
        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        if hypercube is None:
            hypercube = self.corners
        self.logger.debug("Sigma strategy")

        @optimization(hypercube, self.discrete)
        def func_sigma(coords):
            r"""Get the Sigma for a given point.

            Retrieve Gaussian Process estimation of sigma.
            A composite indicator is constructed using POD's modes.

            .. math:: \sum S_i^2 \times \sigma_i

            Function returns `- sum_sigma` in order to have a minimization
            problem.

            :param lst(float) coords: coordinate of the point.
            :return: - sum_sigma.
            :rtype: float.
            """
            _, sigma = self.surrogate(coords)
            sum_sigma = np.sum(self.pod_S ** 2 * sigma)

            return - sum_sigma

        min_x, _ = func_sigma()

        return min_x

    def leave_one_out_sigma(self, point_loo):
        """Mixture of Leave-one-out and Sigma.

        Estimate the quality of the POD by *leave-one-out cross validation*
        (LOOCV), and add a point arround the max error point.
        The point is added within an hypercube around the max error point.
        The size of the hypercube is equal to the distance with
        the nearest point.

        :param tuple point_loo: leave-one-out point.
        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        self.logger.info("Leave-one-out + Sigma strategy")
        # Get the point of max error by LOOCV
        point = np.array(point_loo)

        # Construct the hypercube around the point
        # distance = self.distance_min(point)
        # hypercube = self.hypercube_distance(point, distance)
        hypercube = self.hypercube_optim(point)

        # Global search of the point within the hypercube
        point = self.sigma(hypercube)

        return point

    def leave_one_out_sobol(self, point_loo, dists):
        """Mixture of Leave-one-out and Sobol' indices.

        Same as function :func:`leave_one_out_sigma` but change the shape
        of the hypercube. Using Sobol' indices, the corners are shrinked
        by the corresponding percentage of the total indices.

        :param tuple point_loo: leave-one-out point.
        :param lst(str) dists: List of valid openturns distributions as string.
        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        self.logger.info("Leave-one-out + Sobol strategy")
        # Get the point of max error by LOOCV
        point = np.array(point_loo)

        # Get Sobol' indices
        analyse = bat.uq.UQ(self.surrogate, dists=dists)
        indices = analyse.sobol()[2]
        indices = indices * (indices > 0)
        indices = preprocessing.normalize(indices.reshape(1, -1), norm='max')
        # Prevent indices inferior to 0.1
        indices[indices < 0.1] = 0.1

        # Construct the hypercube around the point
        hypercube = self.hypercube_optim(point)

        # Modify the hypercube with Sobol' indices
        for i in range(self.dim):
            hypercube[i, 0] = hypercube[i, 0] + (1 - indices[0, i])\
                * (hypercube[i, 1]-hypercube[i, 0]) / 2
            hypercube[i, 1] = hypercube[i, 1] - (1 - indices[0, i])\
                * (hypercube[i, 1]-hypercube[i, 0]) / 2

        self.logger.debug("Post Hypercube:\n{}".format(hypercube))

        # Global search of the point within the hypercube
        point = self.sigma(hypercube)

        return point

    def extrema(self, refined_points):
        """Find the min or max point.

        Using an anchor point based on the extremum value at sample points,
        search the hypercube around it. If a new extremum is found,it uses
        *Nelder-Mead* method to add a new point.
        The point is then bounded back by the hypercube.

        :return: The coordinate of the point to add
        :rtype: lst(float)
        """
        self.logger.info("Extrema strategy")
        points = np.delete(self.points, refined_points, 0)
        point = None
        new_points = []

        # Get max-max and max-min then min-max and min-min
        for sign in [-1., 1.]:
            self.logger.debug("Sign (-1 : Maximum ; 1 : Minimum) -> {}"
                              .format(sign))
            # Get a sample point where there is an extrema around
            while point is None:
                # Get min or max point
                evaluations = np.array([self.func(ref_point, sign)
                                        for _, ref_point in enumerate(points)])
                try:
                    min_idx = np.argmin(evaluations)
                except ValueError:
                    point = True
                    break
                point = points[min_idx]
                point_eval = min(evaluations) * sign
                self.logger.debug("Extremum located at sample point: {} -> {}"
                                  .format(point, point_eval))

                # Construct the hypercube around the point
                distance = self.distance_min(point)
                hypercube = self.hypercube_distance(point, distance)

                # Global search of the point within the hypercube
                first_extremum = differential_evolution(self.func,
                                                        hypercube,
                                                        args=(sign,))
                first_extremum.fun *= sign
                self.logger.debug("Optimization first extremum: {} -> {}"
                                  .format(first_extremum.x,
                                          first_extremum.fun))
                second_extremum = differential_evolution(self.func,
                                                         hypercube,
                                                         args=(-sign,))
                second_extremum.fun *= - sign
                self.logger.debug("Optimization second extremum: {} -> {}"
                                  .format(second_extremum.x,
                                          second_extremum.fun))

                # Check for new extrema, compare with the sample point
                if sign * first_extremum.fun < sign * point_eval:
                    # Nelder-Mead expansion
                    first_extremum = np.array([first_extremum.x +
                                               (first_extremum.x - point)])
                    # Constrain to the hypercube
                    first_extremum = np.maximum(first_extremum,
                                                hypercube[:, 0])
                    first_extremum = np.minimum(first_extremum,
                                                hypercube[:, 1])
                    new_points.append(first_extremum[0].tolist())
                    self.logger.debug("Extremum-max: {}"
                                      .format(first_extremum[0]))
                    if sign * second_extremum.fun > sign * point_eval:
                        second_extremum = np.array([second_extremum.x +
                                                    (second_extremum.x - point)])
                        second_extremum = np.maximum(second_extremum,
                                                     hypercube[:, 0])
                        second_extremum = np.minimum(second_extremum,
                                                     hypercube[:, 1])

                        if (second_extremum != first_extremum).all():
                            new_points.append(second_extremum[0].tolist())
                            self.logger.debug("Extremum-min: {}"
                                              .format(second_extremum[0]))
                        else:
                            self.logger.debug("Extremum-min egal: not added.")
                else:
                    point = None

                points = np.delete(points, min_idx, 0)

            point = None
            refined_points.append(min_idx)

        return new_points, refined_points

    def hybrid(self, refined_points, point_loo, method, dists):
        """Composite resampling strategy.

        Uses all methods one after another to add new points.
        It uses the navigator defined within settings file.

        :param lst(int) refined_points: points idx not to consider for extrema
        :param point_loo: leave one out point
        :type point_loo: :class:`batman.space.point.Point`
        :param str strategy: resampling method
        :param lst(str) dists: List of valid openturns distributions as string.
        :return: The coordinate of the point to add
        :rtype: lst(float)
        """
        self.logger.info(">>---Hybrid strategy---<<")

        if method == 'sigma':
            new_point = self.sigma()
        elif method == 'sigma':
            new_point = self.discrepancy()
        elif method == 'loo_sigma':
            new_point = self.leave_one_out_sigma(point_loo)
        elif method == 'loo_sobol':
            new_point = self.leave_one_out_sobol(point_loo, dists)
        elif method == 'extrema':
            new_point, refined_points = self.extrema(refined_points)
        elif method == 'discrepancy':
            new_point = self.discrepancy()
        elif method == 'optimization':
            new_point = self.optimization()
        else:
            self.logger.exception("Resampling method does't exits")
            raise SystemExit

        return new_point, refined_points

    def optimization(self, method='EI', extremum='min'):
        """Maximization of the Probability/Expected Improvement.

        :param str method: Flag ['EI', 'PI'].
        :param str extremum: minimization or maximization objective
          ['min', 'max'].
        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        sign = 1 if extremum == 'min' else -1
        gen = [self.func(x, sign=sign)
               for x in self.scaler.inverse_transform(self.points)]
        arg_min = np.argmin(gen)
        min_value = gen[arg_min]
        min_x = self.points[arg_min]
        self.logger.info('Current extrema value is: f(x)={} for x={}'
                         .format(sign * min_value, min_x))

        # Do not check point is close on the discrete parameter
        if self.discrete is not None:
            no_discrete = [list(range(self.points.shape[1]))]
            no_discrete[0].pop(self.discrete)
        else:
            no_discrete = None

        @optimization(self.corners, self.discrete)
        def probability_improvement(x):
            """Do probability of improvement."""
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            too_close = np.array([True if np.linalg.norm(
                x_scaled[0][no_discrete] - p[no_discrete], -1) < 0.02
                                  else False for p in self.points]).any()
            if too_close:
                return np.inf

            pred, sigma = self.pred_sigma(x)
            pred = sign * pred
            std_dev = np.sqrt(sigma)
            pi = norm.cdf((target - pred) / std_dev)

            return - pi

        @optimization(self.corners, self.discrete)
        def expected_improvement(x):
            """Do expected improvement."""
            x_scaled = self.scaler.transform(x.reshape(1, -1))
            too_close = np.array([True if np.linalg.norm(
                x_scaled[0][no_discrete] - p[no_discrete], -1) < 0.02
                                  else False for p in self.points]).any()
            if too_close:
                return np.inf

            pred, sigma = self.pred_sigma(x)
            pred = sign * pred
            std_dev = np.sqrt(sigma)
            diff = min_value - pred
            ei = diff * norm.cdf(diff / std_dev)\
                + std_dev * norm.pdf(diff / std_dev)

            return - ei

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == 'PI':
                target = min_value - 0.1 * np.abs(min_value)
                max_ei, _ = probability_improvement()
            else:
                max_ei, _ = expected_improvement()

        return max_ei

    def sigma_discrepancy(self, weights=None):
        """Maximization of the composite indicator: sigma - discrepancy.

        :param list(float) weights: respectively weights of sigma and discrepancy.
        :return: The coordinate of the point to add.
        :rtype: lst(float).
        """
        weights = [0.5, 0.5] if weights is None else weights
        doe = Doe(500, self.corners, 'halton')
        sample = doe.generate()

        _, sigma = zip(*[self.pred_sigma(s) for s in sample])

        disc = [1 / bat.space.Space.discrepancy(np.vstack([self.space, p]),
                                                self.space.corners)
                for p in sample]

        sigma = np.array(sigma).reshape(-1, 1)
        disc = np.array(disc).reshape(-1, 1)

        scale_sigma = preprocessing.StandardScaler().fit(sigma)
        scale_disc = preprocessing.StandardScaler().fit(disc)

        @optimization(self.corners, self.discrete)
        def f_obj(x):
            """Maximize the inverse of the discrepancy plus sigma."""
            _, sigma = self.pred_sigma(x)
            sigma = scale_sigma.transform(sigma.reshape(1, -1))

            disc = 1 / bat.space.Space.discrepancy(np.vstack([self.space, x]),
                                                   self.space.corners)
            disc = scale_disc.transform(disc.reshape(1, -1))

            sigma_disc = sigma * weights[0] + disc * weights[1]

            return - sigma_disc

        max_sigma_disc, _ = f_obj()

        return max_sigma_disc
