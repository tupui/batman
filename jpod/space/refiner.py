# coding: utf8
"""
Refinement Class
================

This class defines all resampling strategies that can be used.

It implements the following methods:

- :func:`Refiner.func`
- :func:`Refiner.func_mse`
- :func:`Refiner.distance_min`
- :func:`Refiner.hypercube`
- :func:`Refiner.mse`
- :func:`Refiner.leave_one_out_mse`
- :func:`Refiner.leave_one_out_sobol`
- :func:`Refiner.extrema`
- :func:`Refiner.hybrid`

:Example::

    >> corners = ((10, 400), (18, 450))
    >> resample = Refiner(pod, corners)
    >> new_point = resample.mse()

References
----------

C. Scheidt: Analyse statistique d'expériences simulées : Modélisation adaptative de réponses non régulières par Krigeage et plans d'expériences, Application à la quantification des incertitudes en ingénierie des réservoirs pétroliers. Université Louis Pasteur. 2006

"""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import logging
from scipy.optimize import (differential_evolution, minimize, basinhopping)
import numpy as np
from sklearn import preprocessing
import copy
from collections import OrderedDict
from ..uq import UQ
import jpod.pod


class Refiner(object):

    """Resampling the space of parameters."""

    logger = logging.getLogger(__name__)

    def __init__(self, pod, settings):
        """Initialize the refiner with the POD and space corners.

        Points data are scaled between ``[0, 1]`` based on the size of the
        corners taking into account a ``delta_space`` factor.

        :param jpod.pod.pod.Pod pod: POD
        :param dict settings: parameters
        :param tuple(tuple(float)) corners: Boundaries to add a point within
        """
        self.points = copy.deepcopy(pod.points)
        self.pod = pod
        if self.pod.predictor is None:
            self.pod.predictor = jpod.pod.Predictor("kriging", self.pod)

        self.settings_full = settings
        self.settings = settings['space']
        self.corners = np.array(self.settings['corners']).T
        delta_space = self.settings['resampling']['delta_space']
        self.dim = len(self.corners)

        # Inner delta space contraction: delta_space * 2 factor
        for i in range(self.dim):
            self.corners[i, 0] = self.corners[i, 0] + delta_space\
                * (self.corners[i, 1]-self.corners[i, 0])
            self.corners[i, 1] = self.corners[i, 1] - delta_space\
                * (self.corners[i, 1]-self.corners[i, 0])

        # Data scaling
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.corners.T)
        self.points = self.scaler.transform(self.points)

    def func(self, coords, sign):
        r"""Get the prediction for a given point.

        Retrieve Gaussian Process estimation. The function returns plus or
        minus the function depending on the sign.
        `-1` if we want to find the max and `1` if we want the min.

        :param lst(float) coords: coordinate of the point
        :param float sign: -1. or 1.
        :return: L2 norm of the function at the point
        :rtype: float

        """
        f, _ = self.pod.predictor([coords])
        try:
            _, f = np.split(f[0].data, 2)
        except:
            f = f[0].data
        # sum_f = np.sum(self.pod.S ** 2 * f)
        sum_f = np.sum(f)

        return sign * sum_f

    def func_mse(self, coords):
        r"""Get the MSE for a given point.

        Retrieve Gaussian Process estimation of sigma.
        A composite indicator is constructed using POD's modes.

        .. math:: \sum S_i^2 \times \sigma_i

        Function returns `- sum_sigma` in order to have a minimization problem.

        :param lst(float) coords: coordinate of the point
        :return: - sum_sigma
        :rtype: float

        """
        _, sigma = self.pod.predictor([coords])
        sum_sigma = np.sum(self.pod.S ** 2 * sigma)

        return - sum_sigma

    def distance_min(self, point):
        """Get the distance of influence.

        Compute the distance, Linf norm between the anchor point and
        every sampling points. Linf allows to add this lenght to all 
        coordinates and ensure that no points will be within this hypercube.
        It returns the minimal distance. ``point`` needs to be scaled by ``self.corners`` so the returned distance is scaled.

        :param np.array point: Anchor point
        :return: The distance to the nearest point
        :rtype: float

        """
        point = self.scaler.transform(point.reshape(1, -1))[0]
        distances = np.array([np.linalg.norm(pod_point - point)
                              for _, pod_point in enumerate(self.points)])
        # Do not get itself
        distances = distances[np.nonzero(distances)]
        distance = min(distances)
        self.logger.debug("Distance min: {}".format(distance))

        return distance

    def hypercube_distance(self, point, distance):
        """Get the hypercube to add a point in.

        Propagate the distance around the anchor.
        ``point`` is scaled by ``self.corners`` and input distance has to be.
        Ensure that new values are bounded by corners.

        :param np.array point: Anchor point
        :param float distance: The distance of influence
        :return: The hypercube around the point
        :rtype: np.array

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

        Compute the largest hypercube around the point based on the `L2-norm`.
        Ensure that only the `leave-one-out` point lies within it.
        Ensure that new values are bounded by corners.

        :param np.array point: Anchor point
        :return: The hypercube around the point (a point per column)
        :rtype: np.array

        """
        distance = self.distance_min(point) / 3
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

            n = - np.linalg.norm(hypercube[:, 1] - hypercube[:, 0])

            # Check aspect ratio
            aspect = abs(hypercube[:, 1] - hypercube[:, 0])
            aspect = np.power(np.max(aspect), self.dim) / np.prod(aspect)
            aspect = np.power(aspect, 1 / self.dim)
            if not (aspect <= 1.5):
                return np.inf

            # Verify that LOO point is inside
            insiders = (hypercube[:, 0] <= point).all() & (point <= hypercube[:, 1]).all()
            if not insiders:
                return np.inf

            # Verify that no other point is inside
            for p in gen:
                insiders = (hypercube[:, 0] <= p).all() & (p <= hypercube[:, 1]).all()
                if insiders:
                    return np.inf

            return n

        bounds = np.reshape([self.corners] * 2, (self.dim * 2, 2))
        # results = differential_evolution(min_norm, bounds, popsize=100)
        # results = minimize(min_norm, x0, method='L-BFGS-B', bounds=bounds)
        # results = minimize(min_norm, x0, method='SLSQP', bounds=bounds)
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        results = basinhopping(min_norm, x0, niter=1000, minimizer_kwargs=minimizer_kwargs)
        hypercube = results.x.reshape(2, self.dim)
        for i in range(self.dim):
                hypercube[:, i] = hypercube[hypercube[:, i].argsort()][:, i]
        hypercube = hypercube.T

        self.logger.debug("Corners:\n{}".format(self.corners))
        self.logger.debug("Optimization Hypercube:\n{}".format(hypercube))

        return hypercube

    def mse(self, hypercube=None):
        """Find the point at max MSE.

        It returns the point where the mean square error (sigma) is maximum.
        To do so, it uses Gaussian Process information.
        A genetic algorithm get the global maximum of the function.

        :param np.array hypercube: Corners of the hypercube
        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        if hypercube is None:
            hypercube = self.corners
        self.logger.debug("MSE strategy")
        result = differential_evolution(self.func_mse, hypercube)

        return result.x

    def leave_one_out_mse(self, point_loo):
        """Mixture of Leave-one-out and MSE.

        Estimate the quality of the POD by *leave-one-out cross validation*
        (LOOCV), and add a point arround the max error point.
        The point is added within an hypercube around the max error point.
        The size of the hypercube is equal to the distance with
        the nearest point.

        :param tuple point_loo: leave-one-out point
        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info("Leave-one-out + MSE strategy")
        # Get the point of max error by LOOCV
        point = np.array(point_loo)

        # Construct the hypercube around the point
        # distance = self.distance_min(point)
        # hypercube = self.hypercube_distance(point, distance)
        hypercube = self.hypercube_optim(point)

        # Global search of the point within the hypercube
        point = self.mse(hypercube)

        return point

    def leave_one_out_sobol(self, point_loo):
        """Mixture of Leave-one-out and Sobol' indices.

        Same as function :func:`leave_one_out_mse` but change the shape
        of the hypercube. Using Sobol' indices, the corners are shrinked
        by the corresponding percentage of the total indices.

        :param tuple point_loo: leave-one-out point
        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info("Leave-one-out + Sobol strategy")
        # Get the point of max error by LOOCV
        point = np.array(point_loo)

        # Get Sobol' indices
        analyse = UQ(self.pod, self.settings_full)
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
        point = self.mse(hypercube)

        return point

    def extrema(self, refined_pod_points):
        """Find the min or max point.

        Using an anchor point based on the extremum value at sample points,
        search the hypercube around it. If a new extremum is found,it uses
        Nelder-Mead method to add a new point.
        The point is then bounded back by the hypercube.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info("Extrema strategy")
        points = np.delete(self.points, refined_pod_points, 0)
        point = None
        new_points = []

        # Get max-max and max-min then min-max and min-min
        for sign in [-1., 1.]:
            self.logger.debug("Sign (-1 : Maximum ; 1 : Minimum) -> {}"
                              .format(sign))
            # Get a sample point where there is an extrema around
            while point is None:
                # Get min or max point
                evaluations = np.array([self.func(pod_point, sign)
                                        for _, pod_point in enumerate(points)])
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
            refined_pod_points.append(min_idx)

        return new_points, refined_pod_points

    def hybrid(self, refined_pod_points, point_loo):
        """Composite resampling strategy.

        Uses all methods one after another to add new points.
        It uses the navigator defined within settings file.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info(">>---Hybrid strategy---<<")

        try:
            self.logger.debug('Strategy: {}'.format(self.settings['resampling']['strategy_full'])) 
        except KeyError:
            self.settings['resampling']['strategy_full'] = self.settings['resampling']['hybrid']
            self.logger.info('Strategy: {}'
                             .format(self.settings['resampling']['strategy_full']))

        self.settings['resampling']['hybrid'] = OrderedDict(self.settings['resampling']['hybrid'])
        strategies = self.settings['resampling']['hybrid']

        if sum(strategies.values()) == 0:
            self.settings['resampling']['hybrid'] = OrderedDict(self.settings['resampling']['strategy_full'])
            strategies = self.settings['resampling']['hybrid']

        new_point = []
        for method in strategies:
            if strategies[method] > 0:
                if method == 'sigma':
                    new_point = self.mse()
                    break
                elif method == 'loo_sigma':
                    new_point = self.leave_one_out_mse(point_loo)
                    break
                elif method == 'loo_sobol':
                    new_point = self.leave_one_out_sobol(point_loo)
                    break
                elif method == 'extrema':
                    new_point, refined_pod_points = self.extrema(refined_pod_points)
                    break
                else:
                    self.logger.exception("Resampling method does't exits")
                    raise SystemExit

        self.settings['resampling']['hybrid'][method] -= 1

        return new_point, refined_pod_points
