# -*- coding: utf-8 -*-
"""Resampling the space of parameters."""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import logging
from scipy.optimize import differential_evolution
import numpy as np
import copy
from uq import UQ


class Refiner():

    """
    Refinement Class
    ================

    This class defines all resampling strategies that can be used.

    It implements the following methods:

    - :func:`func`
    - :func:`func_mse`
    - :func:`distance_min`
    - :func:`hypercube`
    - :func:`mse`
    - :func:`leave_one_out_mse`
    - :func:`leave_one_out_sobol`
    - :func:`extrema`
    - :func:`composite`

    :Example:

    >> corners = ((10, 400), (18, 450))
    >> resample = Refiner(pod, corners)
    >> new_point = resample.mse()

    References
    ----------

    C. Scheidt: Analyse statistique d'expériences simulées : Modélisation adaptative de réponses non régulières par Krigeage et plans d'expériences, Application à la quantification des incertitudes en ingénierie des réservoirs pétroliers. Université Louis Pasteur. 2006

    """

    logger = logging.getLogger(__name__)

    def __init__(self, pod, settings, corners):
        """Initialize the refiner with the POD and space corners.

        :param pod: POD
        :param settings: JPOD parameters
        :param tuple(tuple(float)) corners: Boundaries to add a point within
        """
        self.pod = pod
        self.points = copy.deepcopy(pod.points)
        self.kind = settings.prediction['method']
        self.settings = settings
        self.corners = np.array(corners).T

    def func(self, coords, sign):
        r"""Get the prediction for a given point.

        Retrieve the Gaussian Process estimation of sigma: the mean square error.
        A composite indicator is constructed using POD's modes.

        .. math:: \sum S_i^2 \times f_i

        The function returns plus or minus the function depending on the sign.
        -1 if we want to find the max and 1 if we want the min.

        :param lst(float) coords: coordinate of the point
        :param float sign: -1. or 1.
        :return: L2 norm of the function at the point
        :rtype: float

        """
        f, _ = self.pod.predict(self.kind, [coords])
        try:
            _, f = np.split(f[0].data, 2)
        except:
            f = f[0].data
        #sum_f = np.sum(self.pod.S ** 2 * f)
        sum_f = np.sum(f)

        return sign * sum_f

    def func_mse(self, coords):
        r"""Get the MSE for a given point.

        Retrieve the Gaussian Process estimation of sigma: the mean square error.
        A composite indicator is constructed using POD's modes.

        .. math:: \sum S_i^2 \times \sigma_i

        The function returns - sum_sigma in order to have a minimization problem.

        :param lst(float) coords: coordinate of the point
        :return: - sum_sigma
        :rtype: float

        """
        _, sigma = self.pod.predict(self.kind, [coords])
        sum_sigma = np.sum(self.pod.S ** 2 * sigma)

        return - sum_sigma

    def distance_min(self, point):
        """Get the distance of influence.

        Compute the distance, L2 norm between the anchor point and every sampling points.
        It returns the minimal distance.

        :param np.array point: Anchor point
        :return: The distance to the nearest point
        :rtype: float

        """
        distances = np.array([np.linalg.norm(pod_point - point) for _, pod_point in enumerate(self.points)])
        distances = distances[np.nonzero(distances)]
        distance = min(distances)  # * 3 / 2
        self.logger.debug("Distance min: {}".format(distance))

        return distance

    def hypercube(self, point, distance):
        """Get the hypercube to add a point in.

        Propagate the distance around the anchor.
        Ensure that new values are bounded by corners.

        :param np.array point: Anchor point
        :param float distance: The distance of influence
        :return: The hypercube around the point
        :rtype: np.array

        """
        hypercube = np.array([point - distance, point + distance]).T
        self.logger.debug("Prior Hypercube:\n{}".format(hypercube))
        self.logger.debug("Corners:\n{}".format(self.corners))
        hypercube[:, 0] = np.maximum(hypercube[:, 0], self.corners[:, 0])
        hypercube[:, 1] = np.minimum(hypercube[:, 1], self.corners[:, 1])
        self.logger.debug("Post Hypercube:\n{}".format(hypercube))

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

    def leave_one_out_mse(self):
        """Mixture of Leave-one-out and MSE.

        Estimate the quality of the POD by *leave-one-out cross validation* (LOOCV), and add a point arround the max error point.
        The point is added within an hypercube around the max error point.
        The size of the hypercube is equal to the distance with the nearest point.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info("Leave-one-out + MSE strategy")
        # Get the point of max error by LOOCV
        _, point = self.pod.estimate_quality()
        point = np.array(point)

        # Construct the hypercube around the point
        distance = self.distance_min(point)
        hypercube = self.hypercube(point, distance)

        # Global search of the point within the hypercube
        point = self.mse(hypercube)

        return point

    def leave_one_out_sobol(self):
        """Mixture of Leave-one-out and Sobol' indices.

        Same as function :func:`leave_one_out_mse` but change the shape of the hypercube.
        Using Sobol' indices, the corners are shrinked by the corresponding percentage of the total indices.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        self.logger.info("Leave-one-out + Sobol strategy")
        # Get the point of max error by LOOCV
        _, point = self.pod.estimate_quality()
        point = np.array(point)

        # Get Sobol' indices
        analyse = UQ(self.pod, self.settings)
        indices = analyse.sobol()

        # Modify min distance with Sobol' indices
        distance = self.distance_min(point)
        distance = distance * indices[2]
        self.logger.debug("Post Distance min: {}".format(distance))

        # Construct the hypercube around the point
        hypercube = self.hypercube(point, distance)

        # Global search of the point within the hypercube
        point = self.mse(hypercube)

        return point

    def extrema(self, refined_pod_points):
        """Find the min or max point.

        Using an anchor point based on the extremum value at sample points,
        search the hypercube around it. If a new extremum is found, return a point.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        # Get the kind of extrema to compute
        self.logger.info("Extrema strategy")

        try:
            self.points = np.delete(self.points, refined_pod_points, 0)
        except:
            self.logger.debug("No point to delete")

        point = None
        new_points = []

        # Get max-max and max-min then min-max and min-min
        for sign in [-1., 1.]:
            self.logger.debug("Sign: {}".format(sign))
            # Get a sample point where there is an extrema around
            while point is None:
                # Get min or max point
                evaluations = np.array([self.func(pod_point, sign) for _, pod_point in enumerate(self.points)])
                min_idx = np.argmin(evaluations)
                point = self.points[min_idx]
                point_eval = sign * min(evaluations)
                self.logger.debug("Extremum located at sample point: {} -> {}".format(point, point_eval))

                # Construct the hypercube around the point
                distance = self.distance_min(point)
                hypercube = self.hypercube(point, distance)

                # Global search of the point within the hypercube
                first_extremum = differential_evolution(self.func, hypercube, args=(sign,))
                self.logger.debug("Optimization first extremum: {} -> {}".format(first_extremum.x, sign * first_extremum.fun))
                second_extremum = differential_evolution(self.func, hypercube, args=(-sign,))
                self.logger.debug("Optimization second extremum: {} -> {}".format(second_extremum.x, - sign * second_extremum.fun))

                if sign == -1.:
                    if sign * first_extremum.fun > point_eval:
                        #first_extremum = np.array([first_extremum.x + (first_extremum.x - point)])
                        first_extremum = np.array([first_extremum.x])
                        first_extremum = np.maximum(first_extremum, self.corners[:, 0])
                        first_extremum = np.minimum(first_extremum, self.corners[:, 1])
                        new_points.append(first_extremum[0].tolist())
                        if - sign * second_extremum.fun < point_eval:
                            #second_extremum = np.array([second_extremum.x + (second_extremum.x - point)])
                            second_extremum = np.array([second_extremum.x])
                            second_extremum = np.maximum(second_extremum, self.corners[:, 0])
                            second_extremum = np.minimum(second_extremum, self.corners[:, 1])
                            new_points.append(second_extremum[0].tolist())
                    else:
                        point = None
                else:
                    if sign * first_extremum.fun < point_eval:
                        #first_extremum = np.array([first_extremum.x + (first_extremum.x - point)])
                        first_extremum = np.array([first_extremum.x])
                        first_extremum = np.maximum(first_extremum, self.corners[:, 0])
                        first_extremum = np.minimum(first_extremum, self.corners[:, 1])
                        new_points.append(first_extremum[0].tolist())
                        if - sign * second_extremum.fun > point_eval:
                            #second_extremum = np.array([second_extremum.x + (second_extremum.x - point)])
                            second_extremum = np.array([second_extremum.x])
                            second_extremum = np.maximum(second_extremum, self.corners[:, 0])
                            second_extremum = np.minimum(second_extremum, self.corners[:, 1])
                            new_points.append(second_extremum[0].tolist())
                    else:
                        point = None

                # new_points.append(point + (point - result.x)) if result.fun < self.func(point, sign) else None
                self.points = np.delete(self.points, min_idx, 0)
                self.logger.debug("New points: {}".format(new_points))
            point = None
                
            refined_pod_points.append(min_idx)

        self.logger.debug("\nMax-max: {}\nMax-min: {}\nMin-max: {}\nMin-min: {}".format(new_points[0], new_points[1], new_points[2], new_points[3]))

        return new_points, refined_pod_points

    def composite(self):
        """Composite resampling strategy.

        Uses all methods one after the other to add new points.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
