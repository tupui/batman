# -*- coding: utf-8 -*-
"""Resampling the space of parameters."""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import logging
from scipy.optimize import differential_evolution
import numpy as np
from uq import UQ


class Refiner():

    """
    Refinement Class
    ================

    This class defines all resampling strategies that can be used.

    It implements the following methods:

    - :func:`mse`
    - :func:`leave_one_out_mse`
    - :func:`leave_one_out_sobol`

    :Example:

    >> corners = ((10, 400), (18, 450))
    >> resample = Refiner(pod, corners)
    >> new_point = resample.mse()

    """

    logger = logging.getLogger(__name__)

    def __init__(self, pod, settings, corners):
        """Initialize the refiner with the POD and space corners.

        :param pod: POD
        :param settings: JPOD parameters
        :param tuple(tuple(float)) corners: Boundaries to add a point within
        """
        self.pod = pod
        self.kind = settings.prediction['method']
        self.settings = settings
        self.corners = np.array(corners).T
        self.point = None

    def func(self, coords):
        r"""Get the MSE for a given point.

        Retrieve the Gaussian Process estimation of sigma: the mean square error.
        A composite indicator is constructed using POD's modes.

        .. math:: \sum S^2 \times \sigma

        The function returns - sum_sigma in order to have a minimization problem.

        :param lst(float) coords: coordinate of the point
        :return: - sum_sigma
        :rtype: float

        """
        _, sigma = self.pod.predict(self.kind, [coords])
        sum_sigma = np.sum(self.pod.S ** 2 * sigma)

        return - sum_sigma

    def mse(self):
        """Find the point at max MSE.

        It returns the point where the mean square error (sigma) is maximum.
        To do so, it uses Gaussian Process information.
        A genetic algorithm get the global maximum of the function.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        result = differential_evolution(self.func, self.corners)

        return result.x

    def leave_one_out_mse(self):
        """Mixture of Leave-one-out and MSE.

        Estimate the quality of the POD by *leave-one-out cross validation* (LOOCV), and add a point arround the max error point.
        The point is added within an hypercube around the max error point.
        The size of the hypercube is equal to the distance with the nearest point.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        # Get the point of max error by LOOCV
        _, point = self.pod.estimate_quality()
        point = np.array(point)

        # Determine the distance between the point and all other points in Space
        distances = np.array([np.linalg.norm(self.pod.points[i] - point) for i in range(len(self.pod.points))])
        distances = distances[np.nonzero(distances)]
        distance = min(distances) / 2
        self.logger.debug("Distance min: {}".format(distance))

        # Construct the hypercube around the point
        hypercube = np.array([point - distance, point + distance]).T
        self.logger.debug("Prior Hypercube:\n{}".format(hypercube))
        self.logger.debug("Corners:\n{}".format(self.corners))
        hypercube[:, 0] = np.maximum(hypercube[:, 0], self.corners[:, 0])
        hypercube[:, 1] = np.minimum(hypercube[:, 1], self.corners[:, 1])
        self.logger.debug("Post Hypercube:\n{}".format(hypercube))

        # Global search of the point within the hypercube
        result = differential_evolution(self.func, hypercube)

        return result.x

    def leave_one_out_sobol(self):
        """Mixture of Leave-one-out and Sobol' indices.

        Same as function :func:`leave_one_out_mse` but change the shape of the hypercube.
        Using Sobol' indices, the corners are shrinked by the corresponding percentage of the total indices.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        # Get the point of max error by LOOCV
        _, point = self.pod.estimate_quality()
        point = np.array(point)

        # Get Sobol' indices
        analyse = UQ(self.pod, self.settings)
        indices = analyse.sobol()

        # Determine the distance between the point and all other points in Space
        distances = np.array([np.linalg.norm(self.pod.points[i] - point) for i in range(len(self.pod.points))])
        distances = distances[np.nonzero(distances)]
        distance = min(distances) / 2
        self.logger.debug("Prior Distance min: {}".format(distance))
        distance = distance * indices[2]
        self.logger.debug("Post Distance min: {}".format(distance))

        # Construct the hypercube around the point
        hypercube = np.array([point - distance, point + distance]).T
        self.logger.debug("Prior Hypercube:\n{}".format(hypercube))
        self.logger.debug("Corners:\n{}".format(self.corners))
        hypercube[:, 0] = np.maximum(hypercube[:, 0], self.corners[:, 0])
        hypercube[:, 1] = np.minimum(hypercube[:, 1], self.corners[:, 1])
        self.logger.debug("Post Hypercube:\n{}".format(hypercube))

        # Global search of the point within the hypercube
        result = differential_evolution(self.func, hypercube)

        return result.x

