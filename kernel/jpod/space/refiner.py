# -*- coding: utf-8 -*-
"""Resampling the space of parameters."""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import logging
from scipy.optimize import differential_evolution
import numpy as np


class Refiner():

    """
    Refinement Class
    ================

    This class defines all resampling strategies that can be used.

    It implements the following methods:

    - `mse(self)`
    - `leave_one_out_mse(self)`

    :Example:

    >> corners = [(10, 400), (18, 450)]
    >> resample = Refiner(pod, corners)
    >> new_point = resample.mse()

    """

    logger = logging.getLogger(__name__)

    def __init__(self, pod, corners):
        """Initialize the refiner with the POD and space corners."""
        self.pod = pod
        self.corners = np.array(corners).T
        self.point = None

    def func(self, coords):
        r"""Get the MSE for a given point.

        Retrieve the Gaussian Process estimation of sigma: the mean square error.
        A composite indicator is constructed using POD's modes.

        .. math:: \sum S^2 \times \sigma

        The function returns - sum_sigma in order to have a minimization problem.

        :return: - sum_sigma
        :rtype: float

        """
        _, sigma = self.pod.predict('kriging', [coords])
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

        Estimate the quality of the POD by *leave-one-out cross validation* (LOOCV) and add a point arround the max error point.
        The point is added within an hypercube around the max error point.

        :return: The coordinate of the point to add
        :rtype: lst(float)

        """
        # Get the point of max error by LOOCV
        _, point = self.pod.estimate_quality()
        point = np.array(point)

        # Construct the hypercube around the point
        hypercube = np.array([point * 0.99, point * 1.01]).T
        self.logger.debug("Hypercube: {}".format(hypercube))

        result = differential_evolution(self.func, hypercube)

        return result.x
