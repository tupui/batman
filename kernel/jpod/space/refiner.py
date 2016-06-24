# -*- coding: utf-8 -*-
"""
Refinement Class
================

This class defines all resampling strategies that can be used.

"""


import logging
import numpy as N
import resampling
import itertools
from scipy.optimize import differential_evolution
import numpy as np


class Refiner():
    """Refiner class.
    
    Defines the methods:
    - mse(self)
    - leave_one_out_mse(self)

    """

    def __init__(self, pod, corners):
        self.pod = pod
        self.corners = np.array(corners).T
        self.point = None

    def func(self, coords):
        """Get the MSE for a given point.
       
        Retrieve the Gaussian Process estimation of sigma.
        Return - sigma in order to have a minimization problem.

        :return: - sigma
        :rtype: float
       
        """ 
        _, sigma = self.pod.predict('kriging', [coords])
        sum_sigma = np.sum(self.pod.S * sigma)
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

        Estimate the quality of the POD by leave-one-out and add a point arround the max error point.
        The point is added within an hypercube around the max error point.

        :return: The coordinate of the point to add
        :rtype: lst(float)
        
        """
        quality, point = self.pod.estimate_quality()
        point = np.array(point)
        hypercube = np.array([point * 0.9, point * 1.1]).T
        result = differential_evolution(self.func, hypercube)
        return result.x

