# -*- coding: utf-8 -*-
"""
Refinement Module
=================

This module defines all resampling strategies that can be used.

"""


import logging
import numpy as N
import resampling
import itertools

class MSE():
    """MSE class.

    It returns the point where the mean square error (sigma) is maximum.
    To do so, it uses Gaussian Process information.


    """

    def __init__(self, pod):
        self.pod = pod
        self.point = None

    def func(self, coords):
        f, sigma = self.pod.predict('kriging', [coords])
        return sigma

    def get_point(self):
        import numpy as N
        num = 25
        x = N.linspace(-2, 2, num=num)
        y = N.linspace(-2, 2, num=num)
        z = N.linspace(-2, 2, num=num)
        #y = N.linspace(space['corners'][0][1], space['corners'][1][1], num=num)
        xyz = []
        sigma_max = 0.
        point = []
        for i, j, k in itertools.product(x, y, z):
            xyz = [float(i),float(j), float(k)]
            sigma = self.func(xyz)
            if sigma > sigma_max:
                point = xyz
                sigma_max = sigma
        
        return point




