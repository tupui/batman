# coding: utf8
"""
Analytical module
=================

Defines analytical Uncertainty Quantification oriented functions for test and
model evaluation purpose.

It implements the following classes:

- :class:`Michalewicz`,
- :class:`Rosenbrock`,
- :class:`Ishigami`,
- :class:`G_Function`,
- :class:`Channel_Flow`.

"""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import numpy as np
import logging


class Michalewicz(object):

    """Michalewicz class."""

    logger = logging.getLogger(__name__)

    def __init__(self, d=2, m=10):
        """Set up dimension."""
        self.d = d
        self.m = m
        if self.d == 2:
            self.s_first = np.array([0.4540, 0.5678])
            self.s_second = np.array([[0., 0.008], [0.008, 0.]])
            self.s_total = np.array([0.4606, 0.5464])
        self.logger.info("Using function Michalewicz with d={}"
                         .format(self.d))

    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 0.
        for i in range(self.d):
            f += np.sin(x[i]) * np.sin((i + 1) * x[i]
                                       ** 2 / np.pi) ** (2 * self.m)

        return -f


class Rosenbrock(object):

    """Rosenbrock class."""

    logger = logging.getLogger(__name__)

    def __init__(self, d=2):
        """Set up dimension."""
        self.d = d
        if self.d == 2:
            self.s_first = np.array([0.229983, 0.4855])
            self.s_second = np.array([[0., 0.0920076], [0.0935536, 0.]])
            self.s_total = np.array([0.324003, 0.64479])
        self.logger.info("Using function Rosenbrock with d={}"
                         .format(self.d))

    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 0.
        for i in range(self.d - 1):
            f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return f


class Ishigami(object):

    """Ishigami class."""

    logger = logging.getLogger(__name__)

    def __init__(self, a=7., b=0.1):
        """Set up Ishigami.

        Depending on a and b, emphasize the non-linearities.
        Also declare first, second and total Sobol' indices.

        :param float a, b: Ishigami parameters
        """
        self.a = a
        self.b = b
        self.s_first = np.array([0.3139, 0.4424, 0.])
        self.s_second = np.array([[0., 0., 0.2], [0., 0., 0.], [0.2, 0., 0.]])
        self.s_total = np.array([0.558, 0.442, 0.244])
        self.logger.info("Using function Ishigami with a={}, b={}"
                         .format(self.a, self.b))

    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = np.sin(x[0]) + self.a * np.sin(x[1])**2 + \
            self.b * (x[2]**4) * np.sin(x[0])
        return f


class G_Function(object):

    """G-Function class."""

    logger = logging.getLogger(__name__)

    def __init__(self, d=5, a=None):
        """G-function definition.

        :param int d: input dimension
        :param np.array a: (1, d)
        """
        self.d = d

        if a is None:
            self.a = np.arange(1, d + 1)
        else:
            self.a = np.array(a)

        vi = 1. / (3 * (1 + self.a)**2)
        v = -1 + np.prod(1 + vi)
        self.s_first = vi / v

        self.logger.info("Using function G-Function with d={}, a={}"
                         .format(self.d, self.a))

    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 1.
        for i in range(self.d):
            f *= (abs(4. * x[i] - 2) + self.a[i]) / (1. + self.a[i])
        return f


class Channel_Flow(object):

    """Channel Flow class."""

    logger = logging.getLogger(__name__)

    def __init__(self, dx=100., length=40000., width=500.):
        """Initialize the geometrical configuration.

        Also declare first, second and total Sobol' indices.

        :param float dx: discretization
        :param float length: canal length
        :param float width: canal width
        """
        self.w = width
        self.I = 5e-4
        self.g = 9.8
        self.dx = dx
        self.length = length
        self.X = np.arange(self.dx, self.length + 1, self.dx)
        self.dl = int(self.length // self.dx)
        self.hinit = 10.
        self.Zref = - self.X * self.I

        # Sensitivity
        self.s_first = np.array([0.1, 0.8])
        self.s_second = np.array([[0., 0.1], [0.1, 0.]])
        self.s_total = np.array([0.1, 0.9])

        self.logger.info("Using function Channel Flow with: dx={}, length={}, "
                         "width={}".format(dx, length, width))

    def __call__(self, x):
        """Call function.

        :param list x: inputs [Q, Ks]
        :return: Water height along the channel
        :rtype: np.array 1D
        """
        q, ks = x
        hc = np.power((q ** 2) / (self.g * self.w ** 2), 1. / 3.)
        hn = np.power((q ** 2) / (self.I * self.w ** 2 * ks ** 2), 3. / 10.)

        h = self.hinit * np.ones(self.dl)
        for i in range(2, self.dl + 1):
            h[self.dl - i] = h[self.dl - i + 1] - self.dx * self.I\
                * ((1 - np.power(h[self.dl - i + 1] / hn, -10. / 3.))
                    / (1 - np.power(h[self.dl - i + 1] / hc, -3.)))

        return self.Zref + h
