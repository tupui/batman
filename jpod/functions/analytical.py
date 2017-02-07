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

In each case, Sobol' indices are declared.

References
----------

.. [Oakley] Oakley, J., & O'Hagan, A. (2002). Bayesian inference for the uncertainty distribution of computer model outputs. Biometrika, 89(4), 769-784.

.. [Michalewicz] Molga, M., & Smutnicki, C. Test functions for optimization needs (2005).

.. [Rosenbrock] Dixon, L. C. W., & Szego, G. P. (1978). The global optimization problem: an introduction. Towards global optimization, 2, 1-15.

.. [Ishigami] Ishigami, T., & Homma, T. (1990, December): An importance quantification technique in uncertainty analysis for computer models. In Uncertainty Modeling and Analysis, 1990. Proceedings., First International Symposium on (pp. 398-403). IEEE.

.. [G-Function] Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000). Sensitivity analysis (Vol. 134). New York: Wiley.
"""
import numpy as np
import logging


class Michalewicz(object):

    r"""[Michalewicz]_ class.

    It is a multimodal *d*-dimensional function which has :math:`d!`
    local minima

    .. math:: f(x)=-\sum_{i=1}^d \sin(x_i)\sin^{2m}\left(\frac{ix_i^2}{\pi}\right),

    where *m* defines the steepness of the valleys and ridges.

    It is to difficult to search a global minimum when :math:`m`
    reaches large value. Therefore, it is recommended to have :math:`m < 10`.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=2, m=10):
        """Set up dimension."""
        self.d_in = d
        self.d_out = 1
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
        for i in range(self.d_in):
            f += np.sin(x[i]) * np.sin((i + 1) * x[i]
                                       ** 2 / np.pi) ** (2 * self.m)

        return -f


class Rosenbrock(object):

    r"""[Rosenbrock]_ class.

    .. math:: f(x)=\sum_{i=1}^{d-1}[100(x_{i+1}-x_i^2)^2+(x_i-1)^2]

    The function is unimodal, and the global minimum lies in a narrow,
    parabolic valley.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=2):
        """Set up dimension."""
        self.d_in = d
        self.d_out = 1
        if self.d_in == 2:
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
        for i in range(self.d_in - 1):
            f += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
        return f


class Ishigami(object):

    r"""[Ishigami]_ class.

    .. math::
    F = np.sin(X1)+7*np.sin(X2)**2+0.1*(X3**4)*np.sin(X1)

    It exhibits strong nonlinearity and nonmonotonicity.
    Depending on `a` and `b`, emphasize the non-linearities.
    It also has a dependence on x3 due to second order interactions (F13).

    """

    logger = logging.getLogger(__name__)

    def __init__(self, a=7., b=0.1):
        """Set up Ishigami.

        :param float a, b: Ishigami parameters
        """
        self.d_in = 3
        self.d_out = 1
        self.a = a
        self.b = b

        var = 0.5 + self.a ** 2 / 8 + self.b * np.pi ** 4 / 5 + self.b ** 2 * np.pi ** 8 / 18
        v1 = 0.5 + self.b * np.pi ** 4 / 5 + self.b ** 2 * np.pi ** 8 / 50
        v2 = a ** 2 / 8
        v3 = 0
        v12 = 0
        v13 = self.b ** 2 * np.pi ** 8 * 8 / 225
        v23 = 0
        v123 = 0

        self.s_first = np.array([v1 / var, v2 / var, v3 / var])
        self.s_second = np.array([[0.       ,        0., v13 / var],
                                  [v12 / var,        0., v23 / var],
                                  [v13 / var, v23 / var,        0.]])
        self.s_total2 = self.s_first + self.s_second.sum(axis=1)
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

    r"""[G-Function]_ class.

    .. math::
    F = \Pi_{i=1}^d \frac{\lvert 4x_i - 2\rvert + a_i}{1 + a_i}

    Depending on the coefficient :math:`a_i`, their is an impact on the impact
    on the output. The more the coefficient is for a parameter, the less the
    parameter is important.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=5, a=None):
        """G-function definition.

        :param int d: input dimension
        :param np.array a: (1, d)
        """
        self.d_in = d

        if a is None:
            self.a = np.arange(1, d + 1)
        else:
            self.a = np.array(a)

        vi = 1. / (3 * (1 + self.a)**2)
        v = -1 + np.prod(1 + vi)
        self.s_first = vi / v
        self.s_second = np.zeros((self.d_in, self.d_in))
        self.s_total = np.zeros(self.d_in)

        self.logger.info("Using function G-Function with d={}, a={}"
                         .format(self.d_in, self.a))

    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 1.
        for i in range(self.d_in):
            f *= (abs(4. * x[i] - 2) + self.a[i]) / (1. + self.a[i])
        return f


class Channel_Flow(object):

    """Channel Flow class.

    .. math::
        \frac{dh}{ds}=\mathcal{F}(h)=I\frac{1-(h/h_n)^{-10/3}}{1-(h/h_c)^{-3}}

    with :math:`h_c=\left(\frac{q^2}{g}\right)^{1/3}, h_n=\left(\frac{q^2}{IK_s^2}\right)^{3/10}`.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, dx=100., length=40000., width=500.):
        """Initialize the geometrical configuration.

        :param float dx: discretization
        :param float length: canal length
        :param float width: canal width
        """
        self.w = width
        self.I = 5e-4
        self.g = 9.8
        self.dx = dx
        self.length = length
        self.x = np.arange(self.dx, self.length + 1, self.dx)
        self.d_out = len(self.x)
        self.d_in = 2
        self.dl = int(self.length // self.dx)
        self.hinit = 10.
        self.Zref = - self.x * self.I

        # Sensitivity
        self.s_first = np.array([0.92925829, 0.05243018])
        self.s_second = np.array([[0., 0.01405351], [0.01405351, 0.]])
        self.s_total = np.array([0.93746788, 0.05887997])

        self.logger.info("Using function Channel Flow with: dx={}, length={}, "
                         "width={}".format(dx, length, width))

    def __call__(self, x):
        """Call function.

        :param list x: inputs [Ks, Q]
        :return: Water height along the channel
        :rtype: np.array 1D
        """
        ks, q = x
        hc = np.power((q ** 2) / (self.g * self.w ** 2), 1. / 3.)
        hn = np.power((q ** 2) / (self.I * self.w ** 2 * ks ** 2), 3. / 10.)

        h = self.hinit * np.ones(self.dl)
        for i in range(2, self.dl + 1):
            h[self.dl - i] = h[self.dl - i + 1] - self.dx * self.I\
                * ((1 - np.power(h[self.dl - i + 1] / hn, -10. / 3.))
                    / (1 - np.power(h[self.dl - i + 1] / hc, -3.)))

        return self.Zref + h
