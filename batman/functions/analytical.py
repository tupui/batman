# coding: utf8
"""
Analytical module
=================

Defines analytical Uncertainty Quantification oriented functions for test and
model evaluation purpose.

.. seealso:: [Surjanovic2017]_
             `Virtual Library <https://www.sfu.ca/~ssurjano/index.html>`_

It implements the following classes:

- :class:`SixHumpCamel`,
- :class:`Branin`,
- :class:`Michalewicz`,
- :class:`Rosenbrock`,
- :class:`Rastrigin`,
- :class:`Ishigami`,
- :class:`G_Function`,
- :class:`Forrester`,
- :class:`Channel_Flow`,
- :class:`Manning`,
- :class:`ChemicalSpill`.

In most case, Sobol' indices are pre-computed and storred as attributes.

References
----------
.. [Molga2005] Molga, M., & Smutnicki, C. Test functions for optimization needs
    (2005).
.. [Dixon1978] Dixon, L. C. W., & Szego, G. P. (1978). The global optimization
    problem: an introduction. Towards global optimization, 2, 1-15.
.. [Ishigami1990] Ishigami, T., & Homma, T. (1990, December): An importance
    quantification technique in uncertainty analysis for computer models. In
    Uncertainty Modeling and Analysis, 1990. Proceedings., First International
    Symposium on (pp. 398-403). IEEE.
.. [Saltelli2000] Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000).
    Sensitivity analysis (Vol. 134). New York: Wiley.
.. [Forrester2007] Forrester, Sobester. (2007). Multi-Fidelity Optimization via
    Surrogate Modelling. In Proceedings of the Royal Society A: Mathematical,
    Physical and Engineering Sciences.
.. [Forrester2008] Forrester, A., Sobester, A., & Keane, A. (2008). Engineering
    design via surrogate modelling: a practical guide. Wiley.
.. [Bliznyuk2008] Bliznyuk, N., Ruppert, D., Shoemaker, C., Regis, R., Wild,
    S., & Mugunthan, P. (2008). Bayesian calibration and uncertainty analysis
    for computationally expensive models using optimization and radial basis
    function approximation. Journal of Computational and Graphical
    Statistics, 17(2).
.. [Surjanovic2017] Surjanovic, S. & Bingham, D. (2013). Virtual Library of
    Simulation Experiments: Test Functions and Datasets. Retrieved September 11,
    2017, from http://www.sfu.ca/~ssurjano.

"""
import itertools
import logging
import numpy as np
from .utils import multi_eval


class SixHumpCamel:
    r"""SixHumpCamel class [Molga2005]_.

    .. math:: \left(4-2.1x_1^2+\frac{x_1^4}{3}\right)x_1^2+x_1x_2+
        (-4+4x_2^2)x_2^2

    The function has six local minima, two of which are global.

    .. math:: f(x^*) = -1.0316, x^* = (0.0898, -0.7126), (-0.0898,0.7126),
        x_1 \in [-3, 3], x_2 \in [-2, 2]
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Set up attributes."""
        self.d_in = 2
        self.d_out = 1
        if self.d_in == 2:
            self.s_first = np.array([0.775, 0.232])
            self.s_second = np.array([[0., 0.], [0., 0.]])
            self.s_total = np.array([0.774, 0.229])
        self.logger.info('Using function Six Hump Camel')

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = ((4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2 + x[0] * x[1]
             + (- 4 + 4 * x[1] ** 2) * x[1] ** 2)
        return f


class Branin:
    r"""Branin class [Forrester2008]_.

    .. math:: f(x) = \left( x_2 - \frac{5.1}{4\pi^2}x_1^2 + \frac{5}{\pi}x_1 - 6
              \right)^2 + 10 \left[ \left( 1 - \frac{1}{8\pi} \right) \cos(x_1)
              + 1 \right] + 5x_1.

    The function has two local minima and one global minimum. It is a modified
    version of the original Branin function that seek to be representative of
    engineering functions.

    .. math:: f(x^*) = -15,310076, x^* = (-\pi, 12.275), x_1 \in [-5, 10], x_2 \in [0, 15]
    """

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Set up attributes."""
        self.d_in = 2
        self.d_out = 1
        self.s_first = np.array([0.291, 0.216])
        self.s_second = np.array([[0., 0.442], [0.442, 0.]])
        self.s_total = np.array([0.793, 0.704])
        self.logger.info('Using function Branin')

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = (x[1] - 5.1 / (4 * np.pi ** 2) * x[0] ** 2 + 5 / np.pi * x[0] - 6) ** 2\
            + 10 * ((1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 1) + 5 * x[0]

        return f


class Michalewicz:
    r"""Michalewicz class [Molga2005]_.

    It is a multimodal *d*-dimensional function which has :math:`d!`
    local minima

    .. math:: f(x)=-\sum_{i=1}^d \sin(x_i)\sin^{2m}\left(\frac{ix_i^2}{\pi}\right),

    where *m* defines the steepness of the valleys and ridges.

    It is to difficult to search a global minimum when :math:`m`
    reaches large value. Therefore, it is recommended to have :math:`m < 10`.

    .. math:: f(x^*) = -1.8013, x^* = (2.20, 1.57), x \in [0, \pi]^d
    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=2, m=10):
        """Set up dimension."""
        self.d_in = d
        self.d_out = 1
        self.m = m
        if self.d_in == 2:
            self.s_first = np.array([0.4540, 0.5678])
            self.s_second = np.array([[0., 0.008], [0.008, 0.]])
            self.s_total = np.array([0.4606, 0.5464])
        self.logger.info("Using function Michalewicz with d={}"
                         .format(self.d_in))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 0.
        for i in range(self.d_in):
            f += np.sin(x[i]) * np.sin((i + 1) * x[i] ** 2 / np.pi) ** (2 * self.m)

        return -f


class Rosenbrock:
    r"""Rosenbrock class [Dixon1978]_.

    .. math:: f(x)=\sum_{i=1}^{d-1}[100(x_{i+1}-x_i^2)^2+(x_i-1)^2]

    The function is unimodal, and the global minimum lies in a narrow,
    parabolic valley.

    .. math:: f(x^*) = 0, x^* = (1, ..., 1), x \in [-2.048, 2.048]^d

    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=2):
        """Set up dimension."""
        self.d_in = d
        self.d_out = 1
        if self.d_in == 2:
            self.s_first = np.array([0.577, 0.258])
            self.s_second = np.array([[0., 0.304], [0.304, 0.]])
            self.s_total = np.array([0.741, 0.509])
        self.logger.info("Using function Rosenbrock with d={}"
                         .format(self.d_in))

    @multi_eval
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


class Rastrigin:
    r"""Rastrigin class [Molga2005]_.

    It is a multimodal *d*-dimensional function which has regularly distributed
    local minima.

    .. math:: f(x)=10d+\sum_{i=1}^d [x_i^2-10\cos(2\pi x_i)]

    .. math:: f(x^*) = 0, x^* = (0, ..., 0), x \in [-5.12, 5.12]^d
    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=2):
        """Set up dimension."""
        self.d_in = d
        self.d_out = 1
        if self.d_in == 2:
            self.s_first = np.array([0.22772082, 0.59709422])
            self.s_second = np.array([[0., 0.16719219], [0., 0.16719219]])
            self.s_total = np.array([0.46693546, 0.7761338])
        self.logger.info("Using function Rastrigin with d={}"
                         .format(self.d_in))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = 10. * self.d_in
        for i in range(self.d_in):
            f += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])

        return f


class Ishigami:
    r"""Ishigami class [Ishigami1990]_.

    .. math:: F = \sin(x_1)+7\sin(x_2)^2+0.1x_3^4\sin(x_1), x\in [-\pi, \pi]^3

    It exhibits strong nonlinearity and nonmonotonicity.
    Depending on `a` and `b`, emphasize the non-linearities.
    It also has a dependence on X3 due to second order interactions (F13).

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

        var = 0.5 + self.a ** 2 / 8 + self.b * np.pi ** 4 / 5\
            + self.b ** 2 * np.pi ** 8 / 18
        v1 = 0.5 + self.b * np.pi ** 4 / 5 + self.b ** 2 * np.pi ** 8 / 50
        v2 = a ** 2 / 8
        v3 = 0
        v12 = 0
        v13 = self.b ** 2 * np.pi ** 8 * 8 / 225
        v23 = 0

        self.s_first = np.array([v1 / var, v2 / var, v3 / var])
        self.s_second = np.array([[0., 0., v13 / var],
                                  [v12 / var, 0., v23 / var],
                                  [v13 / var, v23 / var, 0.]])
        self.s_total2 = self.s_first + self.s_second.sum(axis=1)
        self.s_total = np.array([0.558, 0.442, 0.244])
        self.logger.info("Using function Ishigami with a={}, b={}"
                         .format(self.a, self.b))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        f = np.sin(x[0]) + self.a * np.sin(x[1])**2 + \
            self.b * (x[2]**4) * np.sin(x[0])
        return f


class G_Function:
    r"""G_Function class [Saltelli2000]_.

    .. math:: F = \Pi_{i=1}^d \frac{\lvert 4x_i - 2\rvert + a_i}{1 + a_i}

    Depending on the coefficient :math:`a_i`, their is an impact on the impact
    on the output. The more the coefficient is for a parameter, the less the
    parameter is important.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, d=4, a=None):
        """G-function definition.

        :param int d: input dimension
        :param np.array a: (1, d)
        """
        self.d_in = d
        self.d_out = 1

        if a is None:
            self.a = np.arange(1, d + 1)
        else:
            self.a = np.array(a)

        vi = 1. / (3 * (1 + self.a)**2)
        v = -1 + np.prod(1 + vi)
        self.s_first = vi / v
        self.s_second = np.zeros((self.d_in, self.d_in))
        self.s_total = vi * np.prod(1 + vi) / v

        self.logger.info("Using function G-Function with d={}, a={}"
                         .format(self.d_in, self.a))

    @multi_eval
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


class Forrester:
    r"""Forrester class [Forrester2007]_.

    .. math:: F_{e}(x) = (6x-2)^2\sin(12x-4), \\
              F_{c}(x) = AF_e(x)+B(x-0.5)+C,

    were :math:`x\in{0,1}` and :math:`A=0.5, B=10, C=-5`.

    This set of two functions are used to represents a high an a low fidelity.

    """

    logger = logging.getLogger(__name__)

    def __init__(self, fidelity='e'):
        """Forrester-function definition.

        ``e`` stands for expansive and ``c`` for cheap.

        :param str fidelity: select the fidelity ``['e'|'f']``
        """
        self.d_in = 1
        self.d_out = 1
        self.fidelity = fidelity

        self.logger.info('Using function Forrester with fidelity: {}'
                         .format(self.fidelity))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        x = x[0]
        f_e = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        if self.fidelity == 'e':
            return f_e
        else:
            f = 0.5 * f_e + 10 * (x - 0.5) - 5

        return f


class Channel_Flow:
    r"""Channel Flow class.

    .. math:: \frac{dh}{ds}=\mathcal{F}(h)=I\frac{1-(h/h_n)^{-10/3}}{1-(h/h_c)^{-3}}\\
        h_c=\left(\frac{q^2}{g}\right)^{1/3}, h_n=\left(\frac{q^2}{IK_s^2}\right)^{3/10}
    """

    logger = logging.getLogger(__name__)

    def __init__(self, dx=8000., length=40000., width=500., slope=5e-4, hinit=10.):
        """Initialize the geometrical configuration.

        :param float dx: discretization.
        :param float length: Canal length.
        :param float width: Canal width.
        :param float slope: Canal slope.
        :param float hinit: Downstream boundary condition.
        """
        self.w = width
        self.slope = slope
        self.g = 9.8
        self.dx = dx
        self.length = length
        self.x = np.arange(self.dx, self.length + 1, self.dx)
        self.d_out = len(self.x)
        self.d_in = 2
        self.dl = int(self.length // self.dx)
        self.hinit = hinit
        self.zref = - self.x * self.slope

        # Sensitivity
        self.s_first = np.array([0.92925829, 0.05243018])
        self.s_second = np.array([[0., 0.01405351], [0.01405351, 0.]])
        self.s_total = np.array([0.93746788, 0.05887997])

        self.logger.info("Using function Channel Flow with: dx={}, length={}, "
                         "width={}".format(dx, length, width))

    @multi_eval
    def __call__(self, x, h_nc=False):
        """Call function.

        :param list x: inputs [Ks, Q].
        :param bool h_nc: Whether to return hc and hn.
        :return: Water height along the channel.
        :rtype: array_like (n_samples, n_features [+ 2])
        """
        ks, q = x
        hc = np.power((q ** 2) / (self.g * self.w ** 2), 1. / 3.)
        hn = np.power((q ** 2) / (self.slope * self.w ** 2 * ks ** 2), 3. / 10.)

        h = self.hinit * np.ones(self.dl)
        for i in range(2, self.dl + 1):
            h[self.dl - i] = h[self.dl - i + 1] - self.dx * self.slope\
                * ((1 - np.power(h[self.dl - i + 1] / hn, -10. / 3.))
                   / (1 - np.power(h[self.dl - i + 1] / hc, -3.)))

        z_h = self.zref + h

        return np.append(z_h, np.array([hc, hn])) if h_nc else z_h


class Manning:
    """Manning equation for rectangular channel class."""

    logger = logging.getLogger(__name__)

    def __init__(self, width=100., slope=5.e-4, inflow=1000, d=1):
        """Initialize the geometrical configuration.

        :param float width: canal width
        :param float slope: canal slope
        :param float inflow: canal inflow (optional)
        :param int dim: 1 (Ks) or 2 (Ks,Q)
        """
        self.d_in = d
        self.d_out = 1
        self.width = width
        self.slope = slope
        self.inflow = inflow

        self.logger.info("Using function Manning :  width={}, "
                         "slope={}, inflow={}, dim={}"
                         .format(width, slope, inflow, d))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs [Ks] or [Ks, Q]
        :return: Water height along the channel
        :rtype: float
        """
        x = np.array(x)
        if self.d_in == 1:
            ks = x
            q = self.inflow
        else:
            ks, q = x

        h = q / (ks * self.width * np.sqrt(self.slope))
        h = np.power(h, 3. / 5.)
        return h


class ChemicalSpill:
    r"""Environmental Model class [Bliznyuk2008]_.

    Model a pollutant spill caused by a chemical accident.
    ``C(x)`` being the concentration of the pollutant at the space-time vector
    ``(s, t)``, with ``0 < s < 3`` and ``t > 0``.

    A mass ``M`` of pollutant is spilled at each of two locations, denoted by
    the space-time vectors ``(0, 0)`` and :math:`(L, \tau)`. Each element of
    the response is a scaled concentration of the pollutant at the space-time
    vector.

    .. math:: f(X) = \sqrt{4\pi}C(X), x \in [[7, 13], [0.02, 0.12], [0.01, 3],
        [30.1, 30.295]]\\
        C(X) = \frac{M}{\sqrt{4\pi D_{t}}}\exp \left(\frac{-s^2}{4D_t}\right) +
        \frac{M}{\sqrt{4\pi D_{t}(t - \tau)}} \exp \left(-\frac{(s-L)^2}{4D(t -
        \tau)}\right) I (\tau < t)

    """

    logger = logging.getLogger(__name__)

    def __init__(self, s=None, tstep=0.3):
        """Definition of the time-space domain.

        :param list s: locations
        :param float tstep: time-step
        """
        self.d_in = 4

        self.s = [0.5, 1, 1.5, 2, 2.5] if s is None else s
        self.ds = len(self.s)

        self.t = np.arange(0.3, 60, tstep)
        self.dt = self.t.shape[0]

        self.d_out = self.ds * self.dt

        self.logger.info("Using function ChemicalSpill with s={}, tstep={}"
                         .format(self.s, tstep))

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param list x: inputs
        :return: f(x)
        :rtype: float
        """
        mass, diff_rate, second_spill, tau = x

        f = np.zeros((self.ds, self.dt))
        for i, j in itertools.product(range(self.ds), range(self.dt)):
            term1 = mass / np.sqrt(4 * np.pi * diff_rate * self.t[j])\
                * np.exp(-self.s[i] ** 2 / (4 * diff_rate * self.t[j]))

            term2 = 0
            if tau < self.t[j]:
                term2 = mass / np.sqrt(4 * np.pi * diff_rate * (self.t[j] - tau))\
                    * np.exp(-(self.s[i] - second_spill) ** 2
                             / (4 * diff_rate * (self.t[j] - tau)))

            f[i, j] = np.sqrt(4 * np.pi) * (term1 + term2)

        return f
