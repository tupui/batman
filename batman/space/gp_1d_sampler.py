# coding: utf8
"""
Gp1dSampler class
=================

Computes instances of a one-dimensional Gaussian Process (GP) discretized over
a mesh (zero mean and parametric covariance). It can be decomposed into three
steps:

1. Compute the Karhunen Loeve decomposition (KLD),
2. Sample the weights of the KLD according to the standard normal distribution,
   or set the weights to fixed values,
3. Build the corresponding GP1D realization(s).

:Example:

::

    >> from batman.space import Gp1dSampler

    Attributes of a Gp_1d_sampler object:
        - t_ini: the initial point of the mesh (default = 0).
        - T: the final point of the mesh (default = 1).
        - Nt: the size of the mesh (default = 100).
        - sigma: the GP standard deviation (default = 1.0).
        - theta: the GP correlation length (default = 1.0).
        - threshold: the minimal relative amplitude of the eigenvalues to consider in the KLD wrt the maximum eigenvalue (default = 0.01).
        - modes: Modes of the KLD evaluated over the mesh ([Nt x Nmodes] matrix).
"""
import logging
import matplotlib.pyplot as plt
import openturns as ot
from math import sqrt
import numpy as np
import time

ot.RandomGenerator.SetSeed(int(time.time() * 1e10))


class Gp1dSampler:

    """Gp1dSampler class."""

    logger = logging.getLogger(__name__)

    def __init__(self, t_ini=0, t_end=1, Nt=100, sigma=1.0, theta=0.5,
                 threshold=0.01, covariance="AbsoluteExponential"):
        """Computes the Karhunen Loeve decomposition and initializes GP1D.

            :param int t_ini: initial point of the mesh
            :param int t_end: final point of the mesh
            :param int Nt: size of the mesh
            :param float sigma: GP standard deviation
            :param float theta: GP correlation length
            :param float threshold: minimal relative amplitude of the
            eigenvalues to consider in the KLD wrt the maximum eigenvalue
            """
        self.t_ini = t_ini
        self.t_end = t_end
        self.Nt = Nt
        self.sigma = sigma
        self.theta = theta
        self.threshold = threshold
        self.covariance = covariance

        # OpenTurns mesh construction
        mesh = ot.IntervalMesher(
            [self.Nt - 1]).build(ot.Interval(self.t_ini, self.t_end))

        # Absolute exponential covariance model
        if covariance == "SquaredExponential":
            model = ot.SquaredExponential([self.theta], [self.sigma])
        elif covariance == "AbsoluteExponential":
            model = ot.AbsoluteExponential([self.theta], [self.sigma])
        elif covariance == "Matern32":
            model = ot.MaternModel([self.theta], [self.sigma], 1.5)
        elif covariance == "Matern52":
            model = ot.MaternModel([self.theta], [self.sigma], 2.5)
        elif covariance == "Exponential":
            model = ot.ExponentialModel(1, [self.sigma], [self.theta])
        elif covariance == "Spherical":
            model = ot.SphericalModel(1, [self.sigma], [self.theta])

        # Karhunen-Loeve decomposition factory using P1 approximation
        factory = ot.KarhunenLoeveP1Factory(mesh, self.threshold)

        # Computation of the eigenvalues and eigen function values at nodes
        ev = ot.NumericalPoint()
        modes = factory.buildAsProcessSample(model, ev)
        n_modes = modes.getSize()

        # Evaluation of the eigen functions
        for i in range(n_modes):
            modes[i] = ot.Field(mesh, modes[i].getValues() * [np.sqrt(ev[i])])

        # Matrix of the modes over the grid (lines <> modes; columns <> times)
        vaep = np.eye(n_modes, self.Nt)
        for i in range(n_modes):
            a = np.array(modes[i].getValues())
            vaep[i, :] = a.T

        # Modes of the KLD evaluated over the mesh ([Nt x Nmodes] matrix)
        self.n_modes = n_modes
        self.modes = vaep.T
        self.t = mesh

    def __str__(self):
        """Summary of GP1D and its Karhunen Loeve decomposition."""
        s = ("Mesh interval = [{},{}]"
             "Mesh size = {}"
             "GP standard deviation = {}"
             "GP correlation length = {}"
             "Threshold for the KLD = {}"
             "Number of modes = {}")
            .format(self.t_ini, self.t_end, self.Nt, self.sigma, self.theta,
                    self.threshold, self.n_modes)
        return s

    def plot_modes(self):
        """Plot the modes of the Karhunen Loeve decomposition."""
        plt.plot(self.t.getVertices(), self.modes)
        plt.show()

    def sample(self, n_sample=1, plot=False):
        """Compute realizations of the GP1D sampler.

        :param int n_sample: number of GP1D instances
        :param bool plot: plot the GP1D sample
        :return: instances of GP discretized over the mesh
        [t_ini:(T-T_ini)/(Nt-1):T] and Coefficients for the KLD
        :rtype: np.array([Nm x Nt]), np.array([Nm x Nmodes])
        """
        dist = ot.ComposedDistribution([ot.Normal(0., 1.)] * self.n_modes,
                                       ot.IndependentCopula(self.n_modes))

        # Sampled weights
        X = dist.getSample(n_sample)
        X = np.array(X)
        # Predictions
        Y = np.eye(n_sample, self.Nt)
        for i in range(n_sample):
            Y[i, :] = np.dot(self.modes, X[i])

        if plot:
            plt.plot(self.t.getVertices(), Y.T)
            plt.show()

        return {'Values': Y.T, 'Coefficients': X}

    def build(self, coeff=[0], plot=False):
        """Compute realization of the GP1D corresponding to :attr:`coeff`.

        :param list coeff: coefficients of the Karhunen Loeve decomposition
        :param bool plot: boolean for plotting the GP1D realization
        :return: Instance of the 1D GP discretized over the mesh
        [t_ini:(T-T_ini)/(Nt-1):T] and Coefficients for the KLD
        :rtype: dict([1 x Nt], [1 x Nmodes])
        """
        X = list(coeff[0:self.n_modes]) + \
            list(np.zeros(max(0, self.n_modes - len(coeff))))
        Y = np.dot(self.modes, X)
        if plot:
            plt.plot(self.t.getVertices(), Y.T)
            plt.show()
        return {'Values': Y.T, 'Coefficients': X}


class Gp2dSampler:
    """The class "Gp_2d_sampler" computes instances of a one-dimensional Gaussian Process (GP) discretized over a mesh. It can be decomposed into three steps: 
        1) Compute the Karhunen Loeve decomposition (KLD); 
        2) Sample the weights of the KLD according to the standard normal distribution.
           OR set the weights to fixed values.
        3) Build the corresponding GP2D realization(s).

    Attributes of a Gp_2d_sampler object:
        - t_ini: the initial point of the mesh (default = [0,0]).
        - T: the final point of the mesh (default = [1,1]).
        - Nt: the size of the mesh (default = [100,100]).
        - sigma: the GP standard deviation (default = 1.0).
        - theta: the GP correlation length (default = [0.5,0.5]).
        - threshold: the minimal relative amplitude of the eigenvalues to consider in the KLD wrt the maximum eigenvalue (default = 0.01).
        - modes: Modes of the KLD evaluated over the mesh ([prod(Nt) x Nmodes] matrix).
    """

    def __init__(self, t_ini=[0.0, 0.0], T=[1.0, 1.0], Nt=[10, 10], sigma=1.0, theta=[0.5, 0.5], threshold=0.01, covariance="AbsoluteExponential"):
        """ This function computes the Karhunen Loeve decomposition and initializes the GP2D object.
        Arguments:
            - t_ini: the initial point of the mesh (default = 0).
            - T: the final point of the mesh (default = 1).
            - Nt: the size of the mesh (default = 100).
            - sigma: the GP standard deviation (default = 1.0).
            - theta: the GP correlation length (default = 1.0).
            - threshold: the minimal relative amplitude of the eigenvalues to consider in the KLD wrt the maximum eigenvalue (default = 0.01).
        Fields :
            - Arguments of the constructor (t_ini, T, Nt, sigma, theta, threshold)
            - modes: the modes of the KLD evaluated over the mesh ([prod(Nt) x Nmodes] matrix)."""
        self.t_ini = t_ini
        self.t_end = T
        self.Nt = Nt
        self.sigma = sigma
        self.theta = theta
        self.threshold = threshold
        self.covariance = covariance

        # OpenTurns mesh construction
        mesh = ot.IntervalMesher(
            [x - 1 for x in self.Nt]).build(ot.Interval(self.t_ini, self.t_end))

        # Absolute exponential covariance model
        if covariance == "SquaredExponential":
            model = ot.SquaredExponential(self.theta, [self.sigma])
        elif covariance == "AbsoluteExponential":
            model = ot.AbsoluteExponential(self.theta, [self.sigma])
        elif covariance == "Matern32":
            model = ot.MaternModel(self.theta, self.sigma, 1.5)
        elif covariance == "Matern52":
            model = ot.MaternModel(self.theta, self.sigma, 2.5)
        elif covariance == "Exponential":
            model = ot.ExponentialModel(1, [self.sigma], self.theta)
        elif covariance == "Spherical":
            model = ot.SphericalModel(1, [self.sigma], self.theta)

        # Karhunen-Loeve decomposition factory using P1 approximation.
        factory = ot.KarhunenLoeveP1Factory(mesh, self.threshold)

        # Computation of the eigenvalues and eigen function values at nodes.
        ev = ot.NumericalPoint()
        modes = factory.buildAsProcessSample(model, ev)
        n_modes = modes.getSize()

        # Evaluation of the eigen functions
        for i in range(n_modes):
            modes[i] = ot.Field(mesh, modes[i].getValues() * [sqrt(ev[i])])

        # Matrix of the modes over the grid (lines <> modes; columns <> times)
        vaep = np.eye(n_modes, np.prod(self.Nt))
        for i in range(n_modes):
            a = np.array(modes[i].getValues())
            vaep[i, :] = a.T

        self.n_modes = n_modes
        self.modes = vaep
        self.t = mesh

    def summary(self):
        """This function gives a summary corresponding to the GP1D and its Karhunen Loeve decomposition"""
        print("Mesh interval = [{},{}]".format(self.t_ini, self.t_end))
        print("Mesh size = {}".format(self.Nt))
        print("GP standard deviation = {}".format(self.sigma))
        print("GP correlation length = {}".format(self.theta))
        print("Threshold for the KLD = {}".format(self.threshold))
        print("Number of modes = {}".format(self.n_modes))

    def plot_modes(self):
        """This function plots the modes of the Karhunen Loeve decomposition."""
        X, Y = np.meshgrid(np.arange(self.t_ini[0], self.t_end[0], (self.t_end[0] - self.t_ini[0]) / self.Nt[
                           0]), np.arange(self.t_ini[1], self.t_end[1], (self.t_end[1] - self.t_ini[1]) / self.Nt[1]))
        for i in range(min(self.n_modes, 9)):
            ax = plt.subplot("33" + str(i + 1))
            Z = np.reshape(self.modes[i], self.Nt)
            CS = plt.contour(X, Y, Z)
            plt.clabel(CS, inline=1, fontsize=10)
            plt.title("Mode " + str(i + 1))

        plt.show()

    def sample(self, N=1):
        """ This function computes "Nm" realizations of the GP2D.
        Arguments:
            - Nm: the number of GP2D instances (default = 1).
        Outputs:
            - ['Values']: Nm instances of the 2D GP discretized over the mesh.
                ** [Nm x prod(Nt)] matrix
            - ['Coefficients']: Coefficients for the KLD.
                ** [Nm x Nmodes] matrix"""
        # --- Input marginals
        normal = ot.Normal(0., 1.)
        collection = ot.DistributionCollection(self.n_modes)
        for i in range(self.n_modes):
            collection[i] = normal
        # --- Input distributions
        copula = ot.IndependentCopula(self.n_modes)
        distribution = ot.ComposedDistribution(collection, ot.Copula(copula))
        # --- Sampled weights
        X = distribution.getSample(N)
        X = np.array(X)
        # --- Predictions
        Y = np.eye(N, np.prod(self.Nt))
        for i in range(N):
            Y[i, :] = np.dot(self.modes.T, X[i])

        return {'Values': Y.T, 'Coefficients': X}

    def build(self, coeff=[0]):
        """ This function computes the realization of the GP1D corresponding to the coefficients "coeff".
        Arguments:
            - coeff: coefficients of the Karhunen Loeve decomposition (default = [0]).
        Outputs:
            - ['Values']: an instance of the 1D GP discretized over the mesh [t_ini:(T-T_ini)/(Nt-1):T].
                ** [1 x Nt] matrix
            - ['Coefficients']: Coefficients for the KLD.
                ** [1 x Nmodes] matrix"""
        X = list(coeff[0:self.n_modes]) + \
            list(np.zeros(max(0, self.n_modes - len(coeff))))
        Y = np.dot(self.modes.T, X)
        return {'Values': Y.T, 'Coefficients': X}
