# coding: utf8
"""
Gaussian process sampler
------------------------
"""
import logging
import os
import matplotlib.pyplot as plt
import openturns as ot
import numpy as np
import batman as bat
from scipy.spatial import Delaunay


class GpSampler(object):
    """
    GpSampler class
    ===============

    Computes instances of a d-dimensional Gaussian process (Gp) discretized
    over a mesh (zero mean and parametric covariance), with d in {1,2,3}.

    It can be decomposed into two steps (Steps 1 and 3' or 3'')
    and two additional ones (Steps 2 and 4):

    1. Compute the Karhunen Loeve decomposition (KLd) using :func:`__init__`.
    2. Plot the modes of the KLd into files using :func:`plot_modes`
    3'. Build GP instances by sampling the weights of the KLd according to the
       standard normal distribution using :func:`__call__` OR
    3''. Build a GP instance by setting the weights of the KLd to fixed values
         using :func:`__call__`.
    4. Plot the GP instance(s) into files using :func:`plot_sample`.

    :Example:

    ::

        >> from batman.space.gp_sampler import GpSampler
        >>
        >> # Dimension 1 - Creation of the Gp sampler
        >> n_nodes = 100
        >> reference = {'indices': [[x/float(n_nodes)] for x in range(n_nodes)], 'values': [0 for x in range(n_nodes)]}
        >> sampler = GpSampler(reference)
        >> print(sampler)
        >> sampler.plot_modes()
        >>
        >> # Dimension 1 - Selection of a Gp instance from KLd coefficients
        >> coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8]
        >> instance = sampler(coeff=coeff)
        >> sampler.plot_sample(instance)
        >>
        >> # Dimension 1 - Sampling the Gp
        >> sample_size = 10
        >> sample = sampler(sample_size=sample_size)
        >> sampler.plot_sample(sample)
        >>
        >> # Dimension 2 - Creation of the Gp sampler
        >> n_nodes_by_dim = 10
        >> n_nodes = n_nodes_by_dim**2
        >> reference = {'indices': [[x/float(n_nodes_by_dim), y/float(n_nodes_by_dim)] for x in range(n_nodes_by_dim) for y in range(n_nodes_by_dim)], 'values': [0 for x in range(n_nodes)]}
        >> sampler = GpSampler(reference, "AbsoluteExponential([0.5, 0.5], [1.0])")
        >> print(sampler)
        >> sampler.plot_modes()
        >>
        >> # Dimension 2 - Selection of a Gp instance from KLd coefficients
        >> coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8]
        >> instance = sampler(coeff=coeff)
        >> sampler.plot_sample(instance)
        >>
        >> # Dimension 2 - Sampling the Gp
        >> sample_size = 10
        >> sample = sampler(sample_size=sample_size)
        >> sampler.plot_sample(sample)

    """

    logger = logging.getLogger(__name__)

    def __init__(self,
                 reference,
                 kernel="AbsoluteExponential([0.5], [1.0])",
                 add=True,
                 threshold=0.01):
        """From a reference function, define the Gp over an index set and
        compute its Karhunen Loeve decomposition.

        :param dict reference: reference function (possibly filename)
                               {'indices': [i1, i2, ..., in],
                                'values': [v1, v2, ..., vn]}
                               where ij = [ij1, ij2, ..., ijd], d in {1, 2, 3}
        :param bool add: add the Gp realizations to the reference function.
        :param str kernel: kernel for the covariance model.
        :param float threshold: minimal relative amplitude of the
               eigenvalues to consider in the KLd wrt the maximum eigenvalue.
        """
        if type(reference) is str or type(reference) is unicode:
            self.reference = np.atleast_1d(np.load(reference))[0]
        else:
            self.reference = reference
        self.n_nodes = len(self.reference['indices'])
        self.n_dim = len(self.reference['indices'][0])
        self.kernel = kernel
        self.add = add
        self.threshold = threshold

        # OpenTurns mesh construction
        indices = self.reference['indices']
        if self.n_dim == 1:
            vertices = indices
            simplices = [[i, i+1] for i in range(self.n_nodes-1)]
        elif self.n_dim == 2:
            vertices = indices
            tri = Delaunay(np.array(vertices))
            simplices = [[np.asscalar(xi) for xi in x] for x in list(tri.simplices)]
        elif self.n_dim == 3:
            vertices = indices
            tri = Delaunay(np.array(vertices))
            simplices = [[np.asscalar(xi) for xi in x] for x in list(tri.simplices)]
        self.mesh = ot.Mesh(vertices, simplices)

        # Kernel
        model = bat.space.kernel_to_ot(self.kernel)

        # Karhunen-Loeve decomposition algorithm using P1 approximation
        algo = ot.KarhunenLoeveP1Algorithm(self.mesh, model, self.threshold)

        # Computation of the eigenvalues and eigen function values at nodes
        algo.run()
        result = algo.getResult()
        eigen_values = result.getEigenValues()
        modes = result.getModesAsProcessSample()
        n_modes = modes.getSize()

        # Evaluation of the eigen functions
        for i in range(n_modes):
            modes[i] = ot.Field(self.mesh, modes[i].getValues() * [np.sqrt(eigen_values[i])])

        # Matrix of the modes over the grid (lines <> modes; columns <> times)
        gridded_modes = np.eye(n_modes, len(vertices))
        for i in range(n_modes):
            gridded_mode = np.array(modes[i].getValues())
            gridded_modes[i, :] = gridded_mode.T

        # Modes of the KLD evaluated over the mesh ([Nt x Nmodes] matrix)
        self.n_modes = n_modes
        self.modes = gridded_modes.T
        self.standard_deviation = np.sqrt(eigen_values)
        self.x_coord = np.array(self.mesh.getVertices())[:, 0].reshape(len(vertices))
        if self.n_dim > 1:
            self.y_coord = np.array(self.mesh.getVertices())[:, 1].reshape(len(vertices))
        if self.n_dim > 2:
            self.z_coord = np.array(self.mesh.getVertices())[:, 2].reshape(len(vertices))

    def __str__(self):
        """Summary of Gp and its Karhunen Loeve decomposition."""
        summary = ("Gp sampler summary:\n"
                   "- Dimension = {}\n"
                   "- Kernel = {}\n"
                   "- Mesh size = {}\n"
                   "- Threshold for the KLd = {}\n"
                   "- Number of modes = {}")

        format_ = [self.n_dim,
                   self.kernel, self.n_nodes, self.threshold, self.n_modes]

        return summary.format(*format_)

    def __call__(self, sample_size=1, coeff=None):
        """Compute realizations of the GP sampler.

        :param int sample_size: number of GP instances.
        :param list coeff: coefficients of the Karhunen Loeve decomposition.
        :return: instances of GP discretized over the mesh and KLd coefficients
        :rtype: np.array([sample_size x sum(n_nodes)]),
                np.array([sample_size x n_modes])
        """
        if coeff is None:
            dist = ot.ComposedDistribution([ot.Normal(0., 1.)] * self.n_modes,
                                           ot.IndependentCopula(self.n_modes))
            # Sampled weights
            weights = np.array(dist.getSample(sample_size))
        else:
            weights = list(coeff[0:self.n_modes]) + \
                      list(np.zeros(max(0, self.n_modes - len(coeff))))
            weights = np.array(weights)

        # Predictions
        sample = np.dot(self.modes, weights.T).T

        if self.add:
            sample += self.reference['values']

        return {'Values': sample, 'Coefficients': weights}

    def plot_modes(self, path='.'):
        """Plot the modes of the Karhunen Loeve decomposition.

        :param str path: path to write plot
        """
        if self.n_dim == 1:
            fig = plt.figure('Modes')
            ind = np.argsort(np.array(self.mesh.getVertices()).T)
            plt.plot(self.mesh.getVertices().sort(), self.modes[ind][0])
            fig.tight_layout()
            path = os.path.join(path, 'modes_gp.pdf')
            fig.savefig(path, transparent=True, bbox_inches='tight')
            plt.close('all')
        elif self.n_dim == 2:
            for i in range(self.n_modes):
                mode = np.reshape(self.modes.T[i], self.n_nodes)
                filename = path+'/'+'mode_gp_'+str(i)+'.pdf'
                title = "Gp mode #"+str(i + 1)
                self.surface_plot(mode, filename, title)

    def plot_sample(self, sample, path='.'):
        """Plot the sample.

        :param dict sample: Output of :func:`GpSampler.__call__`
        :param str path: path to write plot
        """
        if self.n_dim == 1:
            fig = plt.figure('Sample')
            if len(sample['Values'].shape) == 1:
                instance = sample['Values']
            else:
                instance = sample['Values'].T
            ind = np.argsort(np.array(self.mesh.getVertices()).T)
            plt.plot(self.mesh.getVertices().sort(), instance[ind[0]])
            fig.tight_layout()
            path = os.path.join(path, 'sample_gp.pdf')
            fig.savefig(path, transparent=True, bbox_inches='tight')
            plt.close('all')
        elif self.n_dim == 2:
            if len(sample['Values'].shape) == 1:
                instance = np.reshape(sample['Values'], self.n_nodes)
                filename = path+'/'+'sample_gp.pdf'
                title = "Gp instance"
                self.surface_plot(instance, filename, title)
            else:
                for i, _ in enumerate(sample['Values']):
                    instance = np.reshape(sample['Values'][i], self.n_nodes)
                    filename = path+'/'+'gp_instance_'+str(i)+'.pdf'
                    title = "Gp instance #" + str(i + 1)
                    self.surface_plot(instance, filename, title)

    def surface_plot(self, z_values, filename="surface_plot.pdf", title=""):
        """Plot a 2D surface plot.
        :param array_like z_values: Gp instance to plot
        :param str filename: filename to write plot
        :param str title: title of the plot
        """
        fig = plt.figure(filename)
        plt.tricontourf(self.x_coord, self.y_coord, z_values, 100)
        plt.plot(self.x_coord, self.y_coord, 'ko ')
        plt.title(title)
        fig.savefig(filename)
        plt.close('all')
