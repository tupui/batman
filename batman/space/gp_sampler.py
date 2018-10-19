# coding: utf8
"""
Gaussian process sampler
------------------------

Computes instances of a d-dimensional Gaussian process (Gp) discretized
over a mesh with a parametric covariance and d in {1,2,3}.

It can be decomposed into two steps (Steps 1 and 3' or 3'')
and two additional ones (Steps 2 and 4, available for d in {1, 2}):

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
    >> reference = {'indices': [[x/float(n_nodes)]
    >>                          for x in range(n_nodes)],
    >>              'values': [0 for x in range(n_nodes)]}
    >> sampler = GpSampler(reference)
    >> print(sampler)
    >> sampler.plot_modes()
    >> sampler.plot_modes("modes.pdf")
    >>
    >> # Dimension 1 - Selection of a Gp instance from KLd coefficients
    >> coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    >> instance = sampler(coeff=coeff)
    >> sampler.plot_sample(instance)
    >> sampler.plot_sample(instance, "instance.pdf")
    >>
    >> # Dimension 1 - Sampling the Gp
    >> sample_size = 10
    >> sample = sampler(sample_size=sample_size)
    >> sampler.plot_sample(sample)
    >> sampler.plot_sample(sample, "sample.pdf")
    >>
    >> # Dimension 2 - Creation of the Gp sampler
    >> n_nodes_by_dim = 10
    >> n_nodes = n_nodes_by_dim**2
    >> reference = {'indices': [[x/float(n_nodes_by_dim),
    >>                           y/float(n_nodes_by_dim)]
    >>                          for x in range(n_nodes_by_dim)
    >>                          for y in range(n_nodes_by_dim)],
    >>              'values': [0 for x in range(n_nodes)]}
    >> sampler = GpSampler(reference,
    >>                     "Matern([0.5, 0.5], nu=0.5)")
    >> print(sampler)
    >> sampler.plot_modes()
    >> sampler.plot_modes("mode.pdf")
    >>
    >> # Dimension 2 - Selection of a Gp instance from KLd coefficients
    >> coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    >> instance = sampler(coeff=coeff)
    >> sampler.plot_sample(instance)
    >> sampler.plot_sample(instance, "instance.pdf")
    >>
    >> # Dimension 2 - Sampling the Gp
    >> sample_size = 10
    >> sample = sampler(sample_size=sample_size)
    >> sampler.plot_sample(sample)
    >> sampler.plot_sample(sample, "instance.pdf")

"""
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.decomposition import PCA
import batman as bat
from .sampling import Doe
PY2 = sys.version_info.major == 2


class GpSampler:
    """GpSampler class."""

    logger = logging.getLogger(__name__)

    def __init__(self, reference, kernel="Matern(0.5, nu=0.5)", add=True,
                 threshold=0.01, std=1.):
        """From a reference function, define the Gp over an index set and
        compute its Karhunen Loeve decomposition.

        :param dict reference: reference function (possibly filename)
                               {'indices': [i1, i2, ..., in],
                                'values': [v1, v2, ..., vn]}
                               where ij = [ij1, ij2, ..., ijd], d in {1, 2, 3}
        :param bool add: add the Gp realizations to the reference function.
        :param str kernel: scikit-learn kernel for the covariance model.
        :param float threshold: the minimal relative amplitude of the
          eigenvalues to consider in the decomposition wrt the sum of the
          preceeding eigenvalues.
        :param float std: standard deviation of the Gaussian process.
        """
        # Check if string (lenient for byte-strings on Py2):
        if isinstance(reference, basestring if PY2 else str):
            self.reference = np.atleast_1d(np.load(reference))[0]
        else:
            self.reference = reference
        self.n_nodes = len(self.reference['indices'])
        self.n_dim = len(self.reference['indices'][0])
        self.kernel = kernel
        self.std = std
        self.add = add
        self.threshold = threshold
        gp = GaussianProcessRegressor(kernel=bat.space.kernel_to_skl(self.kernel))
        y = gp.sample_y(self.reference['indices'], 10000, random_state=None).T
        self.pca = PCA(svd_solver="full")
        y_transform = self.pca.fit_transform(y)

        def truncation_order(x):
            kept = [x[i] / np.sum(x[:i]) > self.threshold for i, _ in enumerate(x[1:], 1)]
            kept.insert(0, True)
            return np.sum(kept)

        standard_deviation = np.std(y_transform, 0)
        self.n_modes = truncation_order((standard_deviation / np.sqrt(self.n_nodes)) ** 2)
        self.modes = self.pca.components_[:self.n_modes, :]
        self.standard_deviation = standard_deviation[:self.n_modes] / np.sqrt(self.n_nodes)
        self.scaled_modes = (self.modes.T * self.standard_deviation).T

        def extract_coord(i):
            temp = np.array([[x[i]] for x in self.reference['indices']])
            return temp.reshape(self.n_nodes)

        self.x_coord = extract_coord(0)
        if self.n_dim > 1:
            self.y_coord = extract_coord(1)
        if self.n_dim > 2:
            self.z_coord = extract_coord(2)

    def __repr__(self):
        """Summary of Gp and its Karhunen Loeve decomposition."""
        summary = ("Gp sampler summary:\n"
                   "- Dimension = {}\n"
                   "- Kernel = {}\n"
                   "- Standard deviation = {}\n"
                   "- Mesh size = {}\n"
                   "- Threshold for the KLd = {}\n"
                   "- Number of modes = {}")

        format_ = [self.n_dim, self.kernel, self.std, self.n_nodes,
                   self.threshold, self.n_modes]

        return summary.format(*format_)

    def __call__(self, sample_size=1, coeff=None, kind="mc"):
        """Compute realizations of the GP sampler.

        :param int sample_size: number of GP instances.
        :param list coeff: coefficients of the Karhunen Loeve decomposition.
        :param str kind: Sampling Method if string can be one of
          ['halton', 'sobol', 'faure', 'lhs[c]', 'sobolscramble', 'uniform',
          'discrete'] otherwize can be a list of openturns distributions.
        :return: instances of GP discretized over the mesh and KLd coefficients
        :rtype: np.array([sample_size x sum(n_nodes)]),
                np.array([sample_size x n_modes])
        """
        if coeff is None:
            if kind == "mc":
                weights = np.random.normal(size=(sample_size, self.n_modes))
            else:
                doe = Doe(sample_size, [[-10.] * self.n_modes, [10.] * self.n_modes],
                          kind, ['Normal(0., 1.)' for i in range(self.n_modes)], None)
                weights = doe.generate()
        else:
            def pad(x):
                x_n_modes = np.array(x[:self.n_modes])
                n_non_modes = max(0, self.n_modes - len(x))
                temp = np.pad(x_n_modes, (0, n_non_modes),
                              'constant', constant_values=(0.))
                return temp

            weights = np.array([pad(x) for x in coeff])

        sample = weights.dot(self.scaled_modes) * self.std

        if self.add:
            sample += self.reference['values']

        return {'Values': sample, 'Coefficients': weights}

    def plot_modes(self, fname=None):
        """Plot the modes of the Karhunen Loeve decomposition.

        :param str fname: whether to export to filename or display the figures.
        """
        if self.n_dim == 1:
            abscissa = np.array(self.reference['indices']).T
            ind = np.argsort(abscissa)
            values = [[x[i] for i in ind[0]] for x in self.scaled_modes]
            abscissa.sort()

            fig = plt.figure('Modes')
            plt.plot(abscissa[0], np.array(values).T)
            bat.visualization.save_show(fname, [fig])
        elif self.n_dim == 2:
            for i in range(self.n_modes):
                title = "Gp mode #" + str(i)
                if fname is not None:
                    basename = os.path.splitext(fname)[0]
                    extension = os.path.splitext(fname)[1]
                    fname_i = basename + '_' + str(i) + extension
                else:
                    fname_i = None
                self._surface_plot(self.scaled_modes[i, :], fname_i, title)

    def plot_sample(self, sample, fname=None):
        """Plot the sample.

        :param dict sample: Output of :func:`GpSampler.__call__`.
        :param str fname: whether to export to filename or display the figures.
        """
        if self.n_dim == 1:
            abscissa = np.array(self.reference['indices']).T
            ind = np.argsort(abscissa)
            values = [[x[i] for i in ind[0]] for x in sample['Values']]
            abscissa.sort()

            fig = plt.figure('Sample')
            plt.plot(abscissa[0], np.array(values).T)
            bat.visualization.save_show(fname, [fig])
        elif self.n_dim == 2:
            for i, value in enumerate(sample['Values']):
                title = "Gp instance #" + str(i)
                if fname is not None:
                    basename = os.path.splitext(fname)[0]
                    extension = os.path.splitext(fname)[1]
                    fname_i = basename + '_' + str(i) + extension
                else:
                    fname_i = None

                self._surface_plot(value, fname_i, title)

    def _surface_plot(self, z_values, fname=None, title=None):
        """Plot a 2D surface plot.

        :param array_like z_values: Gp instance to plot.
        :param str fname: whether to export to filename or display the figures.
        :param str title: title of the plot.
        """
        fig = plt.figure(fname)
        plt.tricontourf(self.x_coord, self.y_coord, z_values, 100)
        plt.plot(self.x_coord, self.y_coord, 'ko ')
        if title:
            plt.title(title)
        bat.visualization.save_show(fname, [fig])
