# coding: utf8
"""
Doe class
=========

It uses design from class :class:`openturns.LowDiscrepancySequence`.
A sample is created according to the number of sample required, the boudaries
and the method.

:Example:

::

    >> from batman.space import Doe
    >> bounds = np.array([[0, 2], [10, 5]])
    >> kind = 'discrete'
    >> discrete_var = 0
    >> n = 5
    >> doe = Doe(n, bounds, kind, discrete_var)
    >> doe.generate()
    array([[ 5.        ,  3.        ],
       [ 2.        ,  4.        ],
       [ 8.        ,  2.33333333],
       [ 1.        ,  3.33333333],
       [ 6.        ,  4.33333333]])

"""
from scipy import stats
from scipy.stats import randint
import numpy as np
import openturns as ot


class Doe():

    """DOE class."""

    def __init__(self, n_sample, bounds, kind, var=0):
        """Initialize the DOE generation.

        In case of :attr:`kind` is ``uniform``, :attr:`n_sample` is decimated
        in order to have the same number of points in all dimensions.

        If :attr:`kind` is ``discrete``, a join distribution between a discrete
        uniform distribution is made with continuous distributions.

        Another possibility is to set a list of PDF to sample from. Thus one
        can do: `kind=['Uniform(15., 60.)', 'Normal(4035., 400.)']`.

        :param int n_sample: number of samples.
        :param array_like bounds: Space's corners [[min, n dim], [max, n dim]]
        :param str/list kind: Sampling Method if string can be one of
        ['halton', 'sobol', 'faure', 'lhs[c]', 'sobolscramble', 'uniform',
        'discrete'] otherwize can be a list of openturns distributions.
        :param int var: Position of the discrete variable.
        :return: Sampling
        :rtype: lst(array)
        """
        self.n_sample = n_sample
        self.bounds = bounds
        self.kind = kind
        self.dim = bounds.shape[1]

        if self.kind == 'halton':
            self.sequence_type = ot.LowDiscrepancySequence(ot.HaltonSequence(self.dim))
        elif self.kind == 'sobol':
            self.sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(self.dim))
        elif self.kind == 'faure':
            self.sequence_type = ot.LowDiscrepancySequence(ot.FaureSequence(self.dim))
        elif (self.kind == 'lhs') or (self.kind == 'lhsc'):
            distribution = ot.ComposedDistribution([ot.Uniform(0, 1)] * self.dim)
            self.sequence_type = ot.LHSExperiment(distribution, self.n_sample)
        elif (self.kind == 'lhsopt'):
            distribution = ot.ComposedDistribution([ot.Uniform(0, 1)] * self.dim)
            lhs = ot.LHSExperiment(distribution, self.n_sample)
            self.sequence_type = ot.SimulatedAnnealingLHS(lhs, ot.GeometricProfile(),
                                                          ot.SpaceFillingPhiP())
        elif self.kind == 'discrete':
            rv = randint(bounds[0, var], bounds[1, var] + 1)

            points = ot.Sample(10000, 1)
            for i in range(10000):
                points[i] = (rv.rvs(),)

            discrete = ot.UserDefined(points)
            dists = [discrete]
            dists.extend([ot.Uniform(0, 1)] * (self.dim - 1))
            distribution = ot.ComposedDistribution(dists)
            self.sequence_type = ot.LowDiscrepancyExperiment(ot.HaltonSequence(),
                                                             distribution,
                                                             self.n_sample)
        elif isinstance(self.kind, list):
            dists = ','.join(['ot.' + self.kind[i] for i in range(self.dim)])
            distribution = eval("ot.ComposedDistribution([" + dists + "])")
            self.sequence_type = ot.LowDiscrepancyExperiment(ot.HaltonSequence(),
                                                             distribution,
                                                             self.n_sample)

    def generate(self):
        """Generate the DOE."""
        if self.kind in ['lhs', 'lhsc', 'lhsopt', 'discrete']:
            sample = self.sequence_type.generate()
        elif self.kind == 'sobolscramble':
            sample = self.scrambled_sobol_generate()
        elif self.kind == 'uniform':
            sample = self.uniform()
        elif isinstance(self.kind, list):
            return np.array(self.sequence_type.generate())
        else:
            sample = self.sequence_type.generate(self.n_sample)

        # Scale the DOE from [0, 1] to bounds
        b = self.bounds[0]
        a = self.bounds[1] - b
        if self.kind == 'lhsc':
            r = a * ((np.floor_divide(sample, (1. / self.n_sample)) + 1)
                     - 0.5) / self.n_sample + b
        else:
            r = a * sample + b

        if self.kind == 'discrete':
            r[:, 0] = np.array(sample[:, 0]).flatten()

        return r

    def uniform(self):
        """Uniform sampling."""
        n_sample = int(np.floor(np.power(self.n_sample, 1.0 / len(self.bounds[1]))))
        n_sample = [n_sample] * len(self.bounds[1])
        n = np.product(n_sample)
        h = [1. / float(n_sample[i] - 1) for i in range(self.dim)]
        r = np.zeros([n, self.dim])
        compt = np.zeros([1, self.dim], np.int)
        for i in range(1, n):
            compt[0, 0] = compt[0, 0] + 1
            for j in range(self.dim - 1):
                if compt[0, j] > n_sample[j] - 1:
                    compt[0, j] = 0
                    compt[0, j + 1] = compt[0, j + 1] + 1
            for j in range(self.dim):
                r[i, j] = float(compt[0, j]) * h[j]
        return r

    def scrambled_sobol_generate(self):
        """Scrambled Sobol.

        Scramble function as in Owen (1997)

        Reference:

        .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
        """
        # Generate sobol sequence
        self.sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(self.dim))
        samples = self.sequence_type.generate(self.n_sample)
        r = np.empty([self.n_sample, self.dim])

        for i, p in enumerate(samples):
            for j in range(self.dim):
                r[i, j] = p[j]

        # Scramble the sequence
        for col in range(self.dim):
            r[:, col] = self.scramble(r[:, col])

        return r

    def scramble(self, x):
        """Scramble function."""
        Nt = len(x) - (len(x) % 2)

        idx = x[0:Nt].argsort()
        iidx = idx.argsort()

        # Generate binomial values and switch position for the second half of
        # the array
        bi = stats.binom(1, 0.5).rvs(size=Nt // 2).astype(bool)
        pos = stats.uniform.rvs(size=Nt // 2).argsort()

        # Scramble the indexes
        tmp = idx[0:Nt // 2][bi]
        idx[0:Nt // 2][bi] = idx[Nt // 2:Nt][pos[bi]]
        idx[Nt // 2:Nt][pos[bi]] = tmp

        # Apply the scrambling
        x[0:Nt] = x[0:Nt][idx[iidx]]

        # Apply scrambling to sub intervals
        if Nt > 2:
            x[0:Nt // 2] = self.scramble(x[0:Nt // 2])
            x[Nt // 2:Nt] = self.scramble(x[Nt // 2:Nt])

        return x
