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
import logging
from scipy import stats
from sklearn import preprocessing
import numpy as np
import openturns as ot
import batman as bat


class Doe:
    """DOE class."""

    logger = logging.getLogger(__name__)

    def __init__(self, n_samples, bounds, kind, dists=None, discrete=None):
        """Initialize the DOE generation.

        In case of :attr:`kind` is ``uniform``, :attr:`n_samples` is decimated
        in order to have the same number of points in all dimensions.

        If :attr:`kind` is ``discrete``, a join distribution between a discrete
        uniform distribution is made with continuous distributions.

        Another possibility is to set a list of PDF to sample from. Thus one
        can do: `dists=['Uniform(15., 60.)', 'Normal(4035., 400.)']`. If not
        set, uniform distributions are used.

        :param int n_samples: number of samples.
        :param array_like bounds: Space's corners [[min, n dim], [max, n dim]]
        :param str kind: Sampling Method if string can be one of
          ['halton', 'sobol', 'faure', '[o]lhs[c]', 'sobolscramble', 'uniform',
          'discrete'] otherwize can be a list of openturns distributions.
        :param lst(str) dists: List of valid openturns distributions as string.
        :param int discrete: Position of the discrete variable.
        """
        self.n_samples = n_samples
        self.bounds = np.asarray(bounds)
        self.kind = kind
        self.dim = self.bounds.shape[1]

        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.bounds)

        if dists is None:
            dists = [ot.Uniform(float(self.bounds[0][i]),
                                float(self.bounds[1][i]))
                     for i in range(self.dim)]
        else:
            dists = bat.space.dists_to_ot(dists)

        if discrete is not None:
            # Creating uniform discrete distribution for OT
            disc_list = [[i] for i in range(int(self.bounds[0, discrete]),
                                            int(self.bounds[1, discrete] + 1))]
            disc_dist = ot.UserDefined(disc_list)

            dists.pop(discrete)
            dists.insert(discrete, disc_dist)

        # Join distribution
        self.distribution = ot.ComposedDistribution(dists)

        if self.kind == 'halton':
            self.sequence_type = ot.LowDiscrepancyExperiment(ot.HaltonSequence(),
                                                             self.distribution,
                                                             self.n_samples)
        elif self.kind == 'sobol':
            self.sequence_type = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                                             self.distribution,
                                                             self.n_samples)
        elif self.kind == 'faure':
            self.sequence_type = ot.LowDiscrepancyExperiment(ot.FaureSequence(),
                                                             self.distribution,
                                                             self.n_samples)
        elif (self.kind == 'lhs') or (self.kind == 'lhsc'):
            self.sequence_type = ot.LHSExperiment(self.distribution, self.n_samples)
        elif self.kind == 'olhs':
            lhs = ot.LHSExperiment(self.distribution, self.n_samples)
            self.sequence_type = ot.SimulatedAnnealingLHS(lhs, ot.GeometricProfile(),
                                                          ot.SpaceFillingC2())
        elif self.kind == 'saltelli':
            # Only relevant for computation of Sobol' indices
            size = self.n_samples // (2 * self.dim + 2)  # N(2*dim + 2)
            self.sequence_type = ot.SobolIndicesExperiment(self.distribution,
                                                           size, True).generate()

    def generate(self):
        """Generate the DOE.

        :return: Sampling.
        :rtype: array_like (n_samples, n_features).
        """
        if self.kind == 'sobolscramble':
            return self.scrambled_sobol_generate()
        elif self.kind == 'uniform':
            sample = self.uniform()
        elif self.kind == 'lhsc':
            sample = self.sequence_type.generate()
        elif self.kind == 'saltelli':
            return np.array(self.sequence_type)
        else:
            return np.array(self.sequence_type.generate())

        # Scale the DOE from [0, 1] to bounds
        if self.kind == 'lhsc':
            sample = ((np.floor_divide(sample, (1. / self.n_samples)) + 1)
                      - 0.5) / self.n_samples
        else:
            sample = self.scaler.inverse_transform(sample)

        return sample

    def uniform(self):
        """Uniform sampling."""
        n_samples = int(np.floor(np.power(self.n_samples, 1.0 / len(self.bounds[1]))))
        n_samples = [n_samples] * len(self.bounds[1])
        n = np.product(n_samples)
        h = [1. / float(n_samples[i] - 1) for i in range(self.dim)]
        r = np.zeros([n, self.dim])
        compt = np.zeros([1, self.dim], np.int)
        for i in range(1, n):
            compt[0, 0] = compt[0, 0] + 1
            for j in range(self.dim - 1):
                if compt[0, j] > n_samples[j] - 1:
                    compt[0, j] = 0
                    compt[0, j + 1] = compt[0, j + 1] + 1
            for j in range(self.dim):
                r[i, j] = float(compt[0, j]) * h[j]
        return r

    def scrambled_sobol_generate(self):
        """Scrambled Sobol.

        Scramble function as in Owen (1997).
        """
        # Generate sobol sequence
        samples = ot.LowDiscrepancyExperiment(ot.SobolSequence(),
                                              self.distribution,
                                              self.n_samples)
        r = np.array(samples.generate())

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
