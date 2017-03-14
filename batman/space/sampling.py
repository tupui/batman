# coding: utf8
from scipy import stats
import numpy as np
import openturns as ot

def doe(n_sample, bounds, kind):
    """Design of Experiment.

    :param int n_sample: number of samples
    :param np.array bounds: Space's corners
    :param str kind: Sampling Method
    :return: Sampling
    :rtype: lst(array)
    """
    dim = bounds.shape[1]
    r = np.zeros([n_sample, dim])
    if kind == 'halton':
        sequence_type = ot.LowDiscrepancySequence(ot.HaltonSequence(dim))
    elif kind == 'sobol':
        sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(dim))
    elif kind == 'faure':
        sequence_type = ot.LowDiscrepancySequence(ot.FaureSequence(dim))
    elif (kind == 'lhs') or (kind == 'lhsc'):
        distribution = ot.ComposedDistribution([ot.Uniform(0, 1)] * dim)
        sequence_type = ot.LHSExperiment(distribution, n_sample)
    elif kind == 'sobolscramble':
        pass
    elif kind == 'uniform':
        n = [n_sample] * len(bounds[1])
        return uniform(dim, n, bounds)
    else:
        raise ValueError('Bad sampling method: ' + kind)

    if (kind == 'lhs') or (kind == 'lhsc'):
        sample = sequence_type.generate()
    elif kind == 'sobolscramble':
        sample = scrambled_sobol_generate(dim, n_sample)
    else:
        sample = sequence_type.generate(n_sample)

    if kind == 'lhsc':
        for j in range(dim):
            b = bounds[0, j]
            a = bounds[1, j] - b
            for i, p in enumerate(sample):
                r[i, j] = a * ((p[j] // (1. / n_sample) + 1) - 0.5) / n_sample + b
    else:
        for j in range(dim):
            b = bounds[0, j]
            a = bounds[1, j] - b
            for i, p in enumerate(sample):
                r[i, j] = a * p[j] + b
    return r


def uniform(dim, n_sample, bounds):
    """Uniform sampling."""
    n = np.product(n_sample)
    h = []
    for i in range(dim):
        h1 = 1. / float(n_sample[i] - 1)
        h.append(h1)
    r = np.zeros([n, dim])
    compt = np.zeros([1, dim], np.int)
    for i in range(1, n):
        compt[0, 0] = compt[0, 0] + 1
        for j in range(dim - 1):
            if compt[0, j] > n_sample[j] - 1:
                compt[0, j] = 0
                compt[0, j + 1] = compt[0, j + 1] + 1
        for j in range(dim):
            r[i, j] = float(compt[0, j]) * h[j]
    for i in range(dim):
        b = bounds[0, i]
        a = bounds[1, i] - b
        for j in range(n):
            r[j, i] = a * r[j, i] + b
    return r


def scrambled_sobol_generate(dim, n_sample):
    """Scrambled Sobol.

    Scramble function as in Owen (1997)

    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """
    # Generate sobol sequence
    sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(dim))
    samples = sequence_type.generate(n_sample)
    r = np.zeros([n_sample, dim])

    for i, p in enumerate(samples):
        for j in range(dim):
            r[i, j] = p[j]

    # Scramble the sequence
    for col in range(0, dim):
        r[:, col] = scramble(r[:, col])

    return r


def scramble(x):
    """Scramble function."""
    Nt = len(x) - (len(x) % 2)

    idx = x[0:Nt].argsort()
    iidx = idx.argsort()

    # Generate binomial values and switch position for the second half of the
    # array
    bi = stats.binom(1, 0.5).rvs(size=Nt / 2).astype(bool)
    pos = stats.uniform.rvs(size=Nt / 2).argsort()

    # Scramble the indexes
    tmp = idx[0:Nt / 2][bi]
    idx[0:Nt / 2][bi] = idx[Nt / 2:Nt][pos[bi]]
    idx[Nt / 2:Nt][pos[bi]] = tmp

    # Apply the scrambling
    x[0:Nt] = x[0:Nt][idx[iidx]]

    # Apply scrambling to sub intervals
    if Nt > 2:
        x[0:Nt / 2] = scramble(x[0:Nt / 2])
        x[Nt / 2:Nt] = scramble(x[Nt / 2:Nt])

    return x
