# ==========================================================================
# Project: cfd - POD - Copyright (c) 2009 by CERFACS
# Type   :
# File   : sampling.py
# Vers   : V1.0
# Chrono : No  Date       Author                 V   Comments
# 1.0 11/08/2009 Braconnier             0.1 Creation
# ==========================================================================

from scipy import stats

import numpy as N
import openturns as ot


def mat_yy(dim):
    n = 2 ** dim
    yy = N.zeros([n, dim])
    for j in range(dim):
        k = 2 ** (j + 1)
        nk = n / k
        for i in range(k):
            yy[i * nk:(i + 1) * nk, j:j + 1] = (-1) ** (i + 1) * \
                N.ones([nk, 1])
    return yy

# Return a DOE of size [0,1]^d with d he dimension of the space


def DOE(dim, n1, bounds, kind):
    r = N.zeros([n1, dim])
    if kind == 'halton':
        sequence_type = ot.LowDiscrepancySequence(ot.HaltonSequence(dim))
    elif kind == 'sobol':
        sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(dim))
    elif kind == 'faure':
        sequence_type = ot.LowDiscrepancySequence(ot.FaureSequence(dim))
    elif (kind == 'lhs') or (kind == 'lhsc'):
        distribution = ot.ComposedDistribution([ot.Uniform(0, 1)] * dim)
        sequence_type = ot.LHSExperiment(distribution, n1)
    elif kind == 'sobolscramble':
        test = 3
    else:
        raise ValueError('Bad sampling method: ' + kind)

    if (kind == 'lhs') or (kind == 'lhsc'):
        sample = sequence_type.generate()
    elif kind == 'sobolscramble':
        sample = scrambled_sobol_generate(dim, n1)
    else:
        sample = sequence_type.generate(n1)

    if kind == 'lhsc':
        for j in range(dim):
            b = bounds[0, j]
            a = bounds[1, j] - b
            for i, p in enumerate(sample):
                r[i, j] = a * ((p[j] // (1. / n1) + 1) - 0.5) / n1 + b
    else:
        for j in range(dim):
            b = bounds[0, j]
            a = bounds[1, j] - b
            for i, p in enumerate(sample):
                r[i, j] = a * p[j] + b
    return r


def rlhc(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'lhs')


def clhc(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'lhsc')


def halton(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'halton')


def sobol(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'sobol')


def sobol_scramble(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'sobolscramble')


def faure(dim, n1, bounds):
    return DOE(dim, n1, bounds, 'faure')


def uniform(dim, n1, bounds):
    n = N.product(n1)
    h = []
    for i in range(dim):
        h1 = 1. / float(n1[i] - 1)
        h.append(h1)
    r = N.zeros([n, dim])
    compt = N.zeros([1, dim], N.int)
    for i in range(1, n):
        compt[0, 0] = compt[0, 0] + 1
        for j in range(dim - 1):
            if compt[0, j] > n1[j] - 1:
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


def scrambled_sobol_generate(dim, n1):
    """
    Scramble function as in Owen (1997)

    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """

    # Generate sobol sequence
    #samples = sobol(dim, n1, bounds)

    sequence_type = ot.LowDiscrepancySequence(ot.SobolSequence(dim))
    samples = sequence_type.generate(n1)
    r = N.zeros([n1, dim])

    for i, p in enumerate(samples):
        for j in range(dim):
            r[i, j] = p[j]

    # Scramble the sequence
    for col in range(0, dim):
        r[:, col] = scramble(r[:, col])

    return r


def scramble(x):
    """
    Scramble function as in Owen (1997)

    Reference:

    .. [1] Saltelli, A., Chan, K., Scott, E.M., "Sensitivity Analysis"
    """

    Nt = len(x) - (len(x) % 2)

    idx = x[0:Nt].argsort()
    iidx = idx.argsort()

    # Generate binomial values and switch position for the second half of the
    # array

    N.random.seed(seed=233423)  # SEED

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

