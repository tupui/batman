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


# dim : dimension
# np : nombre de points
# bounds : points extremes
# tag : nom de la methode
# filename : nom d un fichier


#=========================================================================
# def sampling(dim, np, bounds, tag, filename):
#     if tag == 'halt' or tag == 'HALT':
#         S = halton(dim, np, bounds)
#     elif tag == 'lhcc' or tag == 'LHCC':
#         S = clhc(dim, np, bounds)
#     elif tag == 'lhcr' or tag == 'LHCR':
#         S = rlhc(dim, np, bounds)
#     elif tag == 'sobo' or tag == 'SOBO':
#         S = sobol(dim, np, bounds)
#     elif tag == 'unif' or tag == 'UNIF':
#         S = uniform(dim, np, bounds)
#     elif tag == 'soboscramble' or tag == 'SOBOSCRAMBLE':
#         S = sobol_scramble(dim, np, bounds)
#     elif tag == 'file' or tag == 'FILE':
#         S = N.zeros([np, dim])
#         fid = open(filename, 'r')
#         for i in range(np):
#             temp = fid.readline()
#             istart = 0
#             iend = 0
#             for j in range(dim):
#                 while temp[iend] != ' ' and temp[iend] != '\n':
#                     iend = iend + 1
#                 S[i, j] = float(temp[istart:iend])
#                 istart = iend + 1
#                 iend = istart
#         fid.close()
#     else:
#         print 'Unknown sampling method, check out 4th parameter'
#     return S
#=========================================================================

#=========================================================================
# if __name__ == '__main__':
#     dim = input('dim  ')
#     np = input('np  ')
#     bounds = N.zeros([2, dim])
#     for i in range(dim):
#         print 'dimension ', i + 1
#         bounds[0, i] = input('min ')
#         bounds[1, i] = input('max ')
#     TAG = 'HALT'
#     S = sampling(dim, np, bounds, TAG, file)
#     (NS, MS) = N.shape(S)
#     for i in range(NS):
#         print S[i, :]
#=========================================================================


#=========================================================================
# def mat_yy(dim):
#     n = 2 ** dim
#     yy = N.zeros([n, dim])
#     for j in range(dim):
#         k = 2 ** (j + 1)
#         nk = n / k
#         for i in range(k):
#             yy[i * nk:(i + 1) * nk, j:j + 1] = (-1) ** (i + 1) * \
#                 N.ones([nk, 1])
#     return yy
#
#
# def prime(n):
#     npvec = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
#              61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
#              137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
#              199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
#              277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
#              359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433,
#              439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509,
#              521, 523, 541]
#     if n < 0:
#         p = npvec[99]
#     elif n > 100:
#         p = npvec[99]
#     elif n == 0:
#         p = 1
#     else:
#         p = npvec[n - 1]
#     return p
#
#
# def setparhalton(dim):
#     base = N.zeros([1, dim], N.int)
#     leap = N.zeros([1, dim], N.int)
#     seed = N.zeros([1, dim], N.int)
#     for i in range(dim):
#         base[0, i] = prime(i + 1)
#         leap[0, i] = 1
#         seed[0, i] = 0
#     return (base, leap, seed)
#=========================================================================

#=========================================================================
# def halton(dim, n1, bounds):
#     n = n1 - 2 ** dim + 1
#     (base, leap, seed) = setparhalton(dim)
#     r = N.zeros([n, dim])
#     seed2 = N.zeros([1, n], N.int)
#     digit = N.zeros([1, n], N.int)
#     for i in range(dim):
#         for j in range(n):
#             seed2[0, j] = seed[0, i] + j * leap[0, i]
#         base_inv = 1. / float(base[0, i])
#         test = N.sum(seed2, 1)
#         while test[0] > 0.:
#             for k in range(n):
#                 digit[0, k] = seed2[0, k] % base[0, i]
#             for k in range(n):
#                 r[k, i] = r[k, i] + float(digit[0, k]) * base_inv
#             base_inv = base_inv / float(base[0, i])
#             for k in range(n):
#                 seed2[0, k] = seed2[0, k] / base[0, i]
#             test = N.sum(seed2, 1)
#     r = r[1:n, 0:dim]
#     m = 2 ** dim
#     r = N.concatenate((N.zeros([m, dim]), r), 0)
#     yy = mat_yy(dim)
#     for i in range(m):
#         for j in range(dim):
#             if yy[i, j] > 0.:
#                 r[i, j] = 1.
#     for i in range(dim):
#         b = bounds[0, i]
#         a = bounds[1, i] - b
#         for j in range(n1):
#             r[j, i] = a * r[j, i] + b
#     return r
#=========================================================================


#=========================================================================
# def uniform_01(seed):
#     k = seed / 127773
#     seed = 16807 * (seed - k * 127773) - k * 2836
#     if seed < 0:
#         seed = seed + 2147483647
#     p = float(seed) * 4.656612875e-10
#     return (p, seed)
#
#
# def round(x):
#     ix = int(x)
#     rx = x - ix
#     if rx <= .5:
#         y = ix
#     else:
#         y = ix + 1
#     return y
#
#
# def i4_uniform(a, b, seed):
#     minab = min(a, b)
#     maxab = max(a, b)
#     k = seed / 127773
#     seed = 16807 * (seed - k * 127773) - k * 2836
#     if seed < 0:
#         seed = seed + 2147483647
#     r = float(seed) * 4.656612875e-10
#     r1 = (1. - r) * (float(minab) - .5) + r * (float(maxab) + .5)
#     r = round(r1)
#     p = max(r, minab)
#     p = min(p, maxab)
#     return (p - 1, seed)
#
#
# def ci4_uniform(a, b, seed):
#     a = round(a)
#     b = round(b)
#     minab = min(a, b)
#     maxab = max(a, b)
#     seed = seed % 2147483647
#     if seed < 0:
#         seed = seed + 2147483647
#     k = seed / 127773
#     seed = 16807 * (seed - k * 127773) - k * 2836
#     if seed < 0:
#         seed = seed + 2147483647
#     r = float(seed) * 4.656612875e-10
#     r1 = (1. - r) * (float(minab) - .5) + r * (float(maxab) + .5)
#     r = round(r1)
#     p = max(r, minab)
#     p = min(p, maxab)
#     return (p - 1, seed)
#
#
# def cperm_random(n, seed):
#     p = N.zeros([1, n], N.int)
#     for i in range(n):
#         p[0, i] = i
#     for i in range(n):
#         (j, seed) = ci4_uniform(i + 1, n, seed)
#         k = p[0, i]
#         p[0, i] = p[0, j]
#         p[0, j] = k
#     return (p, seed)
#
#
# def perm_random(n, seed):
#     p = N.zeros([1, n], N.int)
#     for i in range(n):
#         p[0, i] = i
#     for i in range(n):
#         (j, seed) = i4_uniform(i + 1, n, seed)
#         k = p[0, i]
#         p[0, i] = p[0, j]
#         p[0, j] = k
#     return (p, seed)
#=========================================================================

#=========================================================================
# def clhc(dim, n1, bounds):
#     n = n1 - 2 ** dim
#     r = N.zeros([n, dim])
#     perm = N.zeros([1, n], N.int)
#     seed = 1051981636
#     for i in range(dim):
#         (perm, seed) = cperm_random(n, seed)
#         for j in range(n):
#             r[j, i] = float(2 * perm[0, j] + 1) / float(2 * n)
#     m = 2 ** dim
#     r = N.concatenate((N.zeros([m, dim]), r), 0)
#     yy = mat_yy(dim)
#     for i in range(m):
#         for j in range(dim):
#             if yy[i, j] > 0.:
#                 r[i, j] = 1.
#     for i in range(dim):
#         b = bounds[0, i]
#         a = bounds[1, i] - b
#         for j in range(n1):
#             r[j, i] = a * r[j, i] + b
#     return r
#=========================================================================


#=========================================================================
# def rlhc(dim, n1, bounds):
#     n = n1 - 2 ** dim
#     r = N.zeros([n, dim])
#     perm = N.zeros([1, n], N.int)
#     seed = 1051981636
#     for i in range(dim):
#         for j in range(n):
#             (r[j, i], seed) = uniform_01(seed)
#     for i in range(dim):
#         (perm, seed) = perm_random(n, seed)
#         for j in range(n):
#             r[j, i] = (float(perm[0, j]) + r[j, i]) / float(n)
#     m = 2 ** dim
#     r = N.concatenate((N.zeros([m, dim]), r), 0)
#     yy = mat_yy(dim)
#     for i in range(m):
#         for j in range(dim):
#             if yy[i, j] > 0.:
#                 r[i, j] = 1.
#     for i in range(dim):
#         b = bounds[0, i]
#         a = bounds[1, i] - b
#         for j in range(n1):
#             r[j, i] = a * r[j, i] + b
#     return r
#=========================================================================


#=========================================================================
# def i4_bit_hi1(n):
#     i = n
#     bit = 0
#     while i > 0:
#         bit = bit + 1
#         i = i / 2
#     return bit
#
#
# def i4_bit_lo0(n):
#     i = n
#     bit = 1
#     i2 = i / 2
#     while i != 2 * i2:
#         i = i2
#         bit = bit + 1
#         i2 = i / 2
#     return bit
#
#
# def i4_xor(i, j):
#     k = 0
#     l = 1
#     while i != 0 or j != 0:
#         i2 = round(i / 2)
#         j2 = round(j / 2)
#         if i == 2 * i2 and j != 2 * j2 or i != 2 * i2 and j == 2 * j2:
#             k = k + l
#         i = i2
#         j = j2
#         l = 2 * l
#     return k
#
#
# def setdatasobol():
#     v = N.zeros([100, 30], N.int)
#     p = N.zeros([100, 1], N.int)
#     v[:, 0:1] = N.ones([100, 1], N.int)
#     vec = [1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3,
#            1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 3, 1, 3, 1, 1, 1, 3, 3, 1,
#            3, 3, 1, 1, 3, 3, 1, 3, 3, 3, 1, 3, 1, 3, 1, 1, 3, 3, 1, 1, 1, 1, 3,
#            1, 1, 3, 1, 1, 1, 3, 3, 1, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 1,
#            3, 3, 3, 1, 3, 1]
#     for i in range(2, 100):
#         v[i, 1] = vec[i - 2]
#     vec = [7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 7, 1, 7, 5,
#            1, 3, 7, 7, 1, 1, 1, 5, 7, 7, 5, 1, 3, 3, 7, 5, 5, 5, 3, 3, 3, 1, 1,
#            5, 1, 1, 5, 3, 3, 3, 3, 1, 3, 7, 5, 7, 3, 7, 1, 3, 3, 5, 1, 3, 5, 5,
#            7, 7, 7, 1, 1, 3, 3, 1, 1, 5, 1, 5, 7, 5, 1, 7, 5, 3, 3, 1, 5, 7, 1,
#            7, 5, 1, 7, 3]
#     for i in range(3, 100):
#         v[i, 2] = vec[i - 3]
#     vec = [1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9,
#            13, 9, 1, 11, 7, 5, 15, 1, 15, 11, 5, 11, 1, 7, 9, 7, 7, 1, 15, 15,
#            15, 13, 3, 3, 15, 5, 9, 7, 13, 3, 7, 5, 11, 9, 1, 9, 1, 5, 7, 13, 9,
#            9, 1, 7, 3, 5, 1, 11, 11, 13, 7, 7, 9, 9, 1, 1, 3, 9, 15, 1, 5, 13,
#            1, 9, 9, 9, 9, 9, 13, 11, 3, 5, 11, 11, 13]
#     for i in range(5, 100):
#         v[i, 3] = vec[i - 5]
#     vec = [9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31,
#            11, 5, 23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 25, 3, 5, 23, 7,
#            3, 17, 23, 3, 3, 21, 25, 25, 23, 11, 19, 3, 11, 31, 7, 9, 5, 17, 23,
#            17, 17, 25, 13, 11, 31, 27, 19, 17, 23, 7, 5, 11, 19, 19, 7, 13, 21,
#            21, 7, 9, 11, 1, 5, 21, 11, 13, 25, 9, 7, 7, 27, 15, 25, 15, 21, 17]
#     for i in range(7, 100):
#         v[i, 4] = vec[i - 7]
#     vec = [37, 33, 7, 5, 11, 39, 63, 59, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9,
#            49, 33, 19, 29, 11, 19, 27, 15, 25, 63, 55, 17, 63, 49, 19, 41, 59,
#            3, 57, 33, 49, 53, 57, 57, 39, 21, 7, 53, 9, 55, 15, 59, 19, 49, 31,
#            3, 39, 5, 5, 41, 9, 19, 9, 57, 25, 1, 15, 51, 11, 19, 61, 53, 29,
#            19, 11, 9, 21, 19, 43, 13, 13, 41, 25, 31, 9, 11, 19, 5, 53]
#     for i in range(13, 100):
#         v[i, 5] = vec[i - 13]
#     vec = [13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3,
#            113, 61, 89, 45, 107, 21, 71, 79, 19, 71, 61, 41, 57, 121, 87, 119,
#            55, 85, 121, 119, 11, 23, 61, 11, 35, 33, 43, 107, 113, 101, 29, 87,
#            119, 97, 29, 17, 89, 5, 127, 89, 119, 117, 103, 105, 41, 83, 25, 41,
#            55, 69, 117, 49, 127, 29, 1, 99, 53, 83, 15, 31, 73, 115, 35, 21,
#            89, 5, 1]
#     for i in range(19, 100):
#         v[i, 6] = vec[i - 19]
#     vec = [7, 23, 39, 217, 141, 27, 53, 181, 169, 35, 15, 207, 45, 247, 185,
#            117, 41, 81, 223, 151, 81, 189, 61, 95, 185, 23, 73, 113, 239, 85,
#            9, 201, 83, 53, 183, 203, 91, 149, 101, 13, 111, 239, 3, 205, 253,
#            247, 121, 189, 169, 179, 197, 175, 217, 249, 195, 95, 63, 19, 7, 5,
#            75, 217, 245]
#     for i in range(37, 100):
#         v[i, 7] = vec[i - 38]
#     vec = [235, 307, 495, 417, 57, 151, 19, 119, 375, 451, 55, 449, 501, 53,
#            185, 317, 17, 21, 487, 13, 347, 393, 15, 391, 307, 189, 381, 71,
#            163, 99, 467, 167, 433, 337, 257, 179, 47, 385, 23, 117, 369, 425,
#            207, 433, 301, 147, 333]
#     for i in range(53, 100):
#         v[i, 8] = vec[i - 53]
#     vec = [1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109,
#            103, 115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
#            213, 191, 253, 203, 211, 239, 247, 285, 369, 299, 301, 333, 351,
#            355, 357, 361, 391, 397, 425, 451, 463, 487, 501, 529, 539, 545,
#            557, 563, 601, 607, 617, 623, 631, 637, 647, 661, 675, 677, 687,
#            695, 701, 719, 721, 731, 757, 761, 787, 789, 799, 803, 817, 827,
#            847, 859, 865, 875, 877, 883, 895, 901, 911, 949, 953, 967, 971,
#            973, 981, 985, 995, 1001]
#     for i in range(100):
#         p[i, 0] = vec[i]
#     return (v, p)
#
#
# def i4_sobol(dim, seed, seed1, lastq1):
#     (v, p) = setdatasobol()
#     atmost = 2 ** 30 - 1
#     maxcol = i4_bit_hi1(atmost)
#     v[0:1, 0:maxcol] = N.ones([1, maxcol], N.int)
#     for i in range(1, dim):
#         j = p[i, 0]
#         m = 0
#         j = round(j / 2)
#         while j > 0:
#             m = m + 1
#             j = round(j / 2)
#         includ = N.zeros([1, m], N.int)
#         j = p[i, 0]
#         for k in range(m - 1, -1, -1):
#             j2 = round(j / 2)
#             includ[0, k] = j != 2 * j2
#             j = j2
#         for j in range(m, maxcol):
#             newv = v[i, j - m]
#             l = 1
#             for k in range(m):
#                 l = 2 * l
#                 if includ[0, k] == 1:
#                     newv = i4_xor(newv, l * v[i, j - k - 1])
#             v[i, j] = newv
#     l = 1
#     for j in range(maxcol - 2, -1, -1):
#         l = 2 * l
#         v[0:dim, j] = v[0:dim, j] * l
#     recip = 1. / float(2 * l)
#     seed = round(seed)
#     if seed < 0:
#         seed = 0
#     if seed == 0:
#         l = 1
#         lastq1[0:1, 0:dim] = N.zeros([1, dim], N.int)
#     elif seed == seed1 + 1:
#         l = i4_bit_lo0(seed)
#     elif seed <= seed1:
#         seed1 = 0
#         l = 1
#         lastq1[0:1, 0:dim] = N.zeros([1, dim], N.int)
#         for seed_temp in range(seed1, seed - 1):
#             l = i4_bit_lo0(seed_temp)
#             for i in range(dim):
#                 lastq1[0, i] = i4_xor(lastq1[0, i], v[i, l])
#         l = i4_bit_lo0(seed)
#     elif seed1 + 1 < seed:
#         for seed_temp in range(seed1 + 1, seed - 1):
#             l = i4_bit_lo0(seed_temp)
#             for i in range(dim):
#                 lastq1[0, i] = i4_xor(lastq1[0, i], v[i, l])
#         l = i4_bit_lo0(seed)
#     r = N.zeros([1, dim])
#     for i in range(dim):
#         r[0, i] = lastq1[0, i] * recip
#         lastq1[0, i] = i4_xor(lastq1[0, i], v[i, l - 1])
#     seed1 = seed
#     seed = seed + 1
#     return (r, seed, seed1, lastq1)
#=========================================================================


#=========================================================================
# def sobol(dim, n1, bounds):
#     n = n1 - 2 ** dim + 1
#     seed = 0
#     seed1 = seed
#     lastq1 = N.zeros([1, dim], N.int)
#     r = N.zeros([n, dim])
#     for i in range(n):
#         (r[i:i + 1, :], seed, seed1, lastq1) = i4_sobol(dim, seed, seed1,
#                                                         lastq1)
#     r = r[1:n, 0:dim]
#     m = 2 ** dim
#     r = N.concatenate((N.zeros([m, dim]), r), 0)
#     yy = mat_yy(dim)
#     for i in range(m):
#         for j in range(dim):
#             if yy[i, j] > 0.:
#                 r[i, j] = 1.
#     for i in range(dim):
#         b = bounds[0, i]
#         a = bounds[1, i] - b
#         for j in range(n1):
#             r[j, i] = a * r[j, i] + b
#     return r
#=========================================================================