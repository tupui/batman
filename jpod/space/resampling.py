##  ==========================================================================
##                Project: cfd - POD - Copyright (c) 2009 by CERFACS
##  Type   :
##  File   : resampling.py
##  Vers   : V1.0
##  Chrono : No  Date       Author                 V   Comments
##           1.0 11/08/2009 Braconnier             0.1 Creation
##  ==========================================================================

import numpy as N
from sampling import mat_yy


def splitelement(S, S0, DS):
    mm = S.shape[1]
    yy = mat_yy(mm)
    nn = 2 ** mm
    S1 = (S + S0) / 2.
    DS1 = S - S0
    NewS = N.zeros([nn, mm])
    NewS0 = N.zeros([nn, mm])
    NewDS = N.zeros([nn, mm])
    temp1 = N.zeros([1, mm])
    temp2 = N.zeros([1, mm])
    temp3 = N.zeros([1, mm])
    NewS = N.zeros([nn, mm])
    NewS0 = N.zeros([nn, mm])
    NewDS = N.zeros([nn, mm])
    icur = -1
    for i in range(nn):
        for j in range(mm):
            if yy[i, j] < 0:
                temp1[0, j] = S1[0, j]
                temp2[0, j] = S0[0, j]
                temp3[0, j] = DS1[0, j]
            else:
                temp1[0, j] = S1[0, j] + DS[0, j] / 2.
                temp2[0, j] = S[0, j]
                temp3[0, j] = DS[0, j] - DS1[0, j]
        temp4 = S == temp1
        temp5 = abs(temp3) < 1.e-10
        if N.sum(temp4, 1) == 0. and N.sum(temp5, 1) == 0.:
            icur = icur + 1
            NewS[icur:icur + 1, :] = temp1
            NewS0[icur:icur + 1, :] = temp2
            NewDS[icur:icur + 1, :] = temp3
    NewS = NewS[:icur + 1, :]
    NewS0 = NewS0[:icur + 1, :]
    NewDS = NewDS[:icur + 1, :]
    return (NewS, NewS0, NewDS)


def comptnode(S, index, S0, DS):
    ms = S.shape[1]
    Smin = S0
    Smax = S0 + DS
    n = len(index)
    indexin = []
    indexout = []
    indexbox = []
    for i in range(n):
        testin = 0
        testout = 0
        for j in range(ms):
            if Smin[j] < S[index[i], j] and S[index[i], j] < Smax[j]:
                testin = testin + 1
            elif Smin[j] > S[index[i], j] or S[index[i], j] > Smax[j]:
                testout = ms
        if testin == ms:
            indexin.append(index[i])
        elif testout == ms:
            indexout.append(index[i])
        else:
            indexbox.append(index[i])
    return (indexin, indexout, indexbox)


def find(x, valeur):
    n = x.shape[0]
    index = []
    for i in range(n):
        if x[i, 0] == valeur:
            index.append(i)
    return index


def tri(x):
    (n, m) = N.shape(x)
    index = []
    temp = N.zeros([1, n], N.int)
    for i in range(n):
        icur = 0
        while temp[0, icur] < 0:
            icur = icur + 1
        index.append(icur)
        xmin = x[icur, 0]
        for j in range(icur + 1, n):
            if temp[0, j] == 0:
                if x[j, 0] < xmin:
                    index[i] = j
                    xmin = x[j, 0]
        temp[0, index[i]] = -1
    return index


def rawswap(x, index):
    (n, m) = N.shape(x)
    x1 = N.zeros([n, m])
    for j in range(m):
        for i in range(n):
            x1[i, j] = x[i, j]
    for i in range(n):
        x[i:i + 1, :] = x1[index[i]:index[i] + 1, :]
    return x


def init_space_part(S):
    (np, dim) = N.shape(S)
    part = N.zeros([np, 3 * dim + 4])
    Smm = N.zeros([3, dim])
    for i in range(dim):
        Smm[0, i] = 1e30
        Smm[1, i] = -1e30
        for j in range(np):
            if S[j, i] < Smm[0, i]:
                Smm[0, i] = S[j, i]
            if S[j, i] > Smm[1, i]:
                Smm[1, i] = S[j, i]
        Smm[2, i] = Smm[1, i] - Smm[0, i]
        part[0, dim + i] = Smm[0, i]
        part[0, 2 * dim + i] = Smm[2, i]
    index = range(np)
    (indexin, indexout, indexbox) = comptnode(S, index, Smm[0, :], Smm[2, :])
    part[0, 3 * dim] = np
    part[0, 3 * dim + 3] = len(indexbox)
    endp = 0
    ilist = 0
    while endp < np:
        index = find(part[:, 3 * dim + 1:3 * dim + 2], endp)
        if part[endp, 3 * dim] == 1.:
            endp = endp + 1
        else:
            mk = len(index)
            S1 = N.zeros([mk, dim])
            for i in range(mk):
                S1[i:i + 1, :] = S[index[i], :]
            for i in range(dim):
                Smm[0, i] = 1e30
                Smm[1, i] = -1e30
                for j in range(mk):
                    if S1[j, i] < Smm[0, i]:
                        Smm[0, i] = S1[j, i]
                    if S1[j, i] > Smm[1, i]:
                        Smm[1, i] = S1[j, i]
                Smm[2, i] = Smm[1, i] - Smm[0, i]
            dim1 = 0
            temp = Smm[2, 0]
            for i in range(1, dim):
                if Smm[2, i] >= temp:
                    dim1 = i
                    temp = Smm[2, i]
            part[ilist + 1:ilist + 2, :3 * dim + 1] = part[endp:endp + 1, :3
                    * dim + 1]
            test = 0
            alpha = 0.5
            ilist = ilist + 1
            temp = part[endp, dim + dim1]
            temp1 = part[endp, 2 * dim + dim1]
            while test == 0:
                delta = Smm[0, dim1] + alpha * Smm[2, dim1] - temp
                remind = temp - delta
                part[endp, 2 * dim + dim1] = delta
                part[ilist, 2 * dim + dim1] = temp1 - delta
                part[ilist, dim + dim1] = temp + delta
                (indexin1, indexout1, indexbox1) = comptnode(S, index,
                        part[endp, dim:2 * dim], part[endp, 2 * dim:3 * dim])
                (indexin2, indexout2, indexbox2) = comptnode(S, index,
                        part[ilist, dim:2 * dim], part[ilist, 2 * dim:3 * dim])
                nbox = len(indexbox1) + len(indexbox2)
                if nbox == part[endp, 3 * dim + 3]:
                    test = 1
                else:
                    alpha = alpha - 0.05
            compt = len(indexin1) + len(indexbox1)
            part[endp, 3 * dim] = compt
            part[endp, 3 * dim + 3] = len(indexbox1)
            part[ilist, 3 * dim + 3] = len(indexbox2)
            nout = len(indexout1)
            for i in range(nout):
                part[indexout1[i], 3 * dim + 1] = ilist
            part[ilist, 3 * dim] = nout
            if compt == 1:
                if len(indexin1) == 1:
                    part[endp, 3 * dim + 2] = indexin1[0]
                else:
                    part[endp, 3 * dim + 2] = indexbox1[0]
            if nout == 1:
                part[ilist, 3 * dim + 2] = indexout1[0]
    for i in range(np):
        part[i:i + 1, :dim] = S[int(part[i, 3 * dim + 2]), :]
    index = tri(part[:, 3 * dim + 2:3 * dim + 3])
    part = rawswap(part, index)
    return part


if __name__ == '__main__':
    dim = 2
    S = N.zeros([15, dim])
    k = 0
    for i in range(3):
        for j in range(5):
            S[k, 1] = .6 + float(i) * 0.1
            k = k + 1
    k = 0
    for i in range(3):
        for j in range(5):
            S[k, 0] = float(j)
            k = k + 1
    p = init_space_part(S)
    print p
    n = int(raw_input(' n'))
    (NewS, NewS0, NewDS) = splitelement(p[n:n + 1, 0:dim], p[n:n + 1, dim:2
                                        * dim], p[n:n + 1, 2 * dim:3 * dim])
    (nn, mm) = N.shape(NewS)
    print N.shape(NewS), N.shape(NewS0), N.shape(NewDS)
    for i in range(nn):
        print 'newparam(%d):%s %s %s' % (i + 1, NewS[i, :], NewS0[i, :],
                NewDS[i, :])
