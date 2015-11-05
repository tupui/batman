from math import *
import numpy as N
from functools import partial


def f1(n, x):
    m = 10
    v = N.zeros(n)
    for i in range(len(x)):
        v[0] += - sin(x[i]) * sin( (i+1) * x[i]**2 / pi )**(2*m)
    # v[0]  = - sin(x[0]) * sin( (0+1) * x[0]**2 / pi )**(2*m)
    # v[0] += - sin(x[0]) * sin( (1+1) * x[1]**2 / pi )**(2*m)
    # v[0] = x[0]
    # v[0] += 3
    # v[:] = v[0]
    # return v / N.arange(1,n+1)
    return v[0] * N.exp(N.arange(1,n+1)*v[0]*x[0]/x[1])
