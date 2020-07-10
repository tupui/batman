"""
Mock plugin file for tests
"""
import numpy as np
from batman.functions import Ishigami, Branin

f_ishigami = Ishigami()
f_branin = Branin()


def f_snapshot(point):
    return np.array([42, 87, 74, 74])
