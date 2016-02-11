#import matplotlib.pyplot as plt
import numpy as np
from space import Point
import space
#import refiner
#import resampling
#import sampling
#import space
from algebra import Kriging

from sklearn.gaussian_process import GaussianProcess


def g(x):
    """The function to predict (classification will then consist in predicting
    whether g(x) <= 0 or not)"""
    return 5. - x[:, 1] - .5 * x[:, 0] ** 2.

# Design of experiments
X = np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [0.00000000, -0.50000000],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [-5.03823144, 3.10584743],
              [-2.87600388, 6.74310541],
              [5.21301203, 4.26386883]])

# Observations
y = g(X)
yt = y.reshape(8,1)

gp = GaussianProcess(theta0=5e-1)

# Don't perform MLE or you'll get a perfect prediction for this simple example!


gp.fit(X, yt)


test = Kriging(X,yt)

A = Point((-2,-2))

b = test.evaluate(A)

espace = space.Space(((0,0),(1,1)), 100, plot=False)
espace.sampling('halton',10)

test.error_estimation(espace,10)

# Instanciate and fit Gaussian Process Model

