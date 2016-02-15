from matplotlib import cm
from sklearn.gaussian_process import GaussianProcess

from algebra import Kriging
import matplotlib.pyplot as plt
import numpy as np
from space import Point
import space

# Data from
# http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_probabilistic_classification_after_regression.html#example-gaussian-process-plot-gp-probabilistic-classification-after-regression-py
# Testing of the MSE value


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
yt = y.reshape(8, 1)

on_y_croit = np.append(yt, yt, axis=1)
blabla = on_y_croit.reshape(8, 2)
print blabla

test = Kriging(X, yt)
test27 = Kriging(X, blabla)

what = test27.evaluate((0, 0))
print what
#A = Point((-2, -2))

borne = ((np.amin(X[:, 0]), np.amin(X[:, 1])),
         (np.amax(X[:, 0]), np.amax(X[:, 1])))

espace = space.Space(borne, 200, plot=False)
espace.sampling('halton', 10)

print len(espace)

f, grid = test.error_estimation(espace, 20)
z = f.reshape((20, 20))

print f.shape
Xgrid = grid[:, 0].reshape((20, 20))
Ygrid = grid[:, 1].reshape((20, 20))
fsol = f.reshape((20, 20))

#---- Plot
levels = np.linspace(f.min(), f.max(), 10)

plt.contourf(Xgrid, Ygrid, fsol, cmap=cm.jet, levels=levels)
plt.scatter(X[:, 0].reshape((1, 8)), X[:, 1].reshape(
    (1, 8)), marker='o', c='b', s=30, zorder=10)
plt.title('Kriging MSE estimation')
plt.show()
