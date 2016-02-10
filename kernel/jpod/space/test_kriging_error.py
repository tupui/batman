import matplotlib.pyplot as plt
import numpy as N
import point
import refiner
import resampling
import sampling
import space

space2 = space.Space(((0, 0), (1, 1)), 60, plot=False)
discretization = 10


bounds = N.array(space2.corners)
limit_number = bounds.shape[1] ** discretization
uniform_space = space.Space(space2.corners, limit_number, plot=False)
uniform_space.sampling('uniform', discretization)

print limit_number