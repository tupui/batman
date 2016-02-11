#import matplotlib.pyplot as plt
import numpy as N
import point
import refiner
import resampling
import sampling
import space

sampled_space = space.Space(((0, 0), (1, 1)), 60, plot=False)
discretization = 10


bounds = N.array(sampled_space.corners)
limit_number = bounds.shape[1] ** discretization
uniform_space = space.Space(sampled_space.corners, limit_number, plot=False)
uniform_space.sampling('uniform', discretization)

print limit_number
print sampled_space.corners
limit_number = bounds.shape[1] ** discretization
uniform_space = space.Space(sampled_space.corners,limit_number,plot=False)
x = uniform_space.sampling('uniform', discretization)

print len(x)

#u = N.asarray(x).reshape(1, len(x))

u = N.asarray(x)
print u
print type(u)

'''
x1, x2 = N.meshgrid(N.linspace(0, 1, 10),
                     N.linspace(0, 1, 10))
xx = N.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

print xx
print type(xx)
'''
