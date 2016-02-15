import matplotlib.pyplot as plt
import numpy as np
import point
import refiner
import resampling
import sampling
import space

# Test de point.py
A = point.Point((1, 2))
B = point.Point((5, 2))

# Test de sampling.py
print "Test de sampling \n"

print "mat_yy(3)\n", sampling.mat_yy(3)

print "prime(2)\n", sampling.prime(2)

print "setparhalton(2)\n", sampling.setparhalton(3)

bound = np.array([[0, 0], [10, 10]])
halton = sampling.halton(2, 100, bound)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(halton[:, 0], halton[:, 1])
plt.axis([0, 10, 0, 10])
plt.title('Halton Sequence')

print "uniform_01\n", sampling.uniform_01(0), sampling.uniform_01(1), sampling.uniform_01(2)
print "round\n", sampling.round(9.49)  # Arondi....
print "i4_uniform", sampling.i4_uniform(0, 10, 2)

bound = np.array([[0, 0], [10, 10]])
halton = sampling.clhc(2, 100, bound)
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.scatter(halton[:, 0], halton[:, 1])
plt.axis([0, 10, 0, 10])
plt.title('clhc')
plt.show()
