import matplotlib.pyplot as plt
import numpy as np
import point
import refiner
import resampling
import sampling
import space

# Test de point.py

point = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])

# Test de sampling.py
#a = np.asarray([A])
#print [A, B].shape[1]
test = refiner.QuadTreeRefiner(point).refine([0.5, 0.5])
a = np.array(test)
print test
# print a[:, 0]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(a[:, 0], a[:, 1])
#plt.axis([0, 10, 0, 10])
#plt.title('Halton Sequence')
plt.show()

print resampling.init_space_part(point)
