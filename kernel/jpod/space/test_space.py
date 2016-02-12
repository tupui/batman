import matplotlib.pyplot as plt
import numpy as np
import point
import refiner
import resampling
import sampling
import space

# Creation de l'espace vide de point : Corner + nom de pt max
a = space.Space(((0, 0), (1, 1)), 60, plot=False)
print np.array(a.corners)
print np.array(a.corners).shape
# Remplissage de l'espace vide
b = a.sampling('uniform', 3)

# Methode QuadTress autour du point (0.5,0.5)
b.refine_around((0.5, 0.5))
# print b[:]

# Affichage de l'espace et du rafinement

'''
doe = np.array(b)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(doe[:, 0], doe[:, 1])
plt.axis([0, 1, 0, 1])
plt.title('Uniform')
plt.show()
'''
t1 = np.arange(0.0, 5.0, 0.1)
print t1