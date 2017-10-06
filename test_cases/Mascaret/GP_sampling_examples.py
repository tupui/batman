from pyuq import Gp1dSampler, Gp2dSampler
import matplotlib.pyplot as plt
import numpy as np

help(Gp1dSampler)

# Construction of the sampler
#sampler = Gp1dSampler(x=[[0.104],[1.]])
sampler = Gp1dSampler()

# Information about the sampler
print(sampler)

print("User points (values):", sampler.x)
print("User points (indices):", sampler.idx)

# Plot of the modes of the Karhunen Loeve Decomposition
sampler.plot_modes()

# Sample of the GP1D and plot the instances
size = 5
Y = sampler.sample(size, plot=True)

# Build a GP1D instance and plot the instances
coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
Y = sampler.build(coeff, plot=True)

#########################################

help(Gp2dSampler)

# Construction of the sampler
sampler = Gp2dSampler()

# Information about the sampler
print(sampler)

# Plot of the modes of the Karhunen Loeve Decomposition
sampler.plot_modes()

# Sample of the GP2D and plot the instance
size = 1
res = sampler.sample(size)['Values']
X, Y = np.meshgrid(np.arange(sampler.t0[0],sampler.T[0],(sampler.T[0]-sampler.t0[0])/sampler.Nt[0]), np.arange(sampler.t0[1],sampler.T[1],(sampler.T[1]-sampler.t0[1])/sampler.Nt[1]))
Z = np.reshape(res[:,0],sampler.Nt)
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()

# Build a GP2D instance and plot the instances
coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
res = sampler.build(coeff)['Values']
X, Y = np.meshgrid(np.arange(sampler.t0[0],sampler.T[0],(sampler.T[0]-sampler.t0[0])/sampler.Nt[0]), np.arange(sampler.t0[1],sampler.T[1],(sampler.T[1]-sampler.t0[1])/sampler.Nt[1]))
Z = np.reshape(res,sampler.Nt)
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()