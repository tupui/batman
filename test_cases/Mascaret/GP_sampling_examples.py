# coding: utf8
"""
Examples of Gp Sampling
-----------------------
"""
from batman.space.gp_sampler import GpSampler

# GpSampler - Documentation
help(GpSampler)

# Dimension 1 - Creation of the Gp sampler
sampler = GpSampler()
print(sampler)
sampler.plot_modes()

# Dimension 1 - Selection of a Gp instance from KLd coefficients
coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
instance = sampler(coeff=coeff)
sampler.plot_sample(instance)

# Dimension 1 - Sampling the Gp
sample_size = 10
sample = sampler(sample_size=sample_size)
sampler.plot_sample(sample)

# Dimension 2 - Creation of the Gp sampler
sampler = GpSampler([[0, 0], [1, 1]], "AbsoluteExponential([0.5, 0.5], [1.0])")
print(sampler)
sampler.plot_modes()

# Dimension 2 - Selection of a Gp instance from KLd coefficients
coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
instance = sampler(coeff=coeff)
sampler.plot_sample(instance)

# Dimension 2 - Sampling the Gp
sample_size = 200
sample = sampler(sample_size=sample_size)
sampler.plot_sample(sample)
