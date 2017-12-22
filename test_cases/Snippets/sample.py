#!/usr/bin/env python
# coding:utf-8
"""Sample distribution.

Generate sampling using OpenTURNS. 

"""
import openturns as ot
import numpy as np
import json

n_samples = 100
dists = [ot.Uniform(20., 40.), ot.Normal(2345., 400.)]

settings_path = './'

with open(settings_path + 'settings.json', 'r') as f:
    settings = json.load(f)

distribution = ot.ComposedDistribution(dists, ot.IndependentCopula(len(dists)))
experiment = ot.LHSExperiment(distribution, n_samples, True, True)
sample = np.array(experiment.generate()).tolist()

settings['space']['sampling'] = sample

with open(settings_path + 'settings.json', 'w') as f:
    json.dump(settings, f, indent=4)
