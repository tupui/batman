#!/usr/bin/env python
# coding:utf-8

import json
import numpy as np
from batman.input_output import formater
from batman.functions import Channel_Flow

io = formater('json')

# Input from sample-space.json
params = io.read('./batman-coupling/sample-space.json', ['x1', 'x2'])

Ks, Q = params[0, :]
f = Channel_Flow()
X = f.x
Z = f([Ks, Q])

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(X, h)
# plt.show()

# Output
nb_value = np.size(X)

names = ['X', 'F']
data = np.append(np.reshape(X, (-1, 1)), np.reshape(Z, (-1, 1)), axis=1)
io.write('./batman-coupling/sample-data.json', data, names)
