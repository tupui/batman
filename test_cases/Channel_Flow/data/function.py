#!/usr/bin/env python
# coding:utf-8

import json
import numpy as np
from batman.input_output import FORMATER
from batman.functions import Channel_Flow

io = FORMATER['json']

# Input from sample-coord.json
params = io.read('./batman-coupling/sample-coord.json', ['x1', 'x2'])

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

# with open('./batman-coupling/point.dat', 'w') as f:
#     f.writelines('TITLE = \"FUNCTION\" \n')
#     f.writelines('VARIABLES = \"X\" \"F\"  \n')
#     f.writelines('ZONE T=\"zone1\" , I=' + str(nb_value) + ', F=BLOCK  \n')
#     for i, _ in enumerate(X):
#         f.writelines("{:.7E}".format(float(X[i])) + "\t ")
#     f.write('\n')
# 
#     for i, _ in enumerate(Z):
#         f.writelines("{:.7E}".format(float(Z[i])) + "\t ")
#     f.write('\n')
