#!/usr/bin/env python
# coding:utf-8

import json
import numpy as np
from batman.functions import Channel_Flow

# Input from point.json
with open('./batman-coupling/point.json', 'r') as fd:
    params = json.load(fd)

Ks = params['x1']
Q = params['x2']

f = Channel_Flow()
X = f.x
Z = f([Ks, Q])

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(X, h)
# plt.show()

# Output
nb_value = np.size(X)
with open('./batman-coupling/point.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES = \"X\" \"F\"  \n')
    f.writelines('ZONE T=\"zone1\" , I=' + str(nb_value) + ', F=BLOCK  \n')
    for i, _ in enumerate(X):
        f.writelines("{:.7E}".format(float(X[i])) + "\t ")
    f.write('\n')

    for i, _ in enumerate(Z):
        f.writelines("{:.7E}".format(float(Z[i])) + "\t ")
    f.write('\n')
