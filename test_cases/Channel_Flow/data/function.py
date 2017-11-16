#!/usr/bin/env python
# coding:utf-8

import re
import json
import numpy as np
from batman.functions import Channel_Flow

# Input from point.json
with open('./batman-data/point.json', 'r') as fd:
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
with open('./cfd-output-data/function.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES = \"X\" \"F\"  \n')
    f.writelines('ZONE T=\"zone1\" , I=' + str(nb_value) + ', F=BLOCK  \n')
    for i in range(len(X)):
        f.writelines("{:.7E}".format(float(X[i])) + "\t ")
        if i % 1000:
            f.writelines('\n')
    f.writelines('\n')

    for i in range(len(Z)):
        f.writelines("{:.7E}".format(float(Z[i])) + "\t ")
        if i % 1000:
            f.writelines('\n')
        f.writelines('\n')
