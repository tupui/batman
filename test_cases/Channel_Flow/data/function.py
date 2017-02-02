#!/usr/bin/env python
# coding:utf-8

import re
import numpy as np
from jpod.functions import Channel_Flow

# # Input from header.py
with open('./jpod-data/header.py', 'r') as a:
    for line in a.readlines():
        A = re.match(r'x1 = (.*$)', line, re.M | re.I)
        if A:
            x1 = "{:.7}".format(A.group(1))
        B = re.match(r'x2 = (.*$)', line, re.M | re.I)
        if B:
            x2 = "{:.7}".format(B.group(1))

Q = float(x1)
Ks = float(x2)

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
