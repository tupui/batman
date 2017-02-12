#!/usr/bin/env python
# coding:utf-8

import re
import numpy as np

# Input from header.py
with open('./jpod-data/header.py', 'r') as a:
    for line in a.readlines():
        A = re.match(r'x1 = (.*$)', line, re.M | re.I)
        if A:
            x1 = "{:.7}".format(A.group(1))
        B = re.match(r'x2 = (.*$)', line, re.M | re.I)
        if B:
            x2 = "{:.7}".format(B.group(1))
        C = re.match(r'x3 = (.*$)', line, re.M | re.I)
        if C:
            x3 = "{:.7}".format(C.group(1))
        D = re.match(r'x4 = (.*$)', line, re.M | re.I)
        if D:
            x4 = "{:.7}".format(D.group(1))

X1 = float(x1)
X2 = float(x2)
X3 = float(x3)
X4 = float(x4)

X = [X1, X2, X3, X4]

# Function
d = 4
a = np.arange(1, d + 1)
F = 1.
for i in range(d):
    F *= (abs(4. * X[i] - 2) + a[i]) / (1. + a[i])

# Output
with open('./cfd-output-data/function.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES =  \"F\"  \n')
    f.writelines('ZONE F = \"zone1\" , I=' + str(1) + ', F=BLOCK  \n')
    f.writelines("{:.7E}".format(F) + "\t ")
    f.writelines('\n')