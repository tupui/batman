#!/usr/bin/env python
# coding:utf-8

import re
from jpod.functions import Ishigami

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

X1 = float(x1)
X2 = float(x2)
X3 = float(x3)

# Function
f = Ishigami()

F = f([X1, X2, X3])

# Output
with open('./cfd-output-data/function.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES =  \"F\"  \n')
    f.writelines('ZONE F = \"zone1\" , I=' + str(1) + ', F=BLOCK  \n')
    f.writelines("{:.7E}".format(F) + "\t ")
    f.writelines('\n')
