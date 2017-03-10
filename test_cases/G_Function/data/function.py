#!/usr/bin/env python
# coding:utf-8

import re
from batman.functions import G_Function
from batman.input_output import (IOFormatSelector, Dataset)

# Input from header.py
with open('./batman-data/header.py', 'r') as a:
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
f = G_Function(d=4)
data = f(X)

# Output
io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=["F"], shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.dat', dataset)
