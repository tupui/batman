#!/usr/bin/env python
# coding:utf-8

import re
import numpy as np
from batman.input_output import (IOFormatSelector, Dataset)

# Input from header.py
with open('./batman-data/header.py', 'r') as a:
    for line in a.readlines():
        A = re.match(r'x1 = (.*$)', line, re.M | re.I)
        if A:
            x1 = "{:.7}".format(A.group(1))

X1 = float(x1)

# Function
F = 5 + X1 + np.cos(X1)

# Output
data = np.array(F)
names = ["F"]

io = IOFormatSelector('numpy')
dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.npz', dataset)
