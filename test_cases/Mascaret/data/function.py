#!/usr/bin/env python
# coding:utf-8

import re
import numpy as np
import ctypes
import csv
from batman.input_output import (IOFormatSelector, Dataset)
from batman.functions import MascaretApi
from batman.functions.mascaret import print_statistics, histogram, plot_opt

study = MascaretApi('config_canal.json','config_canal_user_KsQ.json')
print(study)


# Input from header.py
with open('./batman-data/header.py', 'r') as a:
    for line in a.readlines():
        A = re.match(r'x1 = (.*$)', line, re.M | re.I)
        if A:
            x1 = "{:.7}".format(A.group(1))
        B = re.match(r'x2 = (.*$)', line, re.M | re.I)
        if B:
            x2 = "{:.7}".format(B.group(1))

X1 = float(x1)
X2 = float(x2)

# Function
F = study(x=[X1, X2])
print('Water level', F)
plot_opt('ResultatsOpthyca.opt')

# Output
data = np.array(F)
names = ["F"]

io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.dat', dataset)
