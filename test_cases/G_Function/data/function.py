#!/usr/bin/env python
# coding:utf-8

import re
import json
from batman.functions import G_Function
from batman.input_output import (IOFormatSelector, Dataset)

# Input from point.json
with open('./batman-data/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']
X2 = params['x2']
X3 = params['x3']
X4 = params['x4']

X = [X1, X2, X3, X4]

# Function
f = G_Function(d=4)
data = f(X)

# Output
io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=["F"], shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.dat', dataset)
