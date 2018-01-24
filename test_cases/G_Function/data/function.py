#!/usr/bin/env python
# coding:utf-8

import json
from batman.functions import G_Function
from batman.input_output import FORMATER

io = FORMATER['csv']

# Input from sample-coord.csv
params = io.read('./batman-coupling/sample-coord.csv', ['x1', 'x2', 'x3', 'x4'])
# X1, X2, X3, X4 = params[0, :]
# X = [X1, X2, X3, X4]
X = list(params.flat)

# Function
f = G_Function(d=4)
data = f(X)

# Output
io.write('./batman-coupling/sample-data.csv', data, ['F'])
