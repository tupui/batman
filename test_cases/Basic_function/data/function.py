#!/usr/bin/env python
# coding:utf-8

import json
import numpy as np
from batman.input_output import FORMATER

io = FORMATER['csv']
params = io.read('./batman-coupling/sample-coord.csv', ['x1'])
X1 = params[0, 0]

# Function
F = 5 + X1 + np.cos(X1)

# Output
io = FORMATER['json']
io.write('./batman-coupling/sample-data.json', F, ['F'])
