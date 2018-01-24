#!/usr/bin/env python
# coding:utf-8

import json
from batman.functions import Michalewicz
from batman.input_output import FORMATER

io = FORMATER['npy']

# Input from sample-coord.npy
params = io.read('./batman-coupling/sample-coord.npy', ['x1', 'x2'])
X1, X2 = params[0, :]

# Function
f = Michalewicz()
data = f([X1, X2])

# Output
io = FORMATER['npz']
io.write('./batman-coupling/sample-data.npy', data, ['F'])
