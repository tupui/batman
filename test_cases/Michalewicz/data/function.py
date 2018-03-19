#!/usr/bin/env python
# coding:utf-8

from batman.functions import Michalewicz
from batman.input_output import formater

io = formater('npy')

# Input from sample-space.npy
params = io.read('./batman-coupling/sample-space.npy', ['x1', 'x2'])
X1, X2 = params[0, :]

# Function
f = Michalewicz()
data = f([X1, X2])

# Output
io = formater('npz')
io.write('./batman-coupling/sample-data.npy', data, ['F'])
