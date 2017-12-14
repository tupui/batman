#!/usr/bin/env python
# coding:utf-8

import re
import json
import numpy as np
from batman.input_output import (IOFormatSelector, Dataset)

# Input from point.json
with open('./batman-coupling/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']

# Function
F = 5 + X1 + np.cos(X1)

# Output
data = np.array(F)
names = ["F"]

io = IOFormatSelector('numpy')
dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./batman-coupling/point.npz', dataset)
