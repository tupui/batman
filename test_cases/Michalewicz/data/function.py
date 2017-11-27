#!/usr/bin/env python
# coding:utf-8

import re
import json
from batman.functions import Michalewicz
from batman.input_output import IOFormatSelector, Dataset

# Input from point.json
with open('./batman-coupling/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']
X2 = params['x2']

# Function
f = Michalewicz()
data = f([X1, X2])

# Output
io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=["F"], shape=[1, 1, 1], data=data)
io.write('./batman-coupling/point.dat', dataset)
