"""Ishigami function test."""
import re
import json
import numpy as np
from batman.functions import Ishigami
from batman.input_output import (IOFormatSelector, Dataset)

# Input from header.py
with open('./batman-data/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']
X2 = params['x2']
X3 = params['x3']

# Function
f = Ishigami()
F = f([X1, X2, X3])

# Output
data = np.array(F)
names = ["F"]

io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.dat', dataset)

