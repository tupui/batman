"""Ishigami function test."""
import re
import json
import numpy as np
from batman.functions import Ishigami
from batman.input_output import FORMATER  # IOFormatSelector, Dataset

# Input from sample-coord.json
io = FORMATER['json']
params = io.read('./batman-coupling/sample-coord.json', ['x1', 'x2', 'x3'])
X1, X2, X3 = params[0, :]

# Function
f = Ishigami()
F = f([X1, X2, X3])

# Output
data = np.array(F)
names = ["F"]

# dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./batman-coupling/sample-data.json', data, ['F'])

