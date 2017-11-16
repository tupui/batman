#!/usr/bin/env python
# coding:utf-8

import re
import json
import numpy as np
import ctypes
import csv
from batman.input_output import (IOFormatSelector, Dataset)
from batman.functions import MascaretApi
from batman.functions.mascaret import print_statistics, histogram, plot_opt

study = MascaretApi('config_canal.json', 'config_canal_user_KsQ.json')
print(study)

# Input from point.json
with open('./batman-data/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']
X2 = params['x2']

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
