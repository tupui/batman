#!/usr/bin/env python
# coding:utf-8

import re
import json
import numpy as np
import ctypes
import csv
from batman.input_output import (IOFormatSelector, Dataset)
from batman.functions import MascaretApi
from batman.functions.telemac_mascaret import print_statistics, histogram, plot_opt

study = MascaretApi('config_canal.json','config_canal_user.json')
#print(study)


# Input from header.py
with open('./batman-data/header.py', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']
X2 = params['x2']

X1 = float(X1)
X2 = float(X2)

# Function
X, F = study(x=[X1, X2])
print('Water level', F)
plot_opt('ResultatsOpthyca.opt')

# Output
nb_value = np.size(X)
with open('./cfd-output-data/function.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES = \"X\" \"F\"  \n')
    f.writelines('ZONE T=\"zone1\" , I=' + str(nb_value) + ', F=BLOCK  \n')
    for i in range(len(X)):
        f.writelines("{:.7E}".format(float(X[i])) + "\t ")
        if i % 1000:
            f.writelines('\n')
    f.writelines('\n')

    for i in range(len(F)):
        f.writelines("{:.7E}".format(float(F[i])) + "\t ")
        if i % 1000:
            f.writelines('\n')
        f.writelines('\n')
