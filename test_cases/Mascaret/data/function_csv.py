#!/usr/bin/env python
# coding:utf-8

import re
import json
import numpy as np
import ctypes
import csv
from batman.input_output import (IOFormatSelector, Dataset)
from batman.functions import MascaretApi

study = MascaretApi('config_canal.json','config_canal_user.json')
print(study)

# Input from point.json
with open('./batman-data/point.json', 'r') as fd:
    params = json.load(fd)

X1 = params['x1']

# Function
nb_timebc = 10
nb_bc = 2
vect_in_timebc = np.zeros(nb_timebc)
mat_in_BC = np.zeros((nb_timebc, nb_bc))
with open('My_BC.csv', newline = '' ) as csvfile:
    myreader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
    i = 0
    for row in myreader:
        print (row[0], row[1], row[2])
        vect_in_timebc[i] = row[0]
        mat_in_BC[i,0] = row[1]
        mat_in_BC[i,1] = row[2]
        i = i+1
print (vect_in_timebc)
print (mat_in_BC)
tab_timebc_c =(ctypes.c_double*nb_timebc)()
for j in range(nb_timebc):
    tab_timebc_c[j] = vect_in_timebc[j]
tab_CL1_c = (ctypes.POINTER(ctypes.c_double)*nb_bc)()
tab_CL2_c = (ctypes.POINTER(ctypes.c_double)*nb_bc)()
for i in range(nb_bc):
    tab_CL1_c[i] = (ctypes.c_double*nb_timebc)()
    tab_CL2_c[i] = (ctypes.c_double*nb_timebc)()
    for j in range(nb_timebc):
        tab_CL1_c[i][j] = mat_in_BC[j][i]
        tab_CL2_c[i][j] = 0.
F = study(Qtime = [nb_timebc, tab_timebc_c, tab_CL1_c, tab_CL2_c])
print('Water level computed with user defined BC matrix', F)

# Output
data = np.array(F)
names = ["F"]

io = IOFormatSelector('fmt_tp_fortran')
dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
io.write('./cfd-output-data/function.dat', dataset)
