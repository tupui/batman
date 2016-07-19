#!/usr/bin/env python
#-*-coding:utf-8-*

import re
import os
import subprocess
import sys
import numpy as np
import h5py
import math

wkdir = os.getcwd()


#------------------------------------------------------------------
#                       INPUT 
#-----------------------------------------------------------------

with open('./jpod-data/header.py', 'r') as a:
    for line in a.readlines():
        A= re.match(r'x1 = (.*$)',line, re.M | re.I)
        if A:
            x1 = "{:.7}".format(A.group(1))
        B= re.match(r'x2 = (.*$)',line, re.M | re.I)
        if B:
            x2 = "{:.7}".format(B.group(1))

X1 = float(x1)
X2 = float(x2)

#------------------------------------------------------------------
#                       FUNCTION
#-----------------------------------------------------------------

F = -1.0-math.sin(X1)*(math.pow(math.sin(X1*X1/math.pi),20.))-math.sin(X2)*(math.pow(math.sin(2*X2*X2/math.pi),20.))

#------------------------------------------------------------------
#                       Output
#-----------------------------------------------------------------

with open('./cfd-output-data/function.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"F\"  \n')
        f.writelines('ZONE F = \"zone1\" , I='+str(1)+', F=BLOCK  \n')
        f.writelines("{:.7E}".format(F)+"\t ")
        f.writelines('\n')

