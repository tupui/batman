#!/usr/bin/env python
#-*-coding:utf-8-*

import re
import os
import subprocess
import sys
import numpy as np
import h5py

wkdir = os.getcwd()


#------------------------------------------------------------------
#                       INPUT 
#-----------------------------------------------------------------

with open('./jpod-data/header.py', 'r') as a:
	for line in a.readlines():
		A= re.match(r'x1 = (.*$)',line, re.M | re.I)
		if A:
			x1 = "{:.7}".format(A.group(1))
print "x1 =", x1 

X1 = float(x1)

#------------------------------------------------------------------
#                       FUNCTION
#-----------------------------------------------------------------

F = X1 * X1

#------------------------------------------------------------------
#                       Output
#-----------------------------------------------------------------

with open('./cfd-output-data/function.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"F\"  \n')
        f.writelines('ZONE F = \"zone1\" , I='+str(1)+', F=BLOCK  \n')
        f.writelines("{:.7E}".format(F)+"\t ")
        f.writelines('\n')

