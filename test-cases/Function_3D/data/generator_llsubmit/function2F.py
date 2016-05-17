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
		C= re.match(r'x3 = (.*$)',line, re.M | re.I)
                if C:
                        x3 = "{:.7}".format(C.group(1))
print "X1 =", x1 
print "X2 =", x2
print "X3 =", x3

X1 = float(x1)
X2 = float(x2)
X3 = float(x3)

#------------------------------------------------------------------
#                       FUNCTION
#-----------------------------------------------------------------

F = np.sin(X1)+7*np.sin(X2)**2+0.1*(X3**4)*np.sin(X1)
F2 = np.sin(X1)+7*np.sin(X2)**2+0.1*(X3**4)*np.sin(X1)+2*X1 

#------------------------------------------------------------------
#                       Output
#-----------------------------------------------------------------

with open('./cfd-output-data/function.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"X\" \"F\"  \n')
        f.writelines('ZONE F = \"zone1\" , I='+str(2)+', F=BLOCK  \n')
        f.writelines("{:.7E}".format(1)+"\t ")
        f.writelines("{:.7E}".format(2)+"\t ")
        f.writelines('\n')
        f.writelines("{:.7E}".format(F)+"\t ")
        f.writelines("{:.7E}".format(F2)+"\t ")
        f.writelines('\n')

