#-*-coding:utf-8-*

import re
import os
import subprocess
import sys
import numpy as np
import h5py
import math
import argparse

n = 0
i = 0
j = 0

wkdir = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("nx", help="display number of points i x direction",
                    type=int)
args = parser.parse_args()

nx = args.nx 
ny = 1 

x = np.zeros((nx,ny))
F = np.zeros((nx,ny))

print "Read HEAD.py"

#------------------------------------------------------------------
#                       INPUT
#-----------------------------------------------------------------

with open('./HEAD.dat', 'r') as a:
        for line in a.readlines():
                A= re.match(r'x1 = (.*$)',line, re.M | re.I)
                if A:
                        x1 = "{:.7}".format(A.group(1))
                        x[i,j]=float(x1)
                        i=i+1
#                       print float(x1)
                if i==20:
                        j=j+1
                        i=0

print "Read FUNC.dat"

i = 0
j = 0

with open('./FUNC.dat', 'r') as a:
        for line in a.readlines():
                A= re.match(r'TITLE = (.*$)',line, re.M | re.I)
                if A:
#                       print "TITLE"   
                        continue
                B= re.match(r'VARIABLES = (.*$)',line, re.M | re.I)
                if B:
#                       print "VARIABLES"
                        continue
                C= re.match(r'ZONE (.*$)',line, re.M | re.I)
                if C:
#                       print "ZONE"
                        continue
#               print "Valeur"
                D= re.match(r'(.*$)',line, re.M | re.I)
                x3 = "{:.16}".format(D.group(1))
#               print float(x3)
                F[i,j]=float(x3)
                i=i+1
                if i==20:
                        j=j+1
                        i=0
                n =n + 1
#print "Total number of points : ", n
#------------------------------------------------------------------
#                       OUTPUT
#-----------------------------------------------------------------

print "Write Function_1D.dat"

j=0

with open('./Function_1D.dat', 'w') as f:
        f.writelines('TITLE = \" FUNCTION 1D \" \n')
        f.writelines('VARIABLES = \"x\" \"F\"  \n')
        f.writelines('ZONE T = \"zone1 \" , I='+str(nx)+', F=BLOCK  \n')
        k=0
        for i in range(nx):
                f.writelines("{:.7E}".format(x[i,j])+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k=0
        for i in range(nx):
                f.writelines("{:.7E}".format(F[i,j])+"\t")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

#-----------------------------------------------------------------

