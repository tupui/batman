#-*-coding:utf-8-*

import re
import os
import subprocess
import sys
import numpy as np
import h5py
import math
import argparse

def function(xx):
    return 1.+xx*xx-xx

n = 0
i = 0
j = 0

wkdir = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("nx", help="display number of points i x direction",
                    type=int)
parser.add_argument("ns", help="display number of snapshots",
                    type=int)
args = parser.parse_args()

nx = args.nx 
ns = args.ns
ny = 1 

x = np.zeros((nx,ny))
xs = np.zeros((ns))
F = np.zeros((nx,ny))
Fct = np.zeros((nx,ny))

print "Read PREDICTIONS -> HEAD.py"

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

print "Read SNAPSHOTS -> SAMP.py"

i = 0

with open('./SAMP.dat', 'r') as a:
        for line in a.readlines():
                A= re.match(r'x1 = (.*$)',line, re.M | re.I)
                if A:
                        x1 = "{:.7}".format(A.group(1))
                        xs[i]=float(x1)
                        i=i+1

print "Read PREDICTIONS -> FUNC.dat"

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

print "Write Sampling_1D.dat"

j=0

with open('./Sampling_1D.dat', 'w') as f:
        f.writelines('TITLE = \" FUNCTION 1D \" \n')
        f.writelines('VARIABLES = \"x\" \"F\"  \n')
        f.writelines('ZONE T = \"zone1 \" , I='+str(ns)+', F=BLOCK  \n')
        k=0
        for i in range(ns):
                f.writelines("{:.7E}".format(xs[i])+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k=0
        for i in range(ns):
                ff = function(xs[i])
                f.writelines("{:.7E}".format(ff)+"\t")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

print "Write Function_1D.dat"

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

print "Write Ref_Function_1D.dat"

with open('./Ref_Function_1D.dat', 'w') as f:
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
                Fct[i,j] = function(x[i,j])
                f.writelines("{:.7E}".format(Fct[i,j])+"\t")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

print "Write Error_1D.dat"

with open('./Error_1D.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"x1\", \"Error\"  \n')
        f.writelines('ZONE T = \"BIG ZONE\" , I='+str(nx)+', J='+str(ny)+', F=BLOCK  \n')

        k=0
        Err_max=0.
        Err_L2=0.
        Err_L2_Fct=0.

        for i in range(nx):
            f.writelines("{:.7E}".format(x[i,j])+"\t ")
            k = k + 1
            if k==7 :
               k=0
               f.writelines('\n')
        f.writelines('\n')
        k = 0
        for i in range(nx):
            Err = 100.*(abs(Fct[i,j]-F[i,j])/abs(Fct[i,j]))
            Err_L2 = Err_L2+(abs(Fct[i,j]-F[i,j]))**2
            Err_L2_Fct =  (Fct[i,j])**2 + Err_L2

            f.writelines("{:.7E}".format(Err)+"\t")

            aaa = [Err,Err_max]
            Err_max=max(aaa)
#          print X1,X2,Err_max,Err,format(Err)

            k = k + 1
            if k==7 :
               k=0
               f.writelines('\n')

#print"Max(F12) :",Fmax
print "Lmax(Error) :", Err_max
print "L2(Error) :", np.sqrt(Err_L2/Err_L2_Fct)
#-----------------------------------------------------------------

