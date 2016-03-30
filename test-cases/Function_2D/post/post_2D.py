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
parser.add_argument("ny", help="display number of points i y direction",
                    type=int)
parser.add_argument("ns", help="display number of snapshots",
                    type=int)
args = parser.parse_args()

nx = args.nx 
ny = args.ny
ns = args.ns

x = np.zeros((nx,ny))
y = np.zeros((nx,ny))
xs = np.zeros((ns))
ys = np.zeros((ns))
F = np.zeros((nx,ny))
F12 = np.zeros((nx,ny))

#------------------------------------------------------------------
#                       YOUR FUNCTION
#-----------------------------------------------------------------

#XB1=1.0
XB2=math.pi
XB1=0.
#YB1=1.0
YB2=math.pi
YB1=0. 

# JCJ extension
#delta_space = 0.05
#XB1 = XB1 - delta_space * (XB2-XB1)
#XB2 = XB2 + delta_space * (XB2-XB1)
#YB1 = YB1 - delta_space * (YB2-YB1)
#YB2 = YB2 + delta_space * (YB2-YB1)

print "Read PREDICTIONS -> HEAD.py"

#------------------------------------------------------------------
#                       INPUT
#-----------------------------------------------------------------

with open('./HEAD.dat', 'r') as a:
        for line in a.readlines():
                A= re.match(r'x1 = (.*$)',line, re.M | re.I)
                if A:
                        x1 = "{:.7}".format(A.group(1))
#                       print "x ",i,j,float(x1)
                        x[i,j]=float(x1)
#                       print float(x1)
                B= re.match(r'x2 = (.*$)',line, re.M | re.I)
                if B:
                        x2 = "{:.7}".format(B.group(1))
#                       print "y ",i,j,float(x2)
                        y[i,j]=float(x2)
                        i=i+1
#                       print float(x2)
                if i==nx:
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
                B= re.match(r'x2 = (.*$)',line, re.M | re.I)
                if B:
                        x2 = "{:.7}".format(B.group(1))
                        ys[i]=float(x2)
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
                j=j+1
                if j==ny:
                        i=i+1
                        j=0
                n =n + 1
#print "Total number of points : ", n
#------------------------------------------------------------------
#                       OUTPUT
#-----------------------------------------------------------------

print "Write Sampling_2D.dat"

with open('./Sampling_2D.dat', 'w') as f:
        for i in range(ns):
                f.writelines("{:.7E}".format(xs[i])+"\t ")
                f.writelines("{:.7E}".format(ys[i])+"\t ")
                f.writelines('\n')

print "Write Function_2D.dat"

with open('./Function_2D.dat', 'w') as f:
        f.writelines('TITLE = \" FUNCTION 1D \" \n')
        f.writelines('VARIABLES = \"x1\", \"x2\", \"F\"  \n')
        f.writelines('ZONE T = \"zone1\" , I='+str(nx)+', J='+str(ny)+', F=BLOCK  \n')
        k=0
        for j in range(ny):
            for i in range(nx):
                f.writelines("{:.7E}".format(x[j,i])+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k = 0
        for j in range(ny):
            for i in range(nx):
                f.writelines("{:.7E}".format(y[j,i])+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k=0
        for i in range(nx):
            for j in range(ny):
                f.writelines("{:.7E}".format(F[j,i])+"\t")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

print "Write Ref_Function_2D.dat"

with open('./Ref_Function_2D.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"x1\", \"x2\", \"F12\"  \n')
        f.writelines('ZONE T = \"BIG ZONE\" , I='+str(nx)+', J='+str(ny)+', F=BLOCK  \n')
        k=0
        for j in range(ny):
            for i in range(nx):
                X1 = XB1+i*(XB2-XB1)/(nx-1)
                f.writelines("{:.7E}".format(X1)+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k = 0
        for j in range(ny):
            for i in range(nx):
                X2 = YB1+j*(YB2-YB1)/(ny-1)
                f.writelines("{:.7E}".format(X2)+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k=0
        for j in range(ny):
            for i in range(nx):
                X1 = XB1+i*(XB2-XB1)/(nx-1)
                X2 = YB1+j*(YB2-YB1)/(ny-1)
                F12[i,j] = -1.0-math.sin(X1)*(math.pow(math.sin(X1*X1/math.pi),20.))-math.sin(X2)*(math.pow(math.sin(2*X2*X2/math.pi),20.))
                f.writelines("{:.7E}".format(F12[i,j])+"\t")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

print "Write Error_2D.dat"

with open('./Error_2D.dat', 'w') as f:
        f.writelines('TITLE = \"FUNCTION\" \n')
        f.writelines('VARIABLES = \"x1\", \"x2\", \"Error\"  \n')
        f.writelines('ZONE T = \"BIG ZONE\" , I='+str(nx)+', J='+str(ny)+', F=BLOCK  \n')

        k=0
        Err_max=0.
        Err_L2=0.
        Err_L2_F12=0.

        for j in range(ny):
            for i in range(nx):
                X1 = XB1+i*(XB2-XB1)/(nx-1)
                f.writelines("{:.7E}".format(X1)+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k = 0
        for j in range(ny):
            for i in range(nx):
                X2 = YB1+j*(YB2-YB1)/(ny-1)
                f.writelines("{:.7E}".format(X2)+"\t ")
                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')
        f.writelines('\n')
        k=0
        for j in range(ny):
            for i in range(nx):
                X1 = XB1+i*(XB2-XB1)/(nx-1)
                X2 = YB1+j*(YB2-YB1)/(ny-1)

                Err = 100.*(abs(F12[i,j]-F[i,j])/abs(F12[i,j]))
                Err_L2 = Err_L2+(abs(F12[i,j]-F[i,j]))**2 
                Err_L2_F12 =  (F12[i,j])**2 + Err_L2

                f.writelines("{:.7E}".format(Err)+"\t")

                aaa = [Err,Err_max]
                Err_max=max(aaa)
                if Err == Err_max :
                   xmax = x1
                   ymax = x2 
                   imax = i
                   jmax = j

                k = k + 1
                if k==7 :
                    k=0
                    f.writelines('\n')

#print"Max(F12) :",Fmax
print "Lmax(Error), xmax, ymax, imax, jmax :", Err_max, xmax, ymax, imax, jmax
print "L2(Error) :", np.sqrt(Err_L2/Err_L2_F12) 

#-----------------------------------------------------------------

