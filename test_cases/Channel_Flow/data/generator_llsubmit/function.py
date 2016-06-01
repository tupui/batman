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
#print "X1 =", x1 
#print "X2 =", x2

Q = float(x1)
Ks = float(x2)

#------------------------------------------------------------------
#                       FUNCTION
#-----------------------------------------------------------------

L=500.
I=5e-4
g=9.8
dx=100
longueur=40000 # 40000
Long=longueur/dx
hc=np.power((Q**2)/(g*L*L),1./3.);
hn=np.power((Q**2)/(I*L*L*Ks*Ks),3./10.);
hinit=10.
hh=hinit*np.ones(Long);

for i in xrange(2,Long):
    hh[Long-i]=hh[Long-i+1]-dx*I*((1-np.power(hh[Long-i+1]/hn,-10./3.))/(1-np.power(hh[Long-i+1]/hc,-3.)))
h=hh

X=np.arange(dx, longueur+1, dx)

Zref=-X*I
Z=Zref+h

#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.plot(X, h)
#plt.show()

#------------------------------------------------------------------
#                       Output
#-----------------------------------------------------------------

nb_value = np.size(X)

with open('./cfd-output-data/function.dat', 'w') as f:
    f.writelines('TITLE = \"FUNCTION\" \n')
    f.writelines('VARIABLES = \"X\" \"F\"  \n')
    f.writelines('ZONE T=\"zone1\" , I='+str(nb_value)+', F=BLOCK  \n')
    for i in range(len(X)):
        f.writelines("{:.7E}".format(float(X[i]))+"\t ")
	if i % 1000:
            f.writelines('\n')
    f.writelines('\n')

    for i in range(len(h)):
        f.writelines("{:.7E}".format(float(Z[i]))+"\t ")
        if i % 1000:
            f.writelines('\n')
        f.writelines('\n')

#        f.writelines("{:.7E}".format(1)+"\t ")
#        f.writelines("{:.7E}".format(2)+"\t ")
#        f.writelines('\n')
#        f.writelines("{:.7E}".format(F)+"\t ")
#        f.writelines("{:.7E}".format(F2)+"\t ")
#        f.writelines('\n')

