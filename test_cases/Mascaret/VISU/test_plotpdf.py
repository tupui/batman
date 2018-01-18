import pickle
import os
import numpy as np
from batman.visualization import  pdf

num_abscurv=220

listsnap=os.listdir(path='/space/ricci/BATMAN/test_cases/Mascaret/OUTPUTS/output6_garonne_pcQuad_P=10_KsQ_15-60_1000-6000_sobol50000_gausstronc/snapshots')
print(listsnap)
print(len(listsnap))

file ='x_garonne.dat'
x = np.loadtxt(file, skiprows=3)
x = x[0:463]

res = np.zeros((len(listsnap), 463))
for i in range(len(listsnap)):
    file ='/space/ricci/BATMAN/test_cases/Mascaret/OUTPUTS/output6_garonne_pcQuad_P=10_KsQ_15-60_1000-6000_sobol50000_gausstronc/snapshots/'+listsnap[i]+'/batman-data/function.dat'
    res[i, :] = np.loadtxt(file, skiprows=466)
print(np.shape(res))

filefig = 'waterlevelpdf_x='+str(num_abscurv)+'.pdf'
myflabel =  'Water level(m) x='+str(num_abscurv)
pdf(res[:, num_abscurv].reshape(-1, 1), flabel = myflabel, fname=filefig)

