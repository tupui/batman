import pickle
import os
import numpy as np
from batman.visualization import HdrBoxplot


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

hdr = HdrBoxplot(res, variance=0.9)

hdr.plot(fname='myhdrplot_Garonne_PC121.pdf',
         x_common=x,
         xlabel='Curv. Abs (m)',
         ylabel='waterlevel (m)',
         samples=100)

hdr.f_hops(x_common=x, 
           xlabel='Curv. Abs (m)',
           ylabel='waterlevel (m)',
           fname='myhdrplot_hops_Garonne_PC121.mp4')

#with open('MascaretOut_Garonne_PC121', "wb") as pick_file:
#     pickle.dump(res, pick_file)
#with open('MascaretOut_Garonne_PC121', "rb") as f:
#     unpickler=pickle.Unpickler(f)
#     res2=unpickler.load()
#print(res2)
