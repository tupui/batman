import pickle
import os
import numpy as np
from batman.visualization import Kiviat3D

file='/space/ricci/BATMAN/test_cases/Mascaret/OUTPUTS/output6_garonne_pcQuad_P=10_KsQ_15-60_1000-6000_sobol50000_gausstronc/surrogate/space.dat'
space = np.loadtxt(file)
print(input)


file ='x_garonne.dat'
x = np.loadtxt(file, skiprows=3)
x = x[0:463]

listsnap=os.listdir(path='/space/ricci/BATMAN/test_cases/Mascaret/OUTPUTS/output6_garonne_pcQuad_P=10_KsQ_15-60_1000-6000_sobol50000_gausstronc/snapshots')
print(listsnap)
print(len(listsnap))

res = np.zeros((len(listsnap), 463))
for i in range(len(listsnap)):
    file ='/space/ricci/BATMAN/test_cases/Mascaret/OUTPUTS/output6_garonne_pcQuad_P=10_KsQ_15-60_1000-6000_sobol50000_gausstronc/snapshots/'+listsnap[i]+'/batman-data/function.dat'
    res[i, :] = np.loadtxt(file, skiprows=466)
print(np.shape(res))

settings = {
    "corners": [[15.0, 1000.0],[60.0, 6000.0]]
    }
kiviat = Kiviat3D(space, res, settings['corners'], plabels=['Ks','Q','-'])
kiviat.plot(fname='myKiviat_Garonne_PC121.pdf')
kiviat.f_hops(fname = 'myKiviat_hops_Garonne_PC121.mp4', frame_rate=400)
