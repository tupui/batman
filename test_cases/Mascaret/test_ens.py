import matplotlib.pyplot as plt
import batman
import numpy as np
from TelApy.tools.studyMASC_UQ import MascaretStudy
from batman.space import (Space, dists_to_ot)
import openturns as ot
import logging
logging.basicConfig(level=logging.INFO)
###
#This script is used to run N simulations of MASCARET code. 
###

# MascaretStudy Object
Study = MascaretStudy('config_Garonne.json', iprint = 0, working_directory = 'study_ens')

# Input space
dists = ['Uniform(30., 40.)', 'Uniform(30., 40.)', 'Uniform(30., 40.)', 'BetaMuSigma(4031, 400, 2000, 5000).getDistribution()']
dists_ot = dists_to_ot(dists)

N_learning = 4
x_learning = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning, True, True).generate() 
x_learning = [list(x_learning[i]) for i in range(N_learning)]
x_learning = np.array(x_learning)

print ('Learning', x_learning)


x_learning_dico = [ ]
for i in range(N_learning):
    x_learning_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning[i,0]},{"type": "zone", "index": 1, "value": x_learning[i,1]},{"type": "zone", "index": 2, "value": x_learning[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning[i,3]}]})


# Run Mascaret for members in x_ens to compute z_ens, q_ens, ... 
h_ens = [ ]
z_ens = [ ]
q_ens = [ ]
idx = 0
for k in range(N_learning):
    x = x_learning_dico[k]
    print ('x_learning_dico[k]=', x_learning_dico[k])
    print("Study#"+str(k))
    print('x=', x )
    Study.initialize_model()
    Output = Study(x)  
    h_ens.append(Output['h'])
    z_ens.append(Output['z'])
    q_ens.append(Output['q'])
    print('Output', Output['h'])
    Study.plot_water(output="WaterElevation_fromJSON {}".format(k))
    Study.save(out_name = 'test_ens', out_idx = k)

del Study
