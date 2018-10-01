import matplotlib.pyplot as plt
import batman
import numpy as np
from TelApy.tools.studyMASC_UQ import MascaretStudy
from batman.space import (Space, dists_to_ot)
import openturns as ot
import logging
from batman.surrogate import SurrogateModel

logging.basicConfig(level=logging.INFO)
### 
#This script is used to run N simulations of MASCARET at the quadrature points of a Polynomial surroagte with degree 7 and store the database file.npy. The learning sample is store as x_learn (snapshots) and y_learn (output)
#the validation data is stored as x_val (snapshots) and y_val (output)
###

# MascaretStudy Object
Study = MascaretStudy('config_Garonne.json', iprint = 0, working_directory = 'study_data_quad')
plabels = ['Ks1', 'Ks2', 'Ks3','Q']
dists = ['Uniform(30., 40.)', 'Uniform(30., 40.)', 'Uniform(30., 40.)','BetaMuSigma(5000, 1500, 3000,7000).getDistribution()']
dists_ot = dists_to_ot(dists)
corners = ([30., 30., 30., 3000], [40., 40., 40, 7000])

# Learning sample
# Build the surrogate and get the quadrature points 

PC_data = SurrogateModel('pc', corners, plabels, strategy='QUAD', degree=3, distributions=dists_ot)
x_quad = PC_data.predictor.sample
x_quad = np.array(x_quad)
np.save('x_quad.npy',x_quad)


(N_quad,_) = np.shape(x_quad)
# Build dictionnairy form of x_quad : x_quad_dico

x_quad_dico = []
for i in range(N_quad):
    x_quad_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad[i,0]},{"type": "zone", "index": 1, "value": x_quad[i,1]},{"type": "zone", "index": 2, "value": x_quad[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad[i,3]}]})

## Run Mascaret for members in x_quad to compute Output
# Learning sample
z_learning= []
for k in range(N_quad):
    x = x_quad_dico[k]
    print("Study#"+str(k))
    print ('x_learn_ens[k]=', x)
    Study.initialize_model()
    Output = Study(x)
    Study.plot_water(output="WaterElevation_learning {}".format(k))
    Study.save(out_name = 'learning', out_idx = k)
    z_learning.append(Output['z'])
np.save('y_learning.npy',z_learning)


print('x_quad',x_quad)

