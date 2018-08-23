import matplotlib.pyplot as plt
import batman
import numpy as np
from TelApy.tools.studyMASC_UQ import MascaretStudy
from batman.space import (Space, dists_to_ot)
from batman.uq import UQ
from batman.visualization import Kiviat3D, HdrBoxplot, response_surface, Tree
from batman.surrogate import SurrogateModel
from batman.surrogate import (PC, Kriging)
from sklearn.metrics import (r2_score, mean_squared_error)
import openturns as ot

import logging
logging.basicConfig(level=logging.INFO)
###
#This script deals with the PC surrogate convergence study for MascaretAPI on the Garonne 1D case study.
#It quantifiate the truncation and the sampling errors via an estimation of water metrics LH and Coefficients metrics LC.
#The PC degree varies in [3,9] and the sampling sizes varies in [1296,2401,4096,6561,10000] 
###

Study = MascaretStudy('config_Garonne.json', iprint = 0, working_directory = 'study_Convergence')
curv_abs = [13150.0, 13250.0, 13350.0, 13450.0, 13550.0, 13650.0, 13750.0, 13850.0, 13950.0, 14025.0, 14128.333333333334, 14231.666666666668, 14335.0, 14448.333333333334, 14561.666666666668, 14675.0, 14780.0, 14885.0, 14990.0, 15095.0, 15200.0, 15312.5, 15425.0, 15537.5, 15650.0, 15762.5, 15875.0, 15981.25, 16087.5, 16193.75, 16300.0, 16406.25, 16512.5, 16618.75, 16725.0, 16830.833333333332, 16936.666666666664, 17042.499999999996, 17148.33333333333, 17254.16666666666, 17360.0, 17500.0, 17640.0, 17750.0, 17860.0, 17970.0, 18080.0, 18190.0, 18300.0, 18403.571428571428, 18507.142857142855, 18610.714285714283, 18714.28571428571, 18817.857142857138, 18921.428571428565, 19025.0, 19131.25, 19237.5, 19343.75, 19450.0, 19556.25, 19662.5, 19768.75, 19875.0, 19979.166666666668, 20083.333333333336, 20187.500000000004, 20291.66666666667, 20395.83333333334, 20500.0, 20603.125, 20706.25, 20809.375, 20912.5, 21015.625, 21118.75, 21221.875, 21325.0, 21425.0, 21525.0, 21625.0, 21725.0, 21825.0, 21925.0, 22032.0, 22139.0, 22246.0, 22353.0, 22460.0, 22576.25, 22692.5, 22808.75, 22925.0, 23031.5, 23138.0, 23244.5, 23351.0, 23457.5, 23564.0, 23670.5, 23777.0, 23883.5, 23990.0, 24110.0, 24230.0, 24350.0, 24455.0, 24560.0, 24665.0, 24770.0, 24875.0, 24975.0, 25075.0, 25175.0, 25275.0, 25375.0, 25475.0, 25575.0, 25675.0, 25775.0, 25875.0, 25975.0, 26075.0, 26175.0, 26275.0, 26383.333333333332, 26491.666666666664, 26599.999999999996, 26708.33333333333, 26816.66666666666, 26924.999999999993, 27033.333333333325, 27141.666666666657, 27250.0, 27359.375, 27468.75, 27578.125, 27687.5, 27796.875, 27906.25, 28015.625, 28125.0, 28240.0, 28355.0, 28470.0, 28585.0, 28700.0, 28810.0, 28920.0, 29030.0, 29140.0, 29250.0, 29360.0, 29463.0, 29566.0, 29669.0, 29772.0, 29875.0, 29978.0, 30081.0, 30184.0, 30287.0, 30390.0, 30491.0, 30592.0, 30693.0, 30794.0, 30895.0, 30996.0, 31097.0, 31198.0, 31299.0, 31400.0, 31505.0, 31610.0, 31715.0, 31820.0, 31830.0, 31990.0, 32000.0, 32075.0, 32177.14285714286, 32279.285714285717, 32381.428571428576, 32483.571428571435, 32585.714285714294, 32687.857142857152, 32790.0, 32904.166666666664, 33018.33333333333, 33132.49999999999, 33246.66666666666, 33360.83333333332, 33475.0, 33582.142857142855, 33689.28571428571, 33796.428571428565, 33903.57142857142, 34010.714285714275, 34117.85714285713, 34225.0, 34332.142857142855, 34439.28571428571, 34546.428571428565, 34653.57142857142, 34760.714285714275, 34867.85714285713, 34975.0, 35077.5, 35180.0, 35282.5, 35385.0, 35487.5, 35590.0, 35698.333333333336, 35806.66666666667, 35915.00000000001, 36023.33333333334, 36131.66666666668, 36240.0, 36290.0, 36340.0, 36441.666666666664, 36543.33333333333, 36644.99999999999, 36746.66666666666, 36848.33333333332, 36950.0, 37066.666666666664, 37183.33333333333, 37300.0, 37408.333333333336, 37516.66666666667, 37625.0, 37725.0, 37825.0, 37926.36363636364, 38027.72727272728, 38129.09090909092, 38230.45454545456, 38331.8181818182, 38433.18181818184, 38534.54545454548, 38635.90909090912, 38737.27272727276, 38838.6363636364, 38940.0, 39041.666666666664, 39143.33333333333, 39244.99999999999, 39346.66666666666, 39448.33333333332, 39550.0, 39650.0, 39750.0, 39850.0, 39950.0, 40051.666666666664, 40153.33333333333, 40254.99999999999, 40356.66666666666, 40458.33333333332, 40560.0, 40663.0, 40766.0, 40869.0, 40972.0, 41075.0, 41178.0, 41281.0, 41384.0, 41487.0, 41590.0, 41700.0, 41810.0, 41920.0, 42030.0, 42140.0, 42247.0, 42354.0, 42461.0, 42568.0, 42675.0, 42793.75, 42912.5, 43031.25, 43150.0, 43262.5, 43375.0, 43487.5, 43600.0, 43712.5, 43825.0, 43929.166666666664, 44033.33333333333, 44137.49999999999, 44241.66666666666, 44345.83333333332, 44450.0, 44557.5, 44665.0, 44772.5, 44880.0, 44987.5, 45095.0, 45202.5, 45310.0, 45418.333333333336, 45526.66666666667, 45635.00000000001, 45743.33333333334, 45851.66666666668, 45960.0, 46076.0, 46192.0, 46308.0, 46424.0, 46540.0, 46650.625, 46761.25, 46871.875, 46982.5, 47093.125, 47203.75, 47314.375, 47425.0, 47533.125, 47641.25, 47749.375, 47857.5, 47965.625, 48073.75, 48181.875, 48290.0, 48393.333333333336, 48496.66666666667, 48600.00000000001, 48703.33333333334, 48806.66666666668, 48910.0, 49015.555555555555, 49121.11111111111, 49226.666666666664, 49332.22222222222, 49437.777777777774, 49543.33333333333, 49648.88888888888, 49754.44444444444, 49860.0, 49965.0, 50070.0, 50175.0, 50280.0, 50385.0, 50490.0, 50601.666666666664, 50713.33333333333, 50825.0, 50939.166666666664, 51053.33333333333, 51167.49999999999, 51281.66666666666, 51395.83333333332, 51510.0, 51620.833333333336, 51731.66666666667, 51842.50000000001, 51953.33333333334, 52064.16666666668, 52175.0, 52291.25, 52407.5, 52523.75, 52640.0, 52744.375, 52848.75, 52953.125, 53057.5, 53161.875, 53266.25, 53370.625, 53475.0, 53591.666666666664, 53708.33333333333, 53825.0, 53967.5, 54110.0, 54211.875, 54313.75, 54415.625, 54517.5, 54619.375, 54721.25, 54823.125, 54925.0, 55034.375, 55143.75, 55253.125, 55362.5, 55471.875, 55581.25, 55690.625, 55800.0, 55905.0, 56010.0, 56115.0, 56220.0, 56325.0, 56428.125, 56531.25, 56634.375, 56737.5, 56840.625, 56943.75, 57046.875, 57150.0, 57250.0, 57350.0, 57450.0, 57550.0, 57650.0, 57750.0, 57850.0, 57957.142857142855, 58064.28571428571, 58171.428571428565, 58278.57142857142, 58385.714285714275, 58492.85714285713, 58600.0, 58712.0, 58824.0, 58936.0, 59048.0, 59160.0, 59266.92307692308, 59373.846153846156, 59480.769230769234, 59587.69230769231, 59694.61538461539, 59801.53846153847, 59908.461538461546, 60015.384615384624, 60122.3076923077, 60229.23076923078, 60336.15384615386, 60443.07692307694, 60550.0, 60654.545454545456, 60759.09090909091, 60863.63636363637, 60968.18181818182, 61072.72727272728, 61177.272727272735, 61281.81818181819, 61386.36363636365, 61490.9090909091, 61595.45454545456, 61700.0, 61818.75, 61937.5, 62056.25, 62175.0]
dists = ['Uniform(15., 60.)','Uniform(15., 60.)', 'Uniform(15., 60.)', 'BetaMuSigma(4031, 400, 1000, 6000).getDistribution()']
dists_ot = dists_to_ot(dists)

# Parameter space
corners = ([15.0, 15.0, 15.0, 1000.0], [60.0, 60.0, 60.0, 6000.0])  # ([min(X1), min(X2)], [max(X1), max(X2)])
n_x = len(curv_abs)
indim = len(corners)
plabels = ['Ks1','Ks2','Ks3', 'Q']
space = Space(corners)

## PC-LS Strategy ##
#learning sample size for trunctation error
N_learning = 10000

#learning samples sizes for samling error
N_learning5 = 10000
N_learning4 = 6561
N_learning3 = 4096
N_learning2 = 2401
N_learning1 = 1296

# Build the learning samples

x_learning = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning, True, True).generate() 
x_learning = [list(x_learning[i]) for i in range(N_learning)]
x_learning = np.array(x_learning)
x_learning_dico= []

for i in range(N_learning):
    x_learning_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning[i,0]},{"type": "zone", "index": 1, "value": x_learning[i,1]},{"type": "zone", "index": 2, "value": x_learning[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning[i,3]}]})

x_learning5 = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning5, True, True).generate()  
x_learning5 = [list(x_learning5[i]) for i in range(N_learning5)]
x_learning5 = np.array(x_learning5)
x_learning_dico5= []

for i in range(N_learning5):
    x_learning_dico5.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning5[i,0]},{"type": "zone", "index": 1, "value": x_learning5[i,1]},{"type": "zone", "index": 2, "value": x_learning5[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning5[i,3]}]})

x_learning4 = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning4, True, True).generate()
x_learning4 = [list(x_learning4[i]) for i in range(N_learning4)]
x_learning4 = np.array(x_learning4)
x_learning_dico4= []
for i in range(N_learning4):
    x_learning_dico4.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning4[i,0]},{"type": "zone", "index": 1, "value": x_learning4[i,1]},{"type": "zone", "index": 2, "value": x_learning4[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning4[i,3]}]})

x_learning3 = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning3, True, True).generate()
x_learning3 = [list(x_learning3[i]) for i in range(N_learning3)]
x_learning3 = np.array(x_learning3)
x_learning_dico3= []
for i in range(N_learning3):
    x_learning_dico3.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning3[i,0]},{"type": "zone", "index": 1, "value": x_learning3[i,1]},{"type": "zone", "index": 2, "value": x_learning3[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning3[i,3]}]})


x_learning2 = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning2, True, True).generate()
x_learning2 = [list(x_learning2[i]) for i in range(N_learning2)]
x_learning2 = np.array(x_learning2)
x_learning_dico2= []
for i in range(N_learning2):
    x_learning_dico2.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning2[i,0]},{"type": "zone", "index": 1, "value": x_learning2[i,1]},{"type": "zone", "index": 2, "value": x_learning2[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning2[i,3]}]})

x_learning1 = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning1, True, True).generate()
x_learning1 = [list(x_learning1[i]) for i in range(N_learning1)]
x_learning1 = np.array(x_learning1)
x_learning_dico1= []
for i in range(N_learning1):
    x_learning_dico1.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learning1[i,0]},{"type": "zone", "index": 1, "value": x_learning1[i,1]},{"type": "zone", "index": 2, "value": x_learning1[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learning1[i,3]}]})

# Reference study for the coefficients metrics

x_learningr = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_learning, True, True).generate() #training sample for estimation of LC metrics (large init_size)
x_learningr = [list(x_learningr[i]) for i in range(N_learning)]
x_learningr = np.array(x_learningr)
x_learning_dicor= []
for i in range(N_learning):
    x_learning_dicor.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_learningr[i,0]},{"type": "zone", "index": 1, "value": x_learningr[i,1]},{"type": "zone", "index": 2, "value": x_learningr[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_learningr[i,3]}]})


y_learning= []
y_learning5= []
y_learning4= []
y_learning3= []
y_learning2= []
y_learning1= []
y_learningr= []
y_validation=[]

#Compute the solution vector with MascaretAPI y = f(x)

idx = 0
for k in range(N_learning):
    x = x_learning_dico[k]
    print("Study learning#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning.append(Output['z'])
    idx+=1 
    
idx = 0
for k in range(N_learning5):
    x = x_learning_dico5[k]
    print("Study learning5#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning5.append(Output['z'])
    idx+=1 
    
idx = 0
for k in range(N_learning4):
    x = x_learning_dico4[k]
    print("Study learning4#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning4.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_learning3):
    x = x_learning_dico3[k]
    print("Study learning3#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning3.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_learning2):
    x = x_learning_dico2[k]
    print("Study learning2#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning2.append(Output['z'])
    idx+=1
    
idx = 0
for k in range(N_learning1):
    x = x_learning_dico1[k]
    print("Study learning1#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learning1.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_learning):
    x = x_learning_dicor[k]
    print("Study learningr#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_learningr.append(Output['z'])
    idx+=1 

#Build Surrogates   

#Changing degree for truncation error
PC_lsp9 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=9, distributions=dists_ot)
PC_lsp8 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=8, distributions=dists_ot)
PC_lsp7 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_lsp6 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=6, distributions=dists_ot)
PC_lsp5 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=5, distributions=dists_ot)
PC_lsp4 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=4, distributions=dists_ot)
PC_lsp3 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=3, distributions=dists_ot)

#Changing sampling size for samling error
PC_ls5 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_ls4 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_ls3 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_ls2 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_ls1 = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)

# fitting of the validation samples

PC_lsp9.fit(x_learning, y_learning)
PC_lsp8.fit(x_learning, y_learning)
PC_lsp7.fit(x_learning, y_learning)
PC_lsp6.fit(x_learning, y_learning)
PC_lsp5.fit(x_learning, y_learning)
PC_lsp4.fit(x_learning, y_learning)
PC_lsp3.fit(x_learning, y_learning)
PC_ls5.fit(x_learning5, y_learning5)
PC_ls4.fit(x_learning4, y_learning4)
PC_ls3.fit(x_learning3, y_learning3)
PC_ls2.fit(x_learning2, y_learning2)
PC_ls1.fit(x_learning1, y_learning1)



### PC-Quad Strategy ##
    
#Changing degree for truncation error : N_quad is fixed with the argument 'N_quad'
PC_qp9 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=9, distributions=dists_ot, N_quad = 10000)
PC_qp8 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=8, distributions=dists_ot, N_quad = 10000)
PC_qp7 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 10000)
PC_qp6 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 10000)
PC_qp5 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=5, distributions=dists_ot, N_quad = 10000)
PC_qp4 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=4, distributions=dists_ot, N_quad = 10000)
PC_qp3 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=3, distributions=dists_ot, N_quad = 10000)

#Changing sampling size for sampling error : P is fixed with the argument 'degree'
PC_q5 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 10000)
PC_q4 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 6561)
PC_q3 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 4096)
PC_q2 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 2401)
PC_q1 = SurrogateModel('pc', corners, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 1296)

# Get the quadrature points and build the quadrature samples

x_quadp9 = PC_qp9.predictor.sample
x_quadp9 = np.array(x_quadp9)
(N_quadp9,_) = np.shape(x_quadp9)
x_quadp9_dico = []
for i in range(N_quadp9):
    x_quadp9_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp9[i,0]},{"type": "zone", "index": 1, "value": x_quadp9[i,1]},{"type": "zone", "index": 2, "value": x_quadp9[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp9[i,3]}]})

x_quadp8 = PC_qp8.predictor.sample
x_quadp8 = np.array(x_quadp8)
(N_quadp8,_) = np.shape(x_quadp8)
x_quadp8_dico = []
for i in range(N_quadp8):
    x_quadp8_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp8[i,0]},{"type": "zone", "index": 1, "value": x_quadp8[i,1]},{"type": "zone", "index": 2, "value": x_quadp8[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp8[i,3]}]})


x_quadp7 = PC_qp7.predictor.sample
x_quadp7 = np.array(x_quadp7)
(N_quadp7,_) = np.shape(x_quadp7)
x_quadp7_dico = []
for i in range(N_quadp7):
    x_quadp7_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp7[i,0]},{"type": "zone", "index": 1, "value": x_quadp7[i,1]},{"type": "zone", "index": 2, "value": x_quadp7[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp7[i,3]}]})

x_quadp6 = PC_qp6.predictor.sample
x_quadp6 = np.array(x_quadp6)
(N_quadp6,_) = np.shape(x_quadp6)
x_quadp6_dico = []
for i in range(N_quadp6):
    x_quadp6_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp6[i,0]},{"type": "zone", "index": 1, "value": x_quadp6[i,1]},{"type": "zone", "index": 2, "value": x_quadp6[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp6[i,3]}]})

x_quadp5 = PC_qp5.predictor.sample
x_quadp5 = np.array(x_quadp5)
(N_quadp5,_) = np.shape(x_quadp5)
x_quadp5_dico = []
for i in range(N_quadp5):
    x_quadp5_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp5[i,0]},{"type": "zone", "index": 1, "value": x_quadp5[i,1]},{"type": "zone", "index": 2, "value": x_quadp5[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp5[i,3]}]})

x_quadp4 = PC_qp4.predictor.sample
x_quadp4 = np.array(x_quadp4)
(N_quadp4,_) = np.shape(x_quadp4)
x_quadp4_dico = []
for i in range(N_quadp4):
    x_quadp4_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp4[i,0]},{"type": "zone", "index": 1, "value": x_quadp4[i,1]},{"type": "zone", "index": 2, "value": x_quadp4[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp4[i,3]}]})

x_quadp3 = PC_qp3.predictor.sample
x_quadp3 = np.array(x_quadp3)
(N_quadp3,_) = np.shape(x_quadp3)
x_quadp3_dico = []
for i in range(N_quadp3):
    x_quadp3_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadp3[i,0]},{"type": "zone", "index": 1, "value": x_quadp3[i,1]},{"type": "zone", "index": 2, "value": x_quadp3[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadp3[i,3]}]})

x_quad5 = PC_q5.predictor.sample
x_quad5 = np.array(x_quad5)
(N_quad5,_) = np.shape(x_quad5)
x_quad5_dico = []
for i in range(N_quad5):
    x_quad5_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad5[i,0]},{"type": "zone", "index": 1, "value": x_quad5[i,1]},{"type": "zone", "index": 2, "value": x_quad5[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad5[i,3]}]})

x_quad4 = PC_q4.predictor.sample
x_quad4 = np.array(x_quad4)
(N_quad4,_) = np.shape(x_quad4)
x_quad4_dico = []
for i in range(N_quad4):
    x_quad4_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad4[i,0]},{"type": "zone", "index": 1, "value": x_quad4[i,1]},{"type": "zone", "index": 2, "value": x_quad4[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad4[i,3]}]})

x_quad3 = PC_q3.predictor.sample
x_quad3 = np.array(x_quad3)
(N_quad3,_) = np.shape(x_quad3)
x_quad3_dico = []

for i in range(N_quad3):
    x_quad3_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad3[i,0]},{"type": "zone", "index": 1, "value": x_quad3[i,1]},{"type": "zone", "index": 2, "value": x_quad3[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad3[i,3]}]})

x_quad2 = PC_q2.predictor.sample
x_quad2 = np.array(x_quad2)
(N_quad2,_) = np.shape(x_quad2)
x_quad2_dico = []
for i in range(N_quad2):
    x_quad2_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad2[i,0]},{"type": "zone", "index": 1, "value": x_quad2[i,1]},{"type": "zone", "index": 2, "value": x_quad2[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad2[i,3]}]})

x_quad1 = PC_q1.predictor.sample
x_quad1 = np.array(x_quad1)
(N_quad1,_) = np.shape(x_quad1)
x_quad1_dico = []
for i in range(N_quad1):
    x_quad1_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quad1[i,0]},{"type": "zone", "index": 1, "value": x_quad1[i,1]},{"type": "zone", "index": 2, "value": x_quad1[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quad1[i,3]}]})

y_quadp9 = []
y_quadp8 = []
y_quadp7 = []
y_quadp6 = []
y_quadp5 = []
y_quadp4 = []
y_quadp3 = []

y_quad5 = []
y_quad4 = []
y_quad3 = []
y_quad2 = []
y_quad1 = []

# compute the solution on the quadrature points y = f(x)
idx = 0
for k in range(N_quadp9):
    x = x_quadp9[k]
    print("Study quad P=9#"+str(idx))
    Output = Study(x)
    y_quadp9.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quadp8):
    x = x_quadp8[k]
    print(x)
    print("Study quad P=8#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp8.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quadp7):
    x = x_quadp7[k]
    print("Study quad P=7#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp7.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quadp6):
    x = x_quadp6[k]
    print("Study quad P=6#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp6.append(Output['z'])
    idx+=1 
idx = 0

idx = 0
for k in range(N_quadp5):
    x = x_quadp5[k]
    print("Study quad P=5#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp5.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quadp4):
    x = x_quadp4[k]
    print("Study quad P=4#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp4.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quadp3):
    x = x_quadp3[k]
    print("Study quad P=3#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadp3.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quad5):
    x = x_quad5[k]
    print("Study quad5#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quad5.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quad4):
    x = x_quad4[k]
    print("Study quad4#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quad4.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quad3):
    x = x_quad3[k]
    print("Study quad3#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quad3.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quad2):
    x = x_quad2[k]
    print("Study quad2#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quad2.append(Output['z'])
    idx+=1 

idx = 0
for k in range(N_quad1):
    x = x_quad1[k]
    print("Study quad1#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quad1.append(Output['z'])
    idx+=1 

# fitting of the learning samples
    
PC_qp9.fit(x_quadp9, y_quadp9)
PC_qp8.fit(x_quadp8, y_quadp8)
PC_qp7.fit(x_quadp7, y_quadp7)
PC_qp6.fit(x_quadp6, y_quadp6)
PC_qp5.fit(x_quadp5, y_quadp5)
PC_qp4.fit(x_quadp4, y_quadp4)
PC_qp3.fit(x_quadp3, y_quadp3)
PC_q5.fit(x_quad5, y_quad5)
PC_q4.fit(x_quad4, y_quad4)
PC_q3.fit(x_quad3, y_quad3)
PC_q2.fit(x_quad2, y_quad2)
PC_q1.fit(x_quad1, y_quad1)

# Validation
#Build the validation sample
N_validation = 5#10000

x_validation = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),N_validation, True, True).generate() #training sample for truncation error (1 sample)
x_validation = [list(x_validation[i]) for i in range(N_validation)]
x_validation = np.array(x_validation)
x_validation_dico= []
for i in range(N_validation):
    x_validation_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_validation[i,0]},{"type": "zone", "index": 1, "value": x_validation[i,1]},{"type": "zone", "index": 2, "value": x_validation[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_validation[i,3]}]})

#Build the validation vector y =f(x)
idx = 0
for k in range(N_validation):
    x = x_validation_dico[k]
    print("Study Validation#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_validation.append(Output['z'])
    idx+=1

    
# predictions
# LS strategy
y_pred_pc_lsp9, _ = PC_lsp9(x_validation)
y_pred_pc_lsp8, _ = PC_lsp8(x_validation)
y_pred_pc_lsp7, _ = PC_lsp7(x_validation)
y_pred_pc_lsp6, _ = PC_lsp6(x_validation)
y_pred_pc_lsp5, _ = PC_lsp5(x_validation)
y_pred_pc_lsp4, _ = PC_lsp4(x_validation)
y_pred_pc_lsp3, _ = PC_lsp3(x_validation)
y_pred_pc_ls5, _ = PC_ls5(x_validation)
y_pred_pc_ls4, _ = PC_ls4(x_validation)
y_pred_pc_ls3, _ = PC_ls3(x_validation)
y_pred_pc_ls2, _ = PC_ls2(x_validation)
y_pred_pc_ls1, _ = PC_ls1(x_validation)

# Quad Strategy
y_pred_pc_qp9, _ = PC_qp9(x_validation)
y_pred_pc_qp8, _ = PC_qp8(x_validation)
y_pred_pc_qp7, _ = PC_qp7(x_validation)
y_pred_pc_qp6, _ = PC_qp6(x_validation)
y_pred_pc_qp5, _ = PC_qp5(x_validation)
y_pred_pc_qp5, _ = PC_qp5(x_validation)
y_pred_pc_qp4, _ = PC_qp4(x_validation)
y_pred_pc_qp3, _ = PC_qp3(x_validation)
y_pred_pc_q5, _ = PC_q5(x_validation)
y_pred_pc_q4, _ = PC_q4(x_validation)
y_pred_pc_q3, _ = PC_q3(x_validation)
y_pred_pc_q2, _ = PC_q2(x_validation)
y_pred_pc_q1, _ = PC_q1(x_validation)


Lh_pc_qp9 = np.zeros(n_x)
Lh_pc_qp8 = np.zeros(n_x)
Lh_pc_qp7 = np.zeros(n_x)
Lh_pc_qp6 = np.zeros(n_x)
Lh_pc_qp5 = np.zeros(n_x)
Lh_pc_qp4 = np.zeros(n_x)
Lh_pc_qp3 = np.zeros(n_x)
Lh_pc_q5 = np.zeros(n_x)
Lh_pc_q4 = np.zeros(n_x)
Lh_pc_q3 = np.zeros(n_x)
Lh_pc_q2 = np.zeros(n_x)
Lh_pc_q1 = np.zeros(n_x)

y_test = np.array(y_validation)

#LC metrics 

# Reference case
PC_r = SurrogateModel('pc', corners, 10000, plabels, strategy='LS', degree=7, distributions=dists_ot)

# Surrogate cases
PC_quad = SurrogateModel('pc', corners, 4096, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 5)
PC_ls1_LC = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)
PC_ls2_LC = SurrogateModel('pc', corners, plabels, strategy='LS', degree=7, distributions=dists_ot)

x_quadq = PC_quad.predictor.sample
x_quadq = np.array(x_quadq)
(N_quadq,_) = np.shape(x_quadq)
x_quadq_dico = []
for i in range(N_quadq):
    x_quadq_dico.append({'friction_coefficients':[{"type": "zone", "index": 0, "value": x_quadq[i,0]},{"type": "zone", "index": 1, "value": x_quadq[i,1]},{"type": "zone", "index": 2, "value": x_quadq[i,2]}],"boundary_conditions":[{"type": "discharge", "index": 0, "value": x_quadq[i,3]}]})

y_quadq = []

idx = 0
for k in range(N_quadq):
    x = x_quadq[k]
    print("Study quadq#"+str(idx))
    Study.initialize_model()
    Output = Study(x)
    y_quadq.append(Output['z'])
    idx+=1 

# fitting of the validation samples

PC_r.fit(x_learningr,y_learningr)
PC_quad.fit(x_quadq,y_quadq)
PC_ls1_LC.fit(x_learning4,y_learning4)
PC_ls2_LC.fit(x_learning3,y_learning3)


surror = PC_r.predictor.pc_result
surroq = PC_quad.predictor.pc_result
surrols1 = PC_ls1_LC.predictor.pc_result
surrols2 = PC_ls2_LC.predictor.pc_result

# Get the coefficients : Coeffs 
Surror = ot.FunctionalChaosResult(surror)
Coeffsr = Surror.getCoefficients()

Surroq = ot.FunctionalChaosResult(surroq)
Coeffsq = Surroq.getCoefficients()

Surrols1 = ot.FunctionalChaosResult(surrols1)
Coeffls1 = Surrols1.getCoefficients()

Surrols2 = ot.FunctionalChaosResult(surrols2)
Coeffls2 = Surrols2.getCoefficients()


# Compute LC metrics 
g_ref = np.array(Coeffsr)
(n_pc,k) = np.shape(g_ref)
gq = np.array(Coeffsq)
gls1 = np.array(Coeffls1)
gls2 = np.array(Coeffls2)
LC_pc_q = np.zeros(n_pc)
LC_pc_ls1 = np.zeros(n_pc)
LC_pc_ls2 = np.zeros(n_pc)
g = np.zeros(n_pc)
for j in range(n_pc):
    for i in range(k):
        g[j]+=g_ref[j,i]
# Compute LC metrics        
for j in range(n_pc):
    for i in range(k):
        LC_pc_q[j]+= (abs(g_ref[j,i]-gq[j,i])**2)
        LC_pc_ls1[j]+= (abs(g_ref[j,i]-gls1[j,i])**2)
        LC_pc_ls2[j]+= (abs(g_ref[j,i]-gls2[j,i])**2)
        
# compute Lh metrics
for j in range(n_x):
    for i in range(N_validation):
        Lh_pc_qp9[j]+= (y_test[i,j]-y_pred_pc_qp9[i,j])**2
        Lh_pc_qp8[j]+= (y_test[i,j]-y_pred_pc_qp8[i,j])**2
        Lh_pc_qp7[j]+= (y_test[i,j]-y_pred_pc_qp7[i,j])**2
        Lh_pc_qp6[j]+= (y_test[i,j]-y_pred_pc_qp6[i,j])**2
        Lh_pc_qp5[j]+= (y_test[i,j]-y_pred_pc_qp5[i,j])**2
        Lh_pc_qp4[j]+= (y_test[i,j]-y_pred_pc_qp4[i,j])**2
        Lh_pc_qp3[j]+= (y_test[i,j]-y_pred_pc_qp3[i,j])**2
        Lh_pc_q5[j]+= (y_test[i,j]-y_pred_pc_q5[i,j])**2
        Lh_pc_q4[j]+= (y_test[i,j]-y_pred_pc_q4[i,j])**2
        Lh_pc_q3[j]+= (y_test[i,j]-y_pred_pc_q3[i,j])**2
        Lh_pc_q3[j]+= (y_test[i,j]-y_pred_pc_q3[i,j])**2
        Lh_pc_q2[j]+= (y_test[i,j]-y_pred_pc_q2[i,j])**2
        Lh_pc_q1[j]+= (y_test[i,j]-y_pred_pc_q1[i,j])**2
        
Lh_pc_qp9 = Lh_pc_qp9/N_validation
Lh_pc_qp8 = Lh_pc_qp8/N_validation
Lh_pc_qp7 = Lh_pc_qp7/N_validation
Lh_pc_q6 = Lh_pc_qp6/N_validation       
Lh_pc_qp5 = Lh_pc_qp5/N_validation
Lh_pc_qp4 = Lh_pc_qp4/N_validation
Lh_pc_qp3 = Lh_pc_qp3/N_validation
Lh_pc_q5 = Lh_pc_q5/N_validation
Lh_pc_q4 = Lh_pc_q4/N_validation
Lh_pc_q3 = Lh_pc_q3/N_validation
Lh_pc_q2 = Lh_pc_q2/N_validation
Lh_pc_q1 = Lh_pc_q1/N_validation

Lh_pc_lsp9 = np.zeros(n_x)
Lh_pc_lsp8 = np.zeros(n_x)
Lh_pc_lsp7 = np.zeros(n_x)
Lh_pc_lsp6 = np.zeros(n_x)
Lh_pc_lsp5 = np.zeros(n_x)
Lh_pc_lsp4 = np.zeros(n_x)
Lh_pc_lsp3 = np.zeros(n_x)
Lh_pc_ls5 = np.zeros(n_x)
Lh_pc_ls4 = np.zeros(n_x)
Lh_pc_ls3 = np.zeros(n_x)
Lh_pc_ls2 = np.zeros(n_x)
Lh_pc_ls1 = np.zeros(n_x)


for j in range(n_x):
    for i in range(N_validation):
        Lh_pc_lsp9[j]+= (y_test[i,j]-y_pred_pc_lsp9[i,j])**2
        Lh_pc_lsp8[j]+= (y_test[i,j]-y_pred_pc_lsp8[i,j])**2
        Lh_pc_lsp7[j]+= (y_test[i,j]-y_pred_pc_lsp7[i,j])**2
        Lh_pc_lsp6[j]+= (y_test[i,j]-y_pred_pc_lsp6[i,j])**2
        Lh_pc_lsp5[j]+= (y_test[i,j]-y_pred_pc_lsp5[i,j])**2
        Lh_pc_lsp4[j]+= (y_test[i,j]-y_pred_pc_lsp4[i,j])**2
        Lh_pc_lsp3[j]+= (y_test[i,j]-y_pred_pc_lsp3[i,j])**2
        Lh_pc_ls5[j]+= (y_test[i,j]-y_pred_pc_ls5[i,j])**2
        Lh_pc_ls4[j]+= (y_test[i,j]-y_pred_pc_ls4[i,j])**2
        Lh_pc_ls3[j]+= (y_test[i,j]-y_pred_pc_ls3[i,j])**2
        Lh_pc_ls2[j]+= (y_test[i,j]-y_pred_pc_ls2[i,j])**2
        Lh_pc_ls1[j]+= (y_test[i,j]-y_pred_pc_ls1[i,j])**2

Lh_pc_lsp9 = Lh_pc_lsp9/N_validation
Lh_pc_lsp8 = Lh_pc_lsp8/N_validation
Lh_pc_lsp7 = Lh_pc_lsp7/N_validation
Lh_pc_lsp6 = Lh_pc_lsp6/N_validation       
Lh_pc_lsp5 = Lh_pc_lsp5/N_validation
Lh_pc_lsp4 = Lh_pc_lsp4/N_validation
Lh_pc_lsp3 = Lh_pc_lsp3/N_validation
Lh_pc_ls5 = Lh_pc_ls5/N_validation
Lh_pc_ls4 = Lh_pc_ls4/N_validation
Lh_pc_ls3 = Lh_pc_ls3/N_validation
Lh_pc_ls2 = Lh_pc_ls2/N_validation
Lh_pc_ls1 = Lh_pc_ls1/N_validation

#Plot the LH and LC metrics
                
#Samling error Quad 
plt.figure(1)
plt.plot(curv_abs,Lh_pc_q5, '--',label='Nquad=10000')
plt.plot(curv_abs,Lh_pc_q4, '--',label='Nquad=6561')
plt.plot(curv_abs,Lh_pc_q3, '--',label='Nquad=4096')
plt.plot(curv_abs,Lh_pc_q2, '--',label='Nquad=2401')
plt.plot(curv_abs,Lh_pc_q1, '--',label='Nquad=1296')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics LH (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_quad_Sampling.pdf')

#Truncation error Quad
plt.figure(2)
plt.plot(curv_abs,Lh_pc_qp9, '--',label='P=9')
plt.plot(curv_abs,Lh_pc_qp8, '--',label='P=8')
plt.plot(curv_abs,Lh_pc_qp7, '--',label='P=7')
plt.plot(curv_abs,Lh_pc_qp6, '--',label='P=6')
plt.plot(curv_abs,Lh_pc_qp4, '--',label='P=5')
plt.plot(curv_abs,Lh_pc_qp4, '--',label='P=4')
plt.plot(curv_abs,Lh_pc_qp3, '--',label='P=3')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics LH (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_quad_Trunc.pdf')
  

#Samling error LS
plt.figure(3)
plt.plot(curv_abs,Lh_pc_ls5,'--',label='N=10000')
plt.plot(curv_abs,Lh_pc_ls4,'--',label='N=6561')
plt.plot(curv_abs,Lh_pc_ls3,'--',label='N=4096')
plt.plot(curv_abs,Lh_pc_ls2,'--',label='N=2401')
plt.plot(curv_abs,Lh_pc_ls1,'--',label='N=1296')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_LS_Sampling.pdf')

#Truncation error LS
plt.figure(4)
plt.plot(curv_abs,Lh_pc_lsp9,'--',label='P=9')
plt.plot(curv_abs,Lh_pc_lsp8,'--',label='P=8')
plt.plot(curv_abs,Lh_pc_lsp7,'--',label='P=7')
plt.plot(curv_abs,Lh_pc_lsp6,'--',label='P=6')
plt.plot(curv_abs,Lh_pc_lsp5,'--',label='P=5')
plt.plot(curv_abs,Lh_pc_lsp4,'--',label='P=4')
plt.plot(curv_abs,Lh_pc_lsp3,'--',label='P=3')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_LS_Trunc.pdf')
   
#Coefficients erorr
plt.figure(5) 
plt.figure(figsize=(30,5))
plt.plot(LC_pc_q, '--',label='Quad, N=4096')
plt.plot(LC_pc_ls1, '--',label='LS, N=6561')
plt.plot(LC_pc_ls2, '--',label='LS, N=4096')
plt.xlabel('spectrum')
plt.ylabel('Water metrics LC')
plt.title('Coefficients metrics along the Garonne river')
plt.legend()
plt.savefig('LC_Sampling.pdf')

# Log scales

plt.figure(6)
plt.semilogy()
plt.plot(curv_abs,Lh_pc_q5, '--',label='Nquad=10000')
plt.plot(curv_abs,Lh_pc_q4, '--',label='Nquad=6561')
plt.plot(curv_abs,Lh_pc_q3, '--',label='Nquad=4096')
plt.plot(curv_abs,Lh_pc_q2, '--',label='Nquad=2401')
plt.plot(curv_abs,Lh_pc_q1, '--',label='Nquad=1296')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics LH (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_quad_Sampling_log.pdf')

#Truncation error Quad
plt.figure(7)
plt.semilogy()
plt.plot(curv_abs,Lh_pc_qp9, '--',label='P=9')
plt.plot(curv_abs,Lh_pc_qp8, '--',label='P=8')
plt.plot(curv_abs,Lh_pc_qp7, '--',label='P=7')
plt.plot(curv_abs,Lh_pc_qp6, '--',label='P=6')
plt.plot(curv_abs,Lh_pc_qp5, '--',label='P=5')
plt.plot(curv_abs,Lh_pc_qp4, '--',label='P=4')
plt.plot(curv_abs,Lh_pc_qp3, '--',label='P=3')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics LH (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_quad_Trunc_log.pdf')
  

#Samling error LS
plt.figure(8)
plt.semilogy()
plt.plot(curv_abs,Lh_pc_ls5,'--',label='N=10000')
plt.plot(curv_abs,Lh_pc_ls4,'--',label='N=6561')
plt.plot(curv_abs,Lh_pc_ls3,'--',label='N=4096')
plt.plot(curv_abs,Lh_pc_ls2,'--',label='N=2401')
plt.plot(curv_abs,Lh_pc_ls1,'--',label='N=1296')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_LS_Sampling_log.pdf')

#Truncation error LS
plt.figure(9)
plt.semilogy()
plt.plot(curv_abs,Lh_pc_lsp9,'--',label='P=9')
plt.plot(curv_abs,Lh_pc_lsp8,'--',label='P=8')
plt.plot(curv_abs,Lh_pc_lsp7,'--',label='P=7')
plt.plot(curv_abs,Lh_pc_lsp6,'--',label='P=6')
plt.plot(curv_abs,Lh_pc_lsp5,'--',label='P=5')
plt.plot(curv_abs,Lh_pc_lsp4,'--',label='P=4')
plt.plot(curv_abs,Lh_pc_lsp3,'--',label='P=3')
plt.xlabel('curvilinear abscissa (m)')
plt.ylabel('Water metrics (m2)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.savefig('LH_LS_Trunc_log.pdf')
   
#Coefficients erorr
plt.figure(10)
plt.figure(figsize=(30,5))
plt.semilogy() 
plt.plot(LC_pc_q, '--',label='Quad, N=4096')
plt.plot(LC_pc_ls1, '--',label='LS, N=6561')
plt.plot(LC_pc_ls2, '--',label='LS, N=4096')
plt.xlabel('spectrum')
plt.ylabel('Water metrics LC(i)')
plt.title('Coefficients metrics along the Garonne river')
plt.legend()
plt.savefig('LC_Sampling_log.pdf')

