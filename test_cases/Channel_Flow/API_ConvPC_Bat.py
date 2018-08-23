import matplotlib.pyplot as plt
import batman
import numpy as np
from batman.functions.analytical import Channel_Flow
from batman.space import (Space, dists_to_ot)
from batman.uq import UQ
from batman.visualization import Kiviat3D, HdrBoxplot, response_surface, Tree
from batman.surrogate import SurrogateModel
from batman.surrogate import (PC, Kriging)
from sklearn.metrics import (r2_score, mean_squared_error)
import openturns as ot
import logging
logging.basicConfig(level=logging.INFO)

#This script deals with the PC surrogate convergence study for the Channel Flow analitical function.
#It quantifiate the truncation and the sampling errors via an estimation of water metrics LH and Coefficients metrics LC.
#The PC degree varies in [6,10] (index 6-10)and the sampling sizes varies in [40,50,100,150,200] (index 1-5)
#defining parameters for the Channel Flow function

dx = 400.
length = 40000.
n_x = int(length/dx-1)

# Test function
fl = Channel_Flow(dx, length)  # Change dx to increase discretization

curv_abs = [dx*i for i in range(n_x+1)]
print(curv_abs)
print(len(curv_abs))
# Parameter space
corners = ([15.0, 2500.0], [60.0, 6000.0])  # ([min(X1), min(X2)], [max(X1), max(X2)])

#samples sizes for samling error

init_size5 = 200
init_size4 = 150
init_size3 = 100
init_size2 = 50
init_size1 = 40

#sample size used for trunctation error

init_size = 1000

indim = 2  # inputs dim
plabels = ['Ks', 'Q']
space = Space(corners)

# Build the learning samples

x_train = np.array(space.sampling(init_size,'halton')) #training sample for truncation error (1 sample)

x_train5 = np.array(space.sampling(init_size5, 'halton')) #training samples for sampling error (init_size varies) 
x_train4 = np.array(space.sampling(init_size4, 'halton'))
x_train3 = np.array(space.sampling(init_size3, 'halton'))
x_train2 = np.array(space.sampling(init_size2, 'halton'))
x_train1 = np.array(space.sampling(init_size1, 'halton'))

x_trainr = np.array(space.sampling(1000,'halton')) #training sample for estimation of LC metrics (large init_size)


#Build the solution vector with y = f(x)

y_train = fl(x_train)

y_train5= fl(x_train5)
y_train4= fl(x_train4)
y_train3= fl(x_train3)
y_train2= fl(x_train2)
y_train1= fl(x_train1)

y_trainr= fl(x_trainr)



# Build the test sample

test_size = 10000  # test size
dists = ['Uniform(15., 60.)', 'Normal(4035., 400.)']
dists_ot = dists_to_ot(dists)
x_test = ot.LHSExperiment(ot.ComposedDistribution(dists_ot),
                          test_size, True, True).generate()
x_test = np.array(x_test)
y_test = fl(x_test)


# Surrogate

## Polynomial Chaos

### Quad
#Changing degree for truncation error
pc_predictor_q10 = SurrogateModel('pc', corners, init_size, plabels, strategy='Quad', degree=10, distributions=dists_ot, N_quad = 121)
pc_predictor_q9 = SurrogateModel('pc', corners, init_size, plabels, strategy='Quad', degree=9, distributions=dists_ot, N_quad = 121)
pc_predictor_q8 = SurrogateModel('pc', corners, init_size, plabels, strategy='Quad', degree=8, distributions=dists_ot, N_quad = 121)
pc_predictor_q7 = SurrogateModel('pc', corners, init_size, plabels, strategy='Quad', degree=7, distributions=dists_ot, N_quad = 121)
pc_predictor_q6 = SurrogateModel('pc', corners, init_size, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 121)

#Changing sampling size for samling error
pc_predictor_q5 = SurrogateModel('pc', corners, init_size5, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 121)
pc_predictor_q4 = SurrogateModel('pc', corners, init_size4, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 100)
pc_predictor_q3 = SurrogateModel('pc', corners, init_size3, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 81)
pc_predictor_q2 = SurrogateModel('pc', corners, init_size2, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 64)
pc_predictor_q1 = SurrogateModel('pc', corners, init_size1, plabels, strategy='Quad', degree=6, distributions=dists_ot, N_quad = 49)

#Carrefull with the sampling size when using Quad method, Surrogate do not take inte account init-size parameter, 
# it only takes the degree to build the sample by using n_sample = (degree+1)^2

x_quad10 = pc_predictor_q10.predictor.sample
x_quad9 = pc_predictor_q9.predictor.sample
x_quad8 = pc_predictor_q8.predictor.sample
x_quad7 = pc_predictor_q7.predictor.sample
x_quad6 = pc_predictor_q6.predictor.sample
x_quad5 = pc_predictor_q5.predictor.sample
x_quad4 = pc_predictor_q4.predictor.sample
x_quad3 = pc_predictor_q3.predictor.sample
x_quad2 = pc_predictor_q2.predictor.sample
x_quad1 = pc_predictor_q1.predictor.sample

y_quad10 = fl(x_quad10)
y_quad9 = fl(x_quad9)
y_quad8 = fl(x_quad8)
y_quad7 = fl(x_quad7)
y_quad6 = fl(x_quad6)
y_quad5 = fl(x_quad5)
y_quad4 = fl(x_quad4)
y_quad3 = fl(x_quad3)
y_quad2 = fl(x_quad2)
y_quad1 = fl(x_quad1)
#

pc_predictor_q10.fit(x_quad10, y_quad10)
pc_predictor_q9.fit(x_quad9, y_quad9)
pc_predictor_q8.fit(x_quad8, y_quad8)
pc_predictor_q7.fit(x_quad7, y_quad7)
pc_predictor_q6.fit(x_quad6, y_quad6)
pc_predictor_q5.fit(x_quad5, y_quad5)
pc_predictor_q4.fit(x_quad4, y_quad4)
pc_predictor_q3.fit(x_quad3, y_quad3)
pc_predictor_q2.fit(x_quad2, y_quad2)
pc_predictor_q1.fit(x_quad1, y_quad1)

y_pred_pc_q10, _ = pc_predictor_q10(x_test)
y_pred_pc_q9, _ = pc_predictor_q9(x_test)
y_pred_pc_q8, _ = pc_predictor_q8(x_test)
y_pred_pc_q7, _ = pc_predictor_q7(x_test)
y_pred_pc_q6, _ = pc_predictor_q6(x_test)
y_pred_pc_q5, _ = pc_predictor_q5(x_test)
y_pred_pc_q4, _ = pc_predictor_q4(x_test)
y_pred_pc_q3, _ = pc_predictor_q3(x_test)
y_pred_pc_q2, _ = pc_predictor_q2(x_test)
y_pred_pc_q1, _ = pc_predictor_q1(x_test)


Lh_pc_q10 = np.zeros(n_x+1)
Lh_pc_q9 = np.zeros(n_x+1)
Lh_pc_q8 = np.zeros(n_x+1)
Lh_pc_q7 = np.zeros(n_x+1)
Lh_pc_q6 = np.zeros(n_x+1)
Lh_pc_q5 = np.zeros(n_x+1)
Lh_pc_q4 = np.zeros(n_x+1)
Lh_pc_q3 = np.zeros(n_x+1)
Lh_pc_q2 = np.zeros(n_x+1)
Lh_pc_q1 = np.zeros(n_x+1)

for j in range(n_x+1):
    for i in range(test_size):
        Lh_pc_q10[j]+= (y_test[i,j]-y_pred_pc_q10[i,j])**2
        Lh_pc_q9[j]+= (y_test[i,j]-y_pred_pc_q9[i,j])**2
        Lh_pc_q8[j]+= (y_test[i,j]-y_pred_pc_q8[i,j])**2
        Lh_pc_q7[j]+= (y_test[i,j]-y_pred_pc_q7[i,j])**2
        Lh_pc_q6[j]+= (y_test[i,j]-y_pred_pc_q6[i,j])**2
        Lh_pc_q5[j]+= (y_test[i,j]-y_pred_pc_q5[i,j])**2
        Lh_pc_q4[j]+= (y_test[i,j]-y_pred_pc_q4[i,j])**2
        Lh_pc_q3[j]+= (y_test[i,j]-y_pred_pc_q3[i,j])**2
        Lh_pc_q2[j]+= (y_test[i,j]-y_pred_pc_q2[i,j])**2
        Lh_pc_q1[j]+= (y_test[i,j]-y_pred_pc_q1[i,j])**2
        
Lh_pc_q10 = Lh_pc_q10/test_size
Lh_pc_q9 = Lh_pc_q9/test_size
Lh_pc_q8 = Lh_pc_q8/test_size
Lh_pc_q7 = Lh_pc_q7/test_size
Lh_pc_q6 = Lh_pc_q6/test_size        
Lh_pc_q5 = Lh_pc_q5/test_size
Lh_pc_q4 = Lh_pc_q4/test_size
Lh_pc_q3 = Lh_pc_q3/test_size
Lh_pc_q2 = Lh_pc_q2/test_size
Lh_pc_q1 = Lh_pc_q1/test_size

#plot error 
#Samling error Quad 
plt.plot(curv_abs,Lh_pc_q5, '--',label='Nquad=10')
plt.plot(curv_abs,Lh_pc_q4, '--',label='Nquad=9')
plt.plot(curv_abs,Lh_pc_q3, '--',label='Nquad=8')
plt.plot(curv_abs,Lh_pc_q2, '--',label='Nquad=7')
plt.plot(curv_abs,Lh_pc_q1, '--',label='Nquad=6')
plt.xlabel('curvilinear abscissa (km)')
plt.ylabel('Water metrics LH')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.show()
#Truncation error Quad
plt.plot(curv_abs,Lh_pc_q10, '--',label='P=10')
plt.plot(curv_abs,Lh_pc_q9, '--',label='P=9')
plt.plot(curv_abs,Lh_pc_q8, '--',label='P=8')
plt.plot(curv_abs,Lh_pc_q7, '--',label='P=7')
plt.plot(curv_abs,Lh_pc_q6, '--',label='P=6')
plt.xlabel('curvilinear abscissa (km)')
plt.ylabel('Water metrics LH')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.show()

### LS
#Changing degree for truncation error
pc_predictor_ls10 = SurrogateModel('pc', corners, init_size, plabels, strategy='LS', degree=10, distributions=dists_ot, N_quad = 100)
pc_predictor_ls9 = SurrogateModel('pc', corners, init_size, plabels, strategy='LS', degree=9, distributions=dists_ot, N_quad = 100)
pc_predictor_ls8 = SurrogateModel('pc', corners, init_size, plabels, strategy='LS', degree=8, distributions=dists_ot, N_quad = 100)
pc_predictor_ls7 = SurrogateModel('pc', corners, init_size, plabels, strategy='LS', degree=7, distributions=dists_ot, N_quad = 100)
pc_predictor_ls6 = SurrogateModel('pc', corners, init_size, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)

#Changing sampling size for samling error
pc_predictor_ls5 = SurrogateModel('pc', corners, init_size5, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)
pc_predictor_ls4 = SurrogateModel('pc', corners, init_size4, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)
pc_predictor_ls3 = SurrogateModel('pc', corners, init_size3, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)
pc_predictor_ls2 = SurrogateModel('pc', corners, init_size2, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)
pc_predictor_ls1 = SurrogateModel('pc', corners, init_size1, plabels, strategy='LS', degree=6, distributions=dists_ot, N_quad = 100)


pc_predictor_ls10.fit(x_train, y_train)
pc_predictor_ls9.fit(x_train, y_train)
pc_predictor_ls8.fit(x_train, y_train)
pc_predictor_ls7.fit(x_train, y_train)
pc_predictor_ls6.fit(x_train, y_train)
pc_predictor_ls5.fit(x_train5, y_train5)
pc_predictor_ls4.fit(x_train4, y_train4)
pc_predictor_ls3.fit(x_train3, y_train3)
pc_predictor_ls2.fit(x_train2, y_train2)
pc_predictor_ls1.fit(x_train1, y_train1)

y_pred_pc_ls10, _ = pc_predictor_ls10(x_test)
y_pred_pc_ls9, _ = pc_predictor_ls9(x_test)
y_pred_pc_ls8, _ = pc_predictor_ls8(x_test)
y_pred_pc_ls7, _ = pc_predictor_ls7(x_test)
y_pred_pc_ls6, _ = pc_predictor_ls6(x_test)
y_pred_pc_ls5, _ = pc_predictor_ls5(x_test)
y_pred_pc_ls4, _ = pc_predictor_ls4(x_test)
y_pred_pc_ls3, _ = pc_predictor_ls3(x_test)
y_pred_pc_ls2, _ = pc_predictor_ls2(x_test)
y_pred_pc_ls1, _ = pc_predictor_ls1(x_test)

Lh_pc_ls10 = np.zeros(n_x+1)
Lh_pc_ls9 = np.zeros(n_x+1)
Lh_pc_ls8 = np.zeros(n_x+1)
Lh_pc_ls7 = np.zeros(n_x+1)
Lh_pc_ls6 = np.zeros(n_x+1)
Lh_pc_ls5 = np.zeros(n_x+1)
Lh_pc_ls4 = np.zeros(n_x+1)
Lh_pc_ls3 = np.zeros(n_x+1)
Lh_pc_ls2 = np.zeros(n_x+1)
Lh_pc_ls1 = np.zeros(n_x+1)

for j in range(n_x+1):
    for i in range(test_size):
        Lh_pc_ls10[j]+= (y_test[i,j]-y_pred_pc_ls10[i,j])**2
        Lh_pc_ls9[j]+= (y_test[i,j]-y_pred_pc_ls9[i,j])**2
        Lh_pc_ls8[j]+= (y_test[i,j]-y_pred_pc_ls8[i,j])**2
        Lh_pc_ls7[j]+= (y_test[i,j]-y_pred_pc_ls7[i,j])**2
        Lh_pc_ls6[j]+= (y_test[i,j]-y_pred_pc_ls6[i,j])**2
        Lh_pc_ls5[j]+= (y_test[i,j]-y_pred_pc_ls5[i,j])**2
        Lh_pc_ls4[j]+= (y_test[i,j]-y_pred_pc_ls4[i,j])**2
        Lh_pc_ls3[j]+= (y_test[i,j]-y_pred_pc_ls3[i,j])**2
        Lh_pc_ls2[j]+= (y_test[i,j]-y_pred_pc_ls2[i,j])**2
        Lh_pc_ls1[j]+= (y_test[i,j]-y_pred_pc_ls1[i,j])**2

Lh_pc_ls10 = Lh_pc_ls10/test_size
Lh_pc_ls9 = Lh_pc_ls9/test_size
Lh_pc_ls8 = Lh_pc_ls8/test_size
Lh_pc_ls7 = Lh_pc_ls7/test_size
Lh_pc_ls6 = Lh_pc_ls6/test_size        
Lh_pc_ls5 = Lh_pc_ls5/test_size
Lh_pc_ls4 = Lh_pc_ls4/test_size
Lh_pc_ls3 = Lh_pc_ls3/test_size
Lh_pc_ls2 = Lh_pc_ls2/test_size
Lh_pc_ls1 = Lh_pc_ls1/test_size

#plot error 

#Samling error LS
plt.plot(curv_abs,Lh_pc_ls5,'--',label='N=100')
plt.plot(curv_abs,Lh_pc_ls4,'--',label='N=90')
plt.plot(curv_abs,Lh_pc_ls3,'--',label='N=80')
plt.plot(curv_abs,Lh_pc_ls2,'--',label='N=70')
plt.plot(curv_abs,Lh_pc_ls1,'--',label='N=60')
plt.xlabel('curvilinear abscissa')
plt.ylabel('Water metrics')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.show()

#Truncation error LS
plt.plot(curv_abs,Lh_pc_ls10,'--',label='P=10')
plt.plot(curv_abs,Lh_pc_ls9,'--',label='P=9')
plt.plot(curv_abs,Lh_pc_ls8,'--',label='P=8')
plt.plot(curv_abs,Lh_pc_ls7,'--',label='P=7')
plt.plot(curv_abs,Lh_pc_ls6,'--',label='P=6')
plt.xlabel('curvilinear abscissa')
plt.ylabel('Water metrics')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.show()

#LC metrics 

pc_predictor_r = SurrogateModel('pc', corners, 1000, plabels, strategy='LS', degree=10, distributions=dists_ot, N_quad = 100)
pc_predictor_q = SurrogateModel('pc', corners, 121, plabels, strategy='Quad', degree=10, distributions=dists_ot, N_quad = 121)
pc_predictor_ls1 = SurrogateModel('pc', corners, init_size5, plabels, strategy='LS', degree=10, distributions=dists_ot, N_quad = 100)
pc_predictor_ls2 = SurrogateModel('pc', corners, init_size3, plabels, strategy='LS', degree=10, distributions=dists_ot, N_quad = 100)

x_quadq = pc_predictor_q.predictor.sample
y_quadq = fl(x_quadq)

pc_predictor_r.fit(x_trainr,y_trainr)
pc_predictor_q.fit(x_quadq,y_quadq)
pc_predictor_ls1.fit(x_train5,y_train5)
pc_predictor_ls2.fit(x_train3,y_train3)

surror = pc_predictor_r.predictor.pc_result
surroq = pc_predictor_q.predictor.pc_result
surrols1 = pc_predictor_ls1.predictor.pc_result
surrols2 = pc_predictor_ls2.predictor.pc_result

Surror = ot.FunctionalChaosResult(surror)
Coeffsr = Surror.getCoefficients()
Surroq = ot.FunctionalChaosResult(surroq)
Coeffsq = Surroq.getCoefficients()
Surrols1 = ot.FunctionalChaosResult(surrols1)
Coeffls1 = Surrols1.getCoefficients()
Surrols2 = ot.FunctionalChaosResult(surrols2)
Coeffls2 = Surrols2.getCoefficients()

#print(Coeffsq)
#print(type(Coeffsq))


g_ref = np.array(Coeffsr)
(n_pc,k) = np.shape(g_ref)
gq = np.array(Coeffsq)
gls1 = np.array(Coeffls1)
gls2 = np.array(Coeffls2)
LC_pc_q = np.zeros(n_pc)
LC_pc_ls1 = np.zeros(n_pc)
LC_pc_ls2 = np.zeros(n_pc)

print(g_ref)
print(np.shape(g_ref))
print(np.shape(gq))
print(np.shape(gls1))
print(np.shape(gls2))


for j in range(n_pc):
    for i in range(k):
        LC_pc_q[j]+= abs(g_ref[j,i]-gq[j,i])
        LC_pc_ls1[j]+= abs(g_ref[j,i]-gls1[j,i])
        LC_pc_ls2[j]+= abs(g_ref[j,i]-gls2[j,i])
                
        
#plot error 
plt.plot(LC_pc_q, '--',label='Quad, N=121')
plt.plot(LC_pc_ls1, '--',label='LS, N=100')
plt.plot(LC_pc_ls2, '--',label='LS, N=50')
plt.xlabel('spectrum')
plt.ylabel('Water metrics LC(i)')
plt.title('Water metrics along the Garonne river')
plt.legend()
plt.show()
