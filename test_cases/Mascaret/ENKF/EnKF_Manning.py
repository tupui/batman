### -----------------------------------------------------
### Melanie Rochoux, Nabil El Mocayd , Sophie Ricci CERFACS
### DA over Manning equation with OpenTURNS (python environment)
### 1D case (stochastic space made of 1 random variable)
### April 2016
### -----------------------------------------------------
print ('\n>> DATA ASSIMILATION ON MANNING EQUATION IN PYTHON ENVIRONMENT')
print ('     * example of Manning equation')
print ('     * Monte Carlo method')


### PACKAGE IMPORTATION #################################
from openturns import *
import math
from numpy import *
import pylab as pl 


### FORWARD MODEL #######################################
def Manning(x):
    ### identification of random variables 
    Ks = x[0]   # strickler rugosity coefficient
    ### model input parameters
    Q = 1000    # discharge
    I = 0.001   # slope
    L = 100     # channel width
    ### manning equation
    y = (Q/(Ks*L*sqrt(I)))
    y = power(y,3./5.)
    return [y]
model = PythonFunction(1,1,Manning)


### ENSEMBLE ############################################
print ('\nENSEMBLE')

### ensemble parameters
indim = 1       # number of random variables -inputs
outdim = 1      # number of random variables -outputs
nsample = 100   # size of the Monte Carlo experiment 

### input distribution definition
##### parameter range
mean = 35.
std = 5.
##### distribution (assumption: Gaussian truncated)
distrib = Normal(mean,std)

### samples
##### input sample
inputsample = distrib.getSample(nsample)
##### output sample
outputsample = model(inputsample)
##### statistics
meanKs = inputsample.computeMean()
print ('-mean Ks: ', meanKs)
meanH = outputsample.computeMean()
print ('-mean h: ', meanH)

### errors
errKs = inputsample-meanKs
errH = outputsample-meanH
print ('-error in Ks: ', errKs)
print ('-error in h: ', errH)

### conversion into array (numpy)
H = array(outputsample)
Ks = array(inputsample)


### OBSERVATION #######################################
print ('\nOBSERVATIONS')

### true/reference values for the input parameter Ks
Ks_true = NumericalSample([35, 37, 38, 32, 30],1)

### synthetic observation generation
H_obs = model(Ks_true) 
print ('-observed h: ', H_obs)

### observation errors
R = [1., 1., 1., 1., 1.]
#R = [0.01, 0.01, 0.01, 0.01, 0.01]
print ('-observed errors (R): ', R)


### KALMAN FILTER UPDATE ##############################
print ('\nENSEMBLE KALMAN FILTER ANALYSIS')
Ks_ana = empty(5)
for i in range(len(R)):
   print ('-----------------------------')
   print ('Data assimilation cycle', i, len(R))
 
   ### error covariance matrices
   BHT = dot(transpose(errKs),errH)
   print ('-error cross-covariance matrix BHT: ', BHT)
   HBHT = dot(transpose(errH),errH)
   print ('-error covariance matrix HBHT: ', HBHT)

   ### Kalman gain matrix
   HBHTpR = HBHT + R[i]
   print ('-total error covariance matrix HBHT+R: ', HBHTpR)
   invHBHTpR = linalg.inv(HBHTpR)
   K = dot(BHT,invHBHTpR)

   ### innovation vector
   Ks_bck = NumericalSample([mean],1)
   Hbck = model(Ks_bck)   
   d = H_obs[i]-Hbck[0]

   ### correction term 
   k = K[0][0]
   D = d[0]
   Kd = D*k
   print ('-innovation (d): ', d)
   print ('-Kalman gain (K): ', K)
   print ('-correction term (Kd): ', Kd)

   ### analysis
   Ks_ana[i] = mean + Kd
   print ('-bck/true/ana parameters: ', Ks_bck[0], Ks_true[i], Ks_ana[i])

### simulation of water level associated with Kalman analysis
Ks_ana = NumericalSample(Ks_ana,1)
H_ana = model(Ks_ana)

### figures
##### parameter space
pl.figure()
T = [1,2,3,4,5]
pl.plot(T,Ks_true,'ok')
pl.plot(T,Ks_ana,'-r')
pl.ylabel('Ks')
pl.xlabel('Assim Cycle')
pl.title('EnKF - Manning equation')
pl.savefig('./EnKF_Manning_Ks.png')

##### observation space
pl.figure()
pl.plot(T,H_obs,'ok')
pl.plot(T,H_ana,'-r')
pl.ylabel('Water level')
pl.xlabel('Assim Cycle')
pl.title('EnKF - Manning equation')
pl.savefig('./EnKF_Manning_h.png')

