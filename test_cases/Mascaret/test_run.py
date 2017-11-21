import ctypes
import csv
import logging
import numpy as np
from batman.functions import MascaretApi
from batman.functions.telemac_mascaret import print_statistics, histogram, plot_opt

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

# Create an instance of MascaretApi
study = MascaretApi('config_canal.json','config_canal_user.json')
#study = MascaretApi('config_garonne_lnhe.json','config_garonne_lnhe_user.json')
#study = MascaretApi('config_garonne_lnhe_casier.json','config_garonne_lnhe_user_casier.json')

# Print informations concerning this study
print(study)

# Run study with Ks and Q specified constant values 
#res = study(x=[30, 3000])
#print('Water level computed with Ks = 30, Q = 3000',res)
#print('Water level computed with Ks = 30, Q = 3000',res[1])
# Plot the water level along the open-channel at final time
#plot_opt('ResultatsOpthyca.opt')

# Run study  with the user defined tasks and values 
res = study()
print('Water level computed with json user defined values', res)
#print('Water level computed with json user defined values', res[1])
# Plot the water level along the open-channel at final time
plot_opt('ResultatsOpthyca.opt')

# Run study  with user provided Boundary Conditions matrix in .csv 
# User defined BC matrix (here 10 time steps and 2 BC - Q and Z)
#nb_timebc = 10
#nb_bc = 2
#vect_in_timebc = np.zeros(nb_timebc)
#mat_in_BC = np.zeros((nb_timebc, nb_bc))
#with open('My_BC.csv', newline = '' ) as csvfile:
#    myreader = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
#    i = 0
#    for row in myreader:
#        print (row[0], row[1], row[2])
#        vect_in_timebc[i] = row[0]
#        mat_in_BC[i,0] = row[1]
#        mat_in_BC[i,1] = row[2]
#        i = i+1
#print (vect_in_timebc)
#print (mat_in_BC)
#tab_timebc_c =(ctypes.c_double*nb_timebc)()
#for j in range(nb_timebc):
#    tab_timebc_c[j] = vect_in_timebc[j]
#tab_CL1_c = (ctypes.POINTER(ctypes.c_double)*nb_bc)()
#tab_CL2_c = (ctypes.POINTER(ctypes.c_double)*nb_bc)()
#for i in range(nb_bc):
#    tab_CL1_c[i] = (ctypes.c_double*nb_timebc)()
#    tab_CL2_c[i] = (ctypes.c_double*nb_timebc)()
#    for j in range(nb_timebc):
#        tab_CL1_c[i][j] = mat_in_BC[j][i]
#        tab_CL2_c[i][j] = 0.
#h = study(Qtime = [nb_timebc, tab_timebc_c, tab_CL1_c, tab_CL2_c])
#print('Water level computed with user defined BC matrix', h)
# Plot the water level along the open-channel at final time
#plot_opt('ResultatsOpthyca.opt')



# Print and plot statistics concerning the model output uncertainty when MC
#print_statistics(h)
#histogram(h, xlab='Water level at Marmande', title='Distribution of the uncertainty')

# Details about MascaretApi
#help(MascaretApi)
