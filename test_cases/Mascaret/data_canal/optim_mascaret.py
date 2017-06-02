# -*- coding: utf-8 -*-
from ctypes import *
import sys
import numpy as np
from scipy.optimize import minimize

"""
MASCARET API with Python

Ex. : Cross-Sections

Author(s) : Fabrice Zaoui

Copyright EDF 2017
"""

# delete MASCARET instance
def deleteMasc(my_id_masc):
    error = mydll.C_DELETE_MASCARET(my_id_masc)
    if error != 0:
        print 'Error while deleting the instantiation #%d' % my_id_masc.value
        exit(0)
    else:
        print 'Deleting MASCARET instantiation #%d...OK' % my_id_masc.value
        return

# load MASCARET dynamic library
mydll = CDLL('./mascaret.so')
if not mydll:
    print 'Error while loading the MASCARET library'
    exit(0)

# error flag
error = c_int()

# print option
iprint = 0

# create a MASCARET model
my_id_masc = c_int()
error = mydll.C_CREATE_MASCARET(byref(my_id_masc))
if error != 0:
    print 'Error while creating a MASCARET model'
    exit(0)
else:
    print 'Instantiation of a MASCARET model #%d...OK' % my_id_masc.value

# import data
files_name = ['mascaret0.xcas','mascaret0.geo','mascaret0_0.loi',
              'mascaret0_1.loi','mascaret0.lis','mascaret0.opt']
files_type = ['xcas','geo','loi','loi','lis','res']             
L_file = len(files_name)
files_name_c = (c_char_p * L_file)(*files_name)
files_type_c = (c_char_p * L_file)(*files_type)
error = mydll.C_IMPORT_MODELE_MASCARET(my_id_masc,files_name_c,files_type_c,
                                       L_file,iprint)
if error != 0:
    print 'Error while reading the MASCARET data files'
    deleteMasc(my_id_masc)
    exit(0)
else:
    print 'Reading MASCARET data files...OK'

# get the number of nodes
var_name = c_char_p('Model.X')
nb_nodes = c_int()
itemp1 = c_int()
itemp2 = c_int()
error = mydll.C_GET_TAILLE_VAR_MASCARET(my_id_masc,var_name,0,byref(nb_nodes),
                                        byref(itemp1),byref(itemp2))
if error != 0:
    print 'Error while reading the number of 1D nodes'
    deleteMasc(my_id_masc)
    exit(0)
else:
    print '1D nodes = %d...OK' % nb_nodes.value

# state initialisation
Q = [0.]*nb_nodes.value
Z = [10.]*nb_nodes.value
Q_c = (c_double * nb_nodes.value)(*Q)
Z_c = (c_double * nb_nodes.value)(*Z)
error = mydll.C_INIT_LIGNE_MASCARET(my_id_masc,byref(Q_c),byref(Z_c),nb_nodes)
if error != 0:
    print 'Error while initialising the state of MASCARET'
    deleteMasc(my_id_masc)
    exit(0)
else:
    print 'State initialisation successfull...OK'

# get the parameters on the simulation time
## time step
dt = c_double()
var_name = c_char_p('Model.DT')
error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name,0,0,0,byref(dt))
if error != 0:
    print 'Error while getting the value of the time step'
    deleteMasc(my_id_masc)
    exit(0)
## initial time
t0 = c_double()
var_name = c_char_p('Model.InitTime')
error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name,0,0,0,byref(t0))
if error != 0:
    print 'Error while getting the value of the initial time'
    deleteMasc(my_id_masc)
    exit(0)
## final time
tend = c_double()
var_name = c_char_p('Model.MaxCompTime')
error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name,0,0,0,byref(tend))
if error != 0:
    print 'Error while getting the value of the final time'
    deleteMasc(my_id_masc)
    exit(0)
print 'Getting the simulation times...OK'

# simulation MASCARET
def simulMASC(x):
    # modification des points bas 4 et 5 du premier profil
    newval = c_double(x)
    var_name = c_char_p('Model.CrossSection.Y')
    error = mydll.C_SET_DOUBLE_MASCARET(my_id_masc,var_name,1,4,0,newval)
    error = mydll.C_SET_DOUBLE_MASCARET(my_id_masc,var_name,1,5,0,newval)
    # appel du simulateur
    error = mydll.C_CALCUL_MASCARET(my_id_masc,t0,tend,dt,iprint)
    # calcul de la fonction cout : ecart de la hauteur d'eau a sa moyenne
    hm = 0.
    ecart = 0.
    z_c = c_double()
    zbot_c = c_double()
    var_name1 = c_char_p('State.Z')
    var_name2 = c_char_p('Model.Zbot')
    #... 1ere passe pour le calcul de la moyenne
    for i in range(nb_nodes.value):
        error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name1,i+1,0,0,byref(z_c))
        error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name2,i+1,0,0,byref(zbot_c))
        hm = hm + (z_c.value - zbot_c.value)
    hm = hm / nb_nodes.value
    #... 2eme passe pour l'ecart quadratique
    for i in range(nb_nodes.value):
        error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name1,i+1,0,0,byref(z_c))
        error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name2,i+1,0,0,byref(zbot_c))
        ecart = ecart + (z_c.value - zbot_c.value - hm)**2
    return ecart

# appel a l'optimiseur pour retrouve une pente qui correspond a une hauteur d'eau normale
# (on modifie la pentedu canal  en jouant sur les points bas du premier profil)
#... les bornes de la variable: cote du fond du premier profil
nvar = 1
vbounds = np.zeros((nvar, 2))
vbounds[0, 0] = 7.5
vbounds[0, 1] = 15.
#... initial guess
v0 = np.array([8.])
#... appel a 'minimize'
res = minimize(simulMASC, v0, \
                bounds=vbounds, method='L-BFGS-B', \
                jac=False, \
                options={'maxiter':100, 'maxfun':10000, \
                'disp':True})
#... verification que la hauteur d'eau est egale a 5 metres en tout point (hauteur normale)
z_c = c_double()
zbot_c = c_double()
var_name1 = c_char_p('State.Z')
var_name2 = c_char_p('Model.Zbot')
for i in range(nb_nodes.value):
    error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name1,i+1,0,0,byref(z_c))
    error = mydll.C_GET_DOUBLE_MASCARET(my_id_masc,var_name2,i+1,0,0,byref(zbot_c))
    print z_c.value - zbot_c.value
















