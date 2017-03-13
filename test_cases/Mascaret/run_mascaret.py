# coding: utf-8
"""
A command line interface to MASCARET V7P2
=========================================

MASCARET C API with Python
Mascaret V7P2 Simulation with API from Python

"""
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
from .mascaret_api import MascaretApi

def run_mascaret():
    """Command line interface."""

    masc = MascaretApi()

    # Create Mascaret model
    id_masc = masc.create()
    if id_masc < 0:
        print('Erreur de creation de modele!')
        errmess = masc.error_message(id_masc)
        print(errmess)
    else:
        print('Create model OK', id_masc)

    # Import data
    myfile = 'run_mascaret.txt'
    np_files_type, np_files_name = masc.Read_FichiersTxt(myfile)
    files_type = np_files_type.tolist()
    files_name = np_files_name.tolist()

    print('files_type=', type(files_type))
    print('files_name=', type(files_name))
    print('files_type=', files_type)
    print('files_name=', files_name)

    erreur = masc.import_model(id_masc, files_name, files_type)
    if erreur != 0:
        print('Erreur importation!')
        errmess = masc.error_message(id_masc)
        print(errmess)
    else:
        print('Import Data OK')

    # Read Parameter file
    myfile = 'run_mascaret_Param.txt'
    np_names, np_values = masc.Read_ParamTxt(myfile)
    print 'np_names=', np_names
    print 'np_values=',np_values
    print('Read Parameter file OK')

    flag_init_lig = int(np_values[0])
    flag_printBC = int(np_values[1])
    flag_changeQ = int(np_values[2])
    flag_getIndZonesFrot = int(np_values[3])
    flag_changeKS = int(np_values[4])
    flag_MC_KS = int(np_values[5])
    print('flag_init_lig=', flag_init_lig, 'flag_printBC=', flag_printBC, 'flag_changeQ=', flag_changeQ, 'flag_changeKS=', flag_changeKS, 'flag_MC_KS=', flag_MC_KS)
    Qcst = np_values[6].tolist()
    Zcst = np_values[7].tolist()
    Num_New_Q_BC = int(np_values[8])
    Value_New_Q_BC = np_values[9]
    ind_KS = c_int(int(np_values[10]))
    new_CF1value = float(np_values[11])
    ind_ZoneKS = int(np_values[12])
    new_ZoneKSvalue = float(np_values[13])
    index_outstate = int(np_values[14])
    mu = float(np_values[15])
    sigma = float(np_values[16])
    nb_simul = int(np_values[17])

    # Get the number of nodes
    var_name = c_char_p('Model.X')
    size_X = c_int()
    erreur, size_X = masc.get_sizevar(id_masc, var_name)
    if erreur != 0:
        print('Erreur getting size')
        errmess = masc.error_message(id_masc)
        print(errmess)
    else:
        print('Get nb of nodes OK, size_X=', size_X.value)

    if flag_init_lig == 1:
        # Initialize Mascaret with .lig file
        lig_file = ['LigneEauInitiale.lig']
        erreur = masc.init_mascaret_file(id_masc, lig_file)
        if erreur != 0:
            print('Erreur Initializing mascaret')
            errmess = masc.error_message(id_masc)
            print(errmess)
        else:
            print('Initializing mascaret with lig file OK')
    else:
        # Initialize Mascaret with constante value
        #   Qcst = [0.]
        #   Zcst = [10.]
        erreur = masc.init_mascaret_constant(id_masc, size_X, Qcst, Zcst)
        if erreur != 0:
            print('Erreur Initializing mascaret')
            errmess = masc.error_message(id_masc)
            print(errmess)
        else:
            print('Initializing mascaret with cst value OK')

    # Get Simulations times
    erreur, dt, tini, tend = masc.get_mascaret_times(id_masc)
    if erreur != 0:
        print('Error getting simulation times')
        errmess = masc.error_message(id_masc)
        print(errmess)
    else:
        print('Get simulation times')

    # Get infos on boundary conditions
    nb_BC = c_int()
    erreur, nb_BC, l_Name_BC, l_Num_BC = masc.get_info_allBC(id_masc)
    print('Get Boundary conditions OK =', nb_BC.value, l_Name_BC, l_Num_BC)
    # Adour Maritime
    # Q9312510.loi   1 Cambo
    # Q3120030.loi   2 Dax
    # Q5421020.loi   3 Orthez
    # Q7412910.loi   4 Escos
    # Q935001001.loi 5 Convergent
    # print 'Loi Q Cambo', Tab_Q_BC[0,:]
    # print 'Loi Q Dax', Tab_Q_BC[1,:]
    # print 'Loi Q Orthez', Tab_Q_BC[2,:]
    # print 'Loi Q Escos', Tab_Q_BC[3,:]
    # Loi Convergent est decrite en Cote

    # Get BC Q(t)
    erreur, Tab_Q_BC = masc.get_BC_Qt(id_masc)
    print('Get BC Q(t) OK ')
    if flag_printBC == 1:
        for k in range(nb_BC.value):
            print('Loi Q', l_Name_BC[k], l_Num_BC[k], Tab_Q_BC[k, :])

    if flag_changeQ == 1:
        # Prescribe BC Q(t)
        New_Tab_Q_BC = Tab_Q_BC
    # Change Q(t) at Cambo to Q(t) = 300.
        New_Tab_Q_BC[Num_New_Q_BC, :] = Value_New_Q_BC
        erreur = masc.set_BC_Qt(id_masc, New_Tab_Q_BC)
        print('Change Q OK')

    if flag_getIndZonesFrot == 1:
       erreur, l_Ind_DebZonesFrot, l_Ind_EndZonesFrot = masc.get_indzonefrot(id_masc)
       print('l_Ind_DebZonesFrot',l_Ind_DebZonesFrot)
       print('l_Ind_EndZonesFrot', l_Ind_EndZonesFrot)
    
    if flag_changeKS == 1:
    # Change minor friction coef over given zone
       erreur = masc.change_zonefriction_minor(id_masc,ind_ZoneKS, new_ZoneKSvalue)
       print('Change Zone KS OK')
    
    if flag_changeKS == 2:
    # Change minor friction coef at given index
    # Change minor friction coef at Peyrehorade (index ind = 1754, Ks = 52.5)
       erreur = masc.change_friction_minor(id_masc,ind_KS, new_CF1value)
       print('Change KS 2 OK')

    # Run Mascaret
    erreur = masc.run_mascaret(id_masc, dt, tini, tend)
    if erreur != 0:
        print('Error Running Mascaret')
        errmess = masc.error_message(id_masc)
        print(errmess)
    else:
        print('Running Mascaret OK')

    # For Adour Maritime test case
    #index_Lesseps = 10
    #error, Z_res_c = masc.get_state(id_masc,index_Lesseps)
    # print 'Z_res_c.value Lesseps=',Z_res_c.value
    #index_Urt = 296
    #error, Z_res_c = masc.get_state(id_masc,index_Urt)
    # print 'Z_res_c.value Urt=',Z_res_c.value
    #index_Pey = 1754
    #error, Z_res_c = masc.get_state(id_masc,index_Pey)
    # print 'Z_res_c.value Peyrehorade=',Z_res_c.value

    error, Z_res_c = masc.get_state(id_masc, index_outstate)
    print('Z_res_c.value =', Z_res_c.value)

    if flag_MC_KS == 1:
        print('Running MC w.r.t. KS')
    # MC Run Mascaret with aleatory CF1 varying
    # Set the friction coefficient as a normal distribution
        CF = np.random.normal(mu, sigma, nb_simul)
    # Graph CF
        count, bins, ignored = plt.hist(CF, 20, normed=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu)**2 / (2 * sigma**2)),
                 linewidth=2, color='r')
        plt.title("Input Histogram (Friction coefficient)")
        plt.xlabel("Value (m3/s)")
        plt.ylabel("Frequency")
        plt.show()
    # Uncertainty propagation
        Z_res = [0.] * nb_simul
        for k in range(nb_simul):
            # Change KS value
            new_CF1value = CF[k]
            print('new_CF1value =', new_CF1value)
            erreur = masc.change_friction_minor(id_masc, ind_KS, new_CF1value)
    # Run MASCARET
            erreur = masc.run_mascaret(id_masc, dt, tini, tend)
    # Get State Z at index_outstate at the end of the MASCARET Run
            error, Z_res_c = masc.get_state(id_masc, index_outstate)
            Z_res[k] = Z_res_c.value
            print('Z_res_c.value perturbed =', Z_res_c.value)
    # plot the distribution on the water level
        plt.hist(Z_res)
        plt.title("Output Histogram (Water Level)")
        plt.xlabel("Value (m)")
        plt.ylabel("Frequency")
        plt.show()
        print('Running MC w.r.t. KS OK')

if __name__ == '__main__':
    run_mascaret()
