# coding: utf-8
"""
MascaretAPI class
=================

Python Wrapper for Mascaret - Assim? - DAMP? - UQ? - ...

"""

import sys
import os
from ctypes import *
import numpy as np


class MascaretApi(object):

    """Mascaret API."""

    def __init__(self):
        """Loading the Mascaret library."""
        path = os.path.dirname(os.path.realpath(__file__))
        libmascaret = path + '/lib'
        print(libmascaret)
        if sys.platform.startswith('linux'):
            try:
                self.libmascaret = CDLL(libmascaret + '/mascaret.so')
            except Exception as e:
                raise Exception('Unable to load : \
        mascaret.so. Check the environment variable LIBMASCARET.', e)
        else:
            raise Exception(u'Unsupported OS')

    def Read_FichiersTxt(self, myfile):
        """Read parameter file."""
        file = open(myfile, "r")
        np_types, np_noms = np.loadtxt(
            myfile, dtype=str, delimiter=' ', usecols=(0, 1), unpack=True)
        file.close()
        return np_types, np_noms

    def Read_ParamTxt(self, myfile):
        """Read parameter file."""
        file = open(myfile, "r")
        np_noms, np_values = np.loadtxt(
            myfile, dtype=str, delimiter=' ', usecols=(0, 1), unpack=True)
        file.close()
        return np_noms, np_values

    def create(self):
        """wrapper 1 : create a model."""
        id_masc = c_int()
        error = self.libmascaret.C_CREATE_MASCARET(byref(id_masc))
        if error != 0:
            print('Error while creating a MASCARET model')
            return -1
        else:
            return id_masc.value

    def import_model(self, id_masc, file_name, file_type):
        """wrapper 2 : import a model."""
        if (type(file_name) != list) | (type(file_type) != list):
            print('Arguments are not lists of files')
            return -1
        # .opt and .lis are writen only if iprint = 1 at the import AND the calcul steps
        iprint = 1
        idmasc = c_int(id_masc)
        L_file = len(file_name)
        file_name_c = (c_char_p * L_file)(*file_name)
        file_type_c = (c_char_p * L_file)(*file_type)
        error = self.libmascaret.C_IMPORT_MODELE_MASCARET(idmasc, file_name_c,
                                                    file_type_c, L_file, iprint)
        if error != 0:
            print('Error while importing a MASCARET model')
            return -1
        return error

    def delete(self, id_masc):
        """wrapper 3 : delete a model."""
        error = self.libmascaret.C_DELETE_MASCARET(id_masc)
        if error != 0:
            print('Error while deleting the instantiation #%d' % id_masc)
            return -1
        else:
            return 0

    def get_sizevar(self, id_masc, var_name):
        """wrapper 4 : get a size of Mascaret var."""
        idmasc = c_int(id_masc)
        nb_nodes = c_int()
        il_temp1 = c_int()
        il_temp2 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(idmasc, var_name, 0, byref(nb_nodes),
                                                     byref(il_temp1), byref(il_temp2))
        if error != 0:
            print('Error while getting size var in model  #%d' % id_masc)
            return -1
        else:
            return error, nb_nodes

    def init_mascaret_constant(self, id_masc, nb_nodes, rl_Q, rl_Z):
        """wrapper 5 : Initialize Mascaret Model from values."""
        idmasc = c_int(id_masc)
        Q = rl_Q * nb_nodes.value
        Z = rl_Z * nb_nodes.value
        Q_c = (c_double * nb_nodes.value)(*Q)
        Z_c = (c_double * nb_nodes.value)(*Z)
        error = self.libmascaret.C_INIT_LIGNE_MASCARET(
            idmasc, byref(Q_c), byref(Z_c), nb_nodes)
        if error != 0:
            print('Error while initialising the state of MASCARET')
        else:
            print('State constant initialisation successfull from constant value...OK')
        return error

    def init_mascaret_file(self, id_masc, init_file_name):
        """wrapper 6 : Initialize Mascaret Model from file."""
        idmasc = c_int(id_masc)
        iprint = 1
        idmasc = c_int(id_masc)
        init_file_name_c = (c_char_p)(*init_file_name)
        error = self.libmascaret.C_INIT_ETAT_MASCARET(idmasc, init_file_name_c, iprint)
        if error != 0:
            print('Error while initialising the state of Mascaret from .lig')
        else:
            print('State constant initialisation successfull from lig...OK')
        return error

    def get_mascaret_times(self, id_masc):
        """wrapper 7 : get the simulation times."""
        idmasc = c_int(id_masc)
        # dt
        dt = c_double()
        var_name = c_char_p('Model.DT')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, 0, 0, 0, byref(dt))
        if error != 0:
            print('Error while getting the value of the time step')
        else:
            print('dt=', dt.value)
        # t0
        t0 = c_double()
        var_name = c_char_p('Model.InitTime')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, 0, 0, 0, byref(t0))
        if error != 0:
            print('Error while getting the value of the initial time')
        else:
            print('t0=', t0.value)
        # tend
        tend = c_double()
        var_name = c_char_p('Model.MaxCompTime')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, 0, 0, 0, byref(tend))
        if error != 0:
            print('Error while getting the value of the final time')
        else:
            print('tend=', tend.value)
        return error, dt, t0, tend

    def run_mascaret(self, id_masc, dt, t0, tend):
        """wrapper 8 : run Mascaret simulatic simulation."""
        idmasc = c_int(id_masc)
        # .opt and .lis are writen only if iprint = 1 at the import AND the calcul steps
        iprint = 1
        error = self.libmascaret.C_CALCUL_MASCARET(idmasc, t0, tend, dt, iprint)
        if error != 0:
            print('Error running Mascaret')
#        else:
#            print('Running Mascaret OK')
        return error

    # wrapper 8bis : run Mascaret simulation with specified BC
    # TO DO avec CALCUL_MASCARET_CONDITION_LIMITE qui prend en argument
    # les vecteurs decrivant les CL. La variable mascaret est
    # Modele.Lois.Debit n'est pas utile avec cet appel lorsqu'on veut modifier
    # la BC, contrairement à ce que fait Fabrice avec CALCUL_MASCARET

    def error_message(self, id_masc):
        """wrapper 9 : error message."""
        idmasc = c_int(id_masc)
        err_mess_c = POINTER(c_char_p)()
        error = self.libmascaret.C_GET_ERREUR_MASCARET(idmasc, byref(err_mess_c))
        if error != 0:
            return -1
        return string_at(err_mess_c)

    def get_info_allBC(self, id_masc):
        """wrapper 10 : nb conditions limites."""
        # Rating curve do not count
        idmasc = c_int(id_masc)
        nbBC = c_int()
        error = self.libmascaret.C_GET_NB_CONDITION_LIMITE_MASCARET(idmasc, byref(nbBC))
        if error != 0:
            print('Error getting the number of boundary conditions')

        l_name_allBC = []
        l_num_allBC = []
        for k in range(nbBC.value):
            NumCL = c_int(nbBC.value)
            NomCL = POINTER(c_char_p)()
            NumLoi = c_int()
            error = self.libmascaret.C_GET_NOM_CONDITION_LIMITE_MASCARET(
                idmasc, k + 1, byref(NomCL), byref(NumLoi))
            if error != 0:
                print('Error getting the name of boundary conditions')
            l_name_allBC.append(string_at(NomCL))
            l_num_allBC.append(NumLoi.value)

        return error, nbBC, l_name_allBC, l_num_allBC

    def get_BC_Qt(self, id_masc):
        """wrapper 10 bis."""
        idmasc = c_int(id_masc)
        var_name = c_char_p('Model.Graph.Discharge')
    # Nb of BC
        size1 = c_int()
    # Nb of time steps in BC
        size2 = c_int()
    # Not used (0)
        size3 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
        print('size Model.Graph.Discharge= ', size1.value, size2.value, size3.value)

        tab_Q_BC = np.ones((size1.value, size2.value), float)
        for k in range(size1.value):
            for kk in range(size2.value):
                Q_BC_c = c_double()
                num_BC_c = c_int(k + 1)
                indextime_BC_c = c_int(kk + 1)
                error = self.libmascaret.C_GET_DOUBLE_MASCARET(
                    idmasc, var_name, num_BC_c, indextime_BC_c, 0, byref(Q_BC_c))
                tab_Q_BC[k, kk] = Q_BC_c.value
        return error, tab_Q_BC

    def set_BC_Qt(self, id_masc, new_tab_Q_BC):
        """wrapper 10 ter."""
        idmasc = c_int(id_masc)
        var_name = c_char_p('Model.Graph.Discharge')
    # Nb of BC
        size1 = c_int()
    # Nb of time steps in BC
        size2 = c_int()
    # Not used (0)
        size3 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
    #    print 'size Model.Graph.Discharge= ' , size1.value, size2.value, size3.value

        for k in range(size1.value):
            for kk in range(size2.value):
                Q_BC_c = c_double()
                num_BC_c = c_int(k + 1)
                indextime_BC_c = c_int(kk + 1)
                Q_BC_c.value = new_tab_Q_BC[k, kk]
                error = self.libmascaret.C_SET_DOUBLE_MASCARET(
                    idmasc, var_name, num_BC_c, indextime_BC_c, 0, byref(Q_BC_c))
        return error

    def get_indzonefrot(self, id_masc):
        idmasc = c_int(id_masc)
        size1 = c_int()
        size2 = c_int()
        size3 = c_int()
    
        var_name = c_char_p('Model.FrictionZone.FirstNode')
        error =self.libmascaret.C_GET_TAILLE_VAR_MASCARET(idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
        print('Number of Friction Zones =', size1.value)
        l_ind_beg_zone = []
        for k in range(size1.value):
           ind_beg_zone_c = c_int()
           error = self.libmascaret.C_GET_INT_MASCARET(idmasc, var_name, k+1, 0, 0, byref(ind_beg_zone_c))
           l_ind_beg_zone.append(ind_beg_zone_c.value)
    
        var_name = c_char_p('Model.FrictionZone.LastNode')
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
        l_ind_end_zone = []
        for k in range(size1.value):
           ind_end_zone_c = c_int()
           error = self.libmascaret.C_GET_INT_MASCARET(idmasc, var_name, k+1, 0, 0, byref(ind_end_zone_c))
           l_ind_end_zone.append(ind_end_zone_c.value)
    
        return error, l_ind_beg_zone, l_ind_end_zone
    
    def change_zonefriction_minor(self, id_masc, indexZone, newZoneCF1):
    
        error, l_ind_beg_zone,l_ind_end_zone = self.get_indzonefrot(id_masc)
        Ind_BegZone = l_ind_beg_zone[indexZone]
        Ind_EndZone = l_ind_end_zone[indexZone]
    
        for index in range(Ind_BegZone, Ind_EndZone+1):
            print(index)
            error = self.change_friction_minor(id_masc,index,newZoneCF1)
    
        return error


    def change_friction_minor(self, id_masc, index, newCF1):
        """wrapper 11 : Change minor friction coefficient CF1 at given index."""
        idmasc = c_int(id_masc)
        var_name = c_char_p('Model.FricCoefMainCh')
        size1 = c_int()
        size2 = c_int()
        size3 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
        print('size Model.FricCoefMinCh = ', size1.value, size2.value, size3.value)
        CF1_c = c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, index, 0, 0, byref(CF1_c))

        print('CF1 old value=', CF1_c.value)
        newCF1_c = c_double(newCF1)
        print('newCF1_c = ', newCF1_c)
        print('CF1 new value= ', newCF1_c.value)
        error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            idmasc, var_name, index, 0, 0, newCF1_c)

        return error

    def change_friction_major(self, id_masc, index, newCF2):
        """wrapper 11b : Change major friction coefficient CF2 at given index."""
        idmasc = c_int(id_masc)
        var_name = c_char_p('Model.FricCoefFP')
        size1 = c_int()
        size2 = c_int()
        size3 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            idmasc, var_name, 0, byref(size1), byref(size2), byref(size3))
        CF2_c = c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, index, 0, 0, byref(CF2_c))

        print('CF2 old value=', CF2_c.value)
        newCF2_c = c_double(newCF2)
        print('newCF2_c = ', newCF2_c)
        print('CF2 new value= ', newCF2_c.value)
        error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            idmasc, var_name, index, 0, 0, newCF2_c)

        return error

    def get_state(self, id_masc, index):
        """wrapper 12 : Get State at given index."""
        idmasc = c_int(id_masc)
        var_name = c_char_p('State.Z')

        itemp0 = c_int()
        itemp1 = c_int()
        itemp2 = c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(idmasc, var_name, 0, byref(itemp0),
                                                     byref(itemp1), byref(itemp2))
        print("itemp", itemp0.value, itemp1.value, itemp2.value)

        Z_res_c = c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            idmasc, var_name, index, 0, 0, byref(Z_res_c))
        return error, Z_res_c
