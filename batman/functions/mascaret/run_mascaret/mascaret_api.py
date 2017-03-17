# coding: utf-8
"""
MascaretAPI class
=================

Python Wrapper for Mascaret.

"""
import sys
import os
import logging
import json
import ctypes
import itertools
import numpy as np
logging.basicConfig(level=logging.DEBUG)


class MascaretApi(object):

    """Mascaret API."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings):
        """Loading the Mascaret library.

        # TODO xcas read and use: Matthias

        :param str settings: settings.json file
        """
        self.logger.info('Using MascaretApi')
        path = os.path.dirname(os.path.realpath(__file__))
        libmascaret = os.path.join(path, 'lib/mascaret.so')
        self.logger.debug(libmascaret)
        if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            try:
                self.libmascaret = ctypes.CDLL(libmascaret)
            except Exception as tb:
                self.logger.exception("Unable to load: mascaret.so. Check the "
                                      "environment variable LIBMASCARET: {}"
                                      .format(tb))
                raise SystemExit
        else:
            self.logger.error('Unsupported OS: macOS or Unix')
            raise SystemExit

        # Create an instance of MASCARET
        id_masc = ctypes.c_int()
        error = self.libmascaret.C_CREATE_MASCARET(ctypes.byref(id_masc))
        if error != 0:
            self.logger.error("Error while creating a MASCARET model: {}"
                              .format(self.error_message()))
        else:
            self.id_masc = ctypes.c_int(id_masc.value)

        # .opt and .lis are writen only if iprint = 1 at the import AND the calcul steps
        self.iprint = 1

        # Read settings
        with open(settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.settings = json.loads(file, encoding="utf-8")

        self.nb_bc = None

        file_type = []
        file_name = []

        for val in self.settings['files'].items():
            if not isinstance(val[1], list):
                file_name.append(val[1].encode('utf8'))
                file_type.append(val[0].encode('utf8'))
            else:
                for sub in val[1]:
                    file_name.append(sub.encode('utf8'))
                    file_type.append(val[0].encode('utf8'))

        # Import a model
        L_file = len(file_name)
        file_name_c = (ctypes.c_char_p * L_file)(*file_name)
        file_type_c = (ctypes.c_char_p * L_file)(*file_type)
        error = self.libmascaret.C_IMPORT_MODELE_MASCARET(self.id_masc, file_name_c,
                                                          file_type_c, L_file, self.iprint)
        if error != 0:
            self.logger.error("Error while importing a MASCARET model: {}"
                              .format(self.error_message()))

        # Get var size
        var_name = ctypes.c_char_p(b'Model.X')
        nb_nodes = ctypes.c_int()
        il_temp1 = ctypes.c_int()
        il_temp2 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(self.id_masc, var_name, 0,
                                                           ctypes.byref(nb_nodes),
                                                           ctypes.byref(il_temp1),
                                                           ctypes.byref(il_temp2))
        if error != 0:
            self.logger.error("Error while getting size var in model  #{}, {}"
                              .format(self.id_masc, self.error_message()))
        else:
            self.logger.debug('Get nb of nodes OK, size_X={}'
                              .format(nb_nodes.value))

        if 'init_lig' in self.settings:
            # Initialize Mascaret Model from values
            Q = self.settings['init_lig']['Q_cst'] * nb_nodes.value
            Z = self.settings['init_lig']['Z_cst'] * nb_nodes.value
            Q_c = (ctypes.c_double * nb_nodes.value)(Q)
            Z_c = (ctypes.c_double * nb_nodes.value)(Z)
            error = self.libmascaret.C_INIT_LIGNE_MASCARET(
                self.id_masc, ctypes.byref(Q_c), ctypes.byref(Z_c), nb_nodes)
            if error != 0:
                self.logger.error("Error while initialising the state of MASCARET: {}"
                                  .format(self.error_message()))
            else:
                self.logger.debug(
                    'State constant initialisation successfull from constant value...OK')
        else:
            # Initialize Mascaret Model from file
            init_file_name_c = (ctypes.c_char_p)(*self.settings['files']['lig'])
            error = self.libmascaret.C_INIT_ETAT_MASCARET(
                self.id_masc, init_file_name_c, self.iprint)
            if error != 0:
                self.logger.error("Error while initialising the state of Mascaret from .lig: {}"
                                  .format(self.error_message()))
            else:
                self.logger.debug(
                    'State constant initialisation successfull from lig...OK')

        # get the simulation times."""
        # dt
        self.dt = ctypes.c_double()
        var_name = ctypes.c_char_p(b'Model.DT')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.dt))
        if error != 0:
            self.logger.error("Error while getting the value of the time step: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('dt={}'.format(self.dt.value))
        # t0
        self.t0 = ctypes.c_double()
        var_name = ctypes.c_char_p(b'Model.InitTime')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.t0))
        if error != 0:
            self.logger.error("Error while getting the value of the initial time: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('t0={}'.format(self.t0.value))
        # tend
        self.tend = ctypes.c_double()
        var_name = ctypes.c_char_p(b'Model.MaxCompTime')
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.tend))
        if error != 0:
            self.logger.error("Error while getting the value of the final time: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('tend={}'.format(self.tend.value))

        # Change Q(t)
        self.bc_qt = self.settings['Q_BC']

        # Get idx zones Ks
        if self.settings['Ks']['ind_zone'] is True:
            l_Ind_DebZonesFrot, l_Ind_EndZonesFrot = self.ind_zone_frot
        self.logger.info('l_Ind_DebZonesFrot = {}\nl_Ind_EndZonesFrot = {}'
                         .format(l_Ind_DebZonesFrot, l_Ind_EndZonesFrot))

        # Change Ks
        self.ks_idx = self.settings['Ks']['idx']
        if self.settings['Ks']['zone'] is False:
            self.friction_minor = self.settings['Ks']
        else:
            self.zone_friction_minor = self.settings['Ks']

    def __del__(self):
        """Delete a model."""
        error = self.libmascaret.C_DELETE_MASCARET(self.id_masc)
        if error != 0:
            self.logger.error("Error while deleting the instantiation #{}"
                              .format(self.id_masc, self.error_message()))

    def run_mascaret(self):
        """Run Mascaret simulation."""
        error = self.libmascaret.C_CALCUL_MASCARET(self.id_masc, self.t0,
                                                   self.tend, self.dt, self.iprint)
        if error != 0:
            self.logger.error("Error running Mascaret: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Running Mascaret OK')

        self.state(self.settings['misc']['index_outstate'])

    def error_message(self):
        """Error message wrapper."""
        err_mess_c = ctypes.POINTER(ctypes.c_char_p)()
        error = self.libmascaret.C_GET_ERREUR_MASCARET(self.id_masc,
                                                       ctypes.byref(err_mess_c))
        if error != 0:
            return 'Error could not be retrieved from MASCARET...'
        return ctypes.string_at(err_mess_c)

    def info_all_bc(self):
        """Nb boundary conditions."""
        # Rating curve do not count
        nb_bc = ctypes.c_int()
        error = self.libmascaret.C_GET_NB_CONDITION_LIMITE_MASCARET(
            self.id_masc, ctypes.byref(nb_bc))
        if error != 0:
            self.logger.error("Error getting the number of boundary conditions: {}"
                              .format(self.error_message()))

        self.nb_bc = nb_bc.value

        l_name_all_bc = []
        l_num_all_bc = []
        for k in range(nb_bc.value):
            NumCL = ctypes.c_int(nb_bc.value)
            NomCL = ctypes.POINTER(ctypes.c_char_p)()
            NumLoi = ctypes.c_int()
            error = self.libmascaret.C_GET_NOM_CONDITION_LIMITE_MASCARET(
                self.id_masc, k + 1, ctypes.byref(NomCL), ctypes.byref(NumLoi))
            if error != 0:
                self.logger.error("Error getting the name of boundary conditions: {}"
                                  .format(self.error_message()))
            l_name_all_bc.append(ctypes.string_at(NomCL))
            l_num_all_bc.append(NumLoi.value)

        self.l_name_all_bc = l_name_all_bc
        self.l_num_all_bc = l_num_all_bc

        return nb_bc, l_name_all_bc, l_num_all_bc

    @property
    def bc_qt(self):
        """Wrapper getter Boundary condition Qt."""
        var_name = ctypes.c_char_p(b'Model.Graph.Discharge')
        # Nb of BC
        size1 = ctypes.c_int()
        # Nb of time steps in BC
        size2 = ctypes.c_int()
        # Not used (0)
        size3 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('size Model.Graph.Discharge= {} {} {}'
                          .format(size1.value, size2.value, size3.value))

        bc_qt = np.ones((size1.value, size2.value), float)
        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            error = self.libmascaret.C_GET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))
            bc_qt[k, kk] = q_bc_c.value

        if error != 0:
            self.logger.error("Error getting discharge: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Get BC Q(t) OK ')

        if self.settings['misc']['bc'] is True:
            if self.nb_bc is None:
                self.info_all_bc()
            for k in range(self.nb_bc):
                self.logger.info("Loi Q: {} {} {}".format(self.l_name_all_bc[k],
                                                          self.l_num_all_bc[k],
                                                          bc_qt[k, :]))

        return bc_qt

    @bc_qt.setter
    def bc_qt(self, q_bc):
        """Wrapper setter Boundary condition Qt."""
        new_tab_q_bc = self.bc_qt
        idx, value = q_bc['idx'], q_bc['value']
        new_tab_q_bc[idx, :] = value

        var_name = ctypes.c_char_p(b'Model.Graph.Discharge')
        # Nb of BC
        size1 = ctypes.c_int()
        # Nb of time steps in BC
        size2 = ctypes.c_int()
        # Not used (0)
        size3 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))

        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            q_bc_c.value = new_tab_q_bc[k, kk]
            error = self.libmascaret.C_SET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))

        if error != 0:
            self.logger.error("Error setting discharge: {}"
                              .format(self.error_message()))

        self.logger.debug('Change Q OK')

    @property
    def ind_zone_frot(self):
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()

        var_name = ctypes.c_char_p(b'Model.FrictionZone.FirstNode')
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Number of Friction Zones = {}'.format(size1.value))
        l_ind_beg_zone = []
        for k in range(size1.value):
            ind_beg_zone_c = ctypes.c_int()
            error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_beg_zone_c))
            l_ind_beg_zone.append(ind_beg_zone_c.value)

        if error != 0:
            self.logger.error("Error getting first node friction zone: {}"
                              .format(self.error_message()))

        var_name = ctypes.c_char_p(b'Model.FrictionZone.LastNode')
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        l_ind_end_zone = []
        for k in range(size1.value):
            ind_end_zone_c = ctypes.c_int()
            error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_end_zone_c))
            l_ind_end_zone.append(ind_end_zone_c.value)

        if error != 0:
            self.logger.error("Error getting last node friction zone: {}"
                              .format(self.error_message()))

        return l_ind_beg_zone, l_ind_end_zone

    @property
    def zone_friction_minor(self):
        """Get minor friction coefficient CF1 at zone."""
        error, l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot
        Ind_BegZone = l_ind_beg_zone[self.ks_idx]
        Ind_EndZone = l_ind_end_zone[self.ks_idx]

        zone_friction = []

        for index in range(Ind_BegZone, Ind_EndZone + 1):
            zone_friction.append(self.friction_minor)

        self.ks_idx = self.settings['Ks']['idx']

        if error != 0:
            self.logger.error("Error getting friction minor on zone: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Get Zone KS OK')

        return zone_friction

    @zone_friction_minor.setter
    def zone_friction_minor(self, cf_1):
        """Change minor friction coefficient CF1 at zone."""
        idx, value = cf_1['idx'], cf_1['value']
        error, l_ind_beg_zone, l_ind_end_zone = self.get_indzonefrot
        Ind_BegZone = l_ind_beg_zone[idx]
        Ind_EndZone = l_ind_end_zone[idx]

        for index in range(Ind_BegZone, Ind_EndZone + 1):
            self.logger.debug(index)
            self.friction_minor = {'idx': index, 'value': value}

        if error != 0:
            self.logger.error("Error setting friction minor on zone: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Change Zone KS OK')

    @property
    def friction_minor(self):
        """Change minor friction coefficient CF1 at index."""
        var_name = ctypes.c_char_p(b'Model.FricCoefMainCh')
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1),
            ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('size Model.FricCoefMinCh = {} {} {}'
                          .format(size1.value, size2.value, size3.value))
        CF1_c = ctypes.c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, ctypes.byref(CF1_c))
        if error != 0:
            self.logger.error("Error setting friction minor: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('CF1 old value= {}'.format(CF1_c.value))

        return CF1_c.value

    @friction_minor.setter
    def friction_minor(self, cf_1):
        """Change minor friction coefficient CF1 at index."""
        var_name = ctypes.c_char_p(b'Model.FricCoefMainCh')
        cf_1_c = ctypes.c_double(cf_1)
        self.logger.debug('cf_1_c = {}'.format(cf_1_c))
        self.logger.debug('CF1 new value= {}'.format(cf_1_c.value))
        error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, cf_1_c)

        if error != 0:
            self.logger.error("Error setting friction minor: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Change KS OK')

    @property
    def friction_major(self):
        """Get major friction coefficient cf_2 at index."""
        var_name = ctypes.c_char_p(b'Model.FricCoefFP')
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        cf_2_c = ctypes.c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, ctypes.byref(cf_2_c))
        if error != 0:
            self.logger.error("Error getting friction major: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('cf_2 value={}'.format(cf_2_c.value))

        return cf_2_c.value

    @friction_major.setter
    def friction_major(self, cf_2):
        """Change major friction coefficient cf_2 at index."""
        idx, value = cf_2['idx'], cf_2['value']
        var_name = ctypes.c_char_p(b'Model.FricCoefFP')
        cf_2_c = ctypes.c_double(value)
        self.logger.debug('cf_2_c = {}'.format(cf_2_c))
        self.logger.debug('CF2 new value= {}'.format(cf_2_c.value))
        error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            self.id_masc, var_name, idx, 0, 0, cf_2_c)

        if error != 0:
            self.logger.error("Error setting friction major: {}"
                              .format(self.error_message()))

    def state(self, index):
        """Get State at given index."""
        var_name = ctypes.c_char_p(b'State.Z')

        itemp0 = ctypes.c_int()
        itemp1 = ctypes.c_int()
        itemp2 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(self.id_masc, var_name, 0, ctypes.byref(itemp0),
                                                           ctypes.byref(itemp1), ctypes.byref(itemp2))
        self.logger.debug('itemp {} {} {}'
                          .format(itemp0.value, itemp1.value, itemp2.value))

        Z_res_c = ctypes.c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, index, 0, 0, ctypes.byref(Z_res_c))

        if error != 0:
            self.logger.error("Error getting state: {}"
                              .format(self.error_message()))

        return Z_res_c
