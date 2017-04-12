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
from collections import OrderedDict
import ctypes
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.patches import Polygon
from io import StringIO as cStringIO
logging.basicConfig(level=logging.DEBUG) #.INFO

class MascaretApi(object):

    """Mascaret API."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings, user_settings):
        """Constructor
           1. Loads the Mascaret library with class method "load_mascaret".
           2. Creates an instance of Mascaret with class method "create_model".
           3. Reads model files from "settings" with class method "file_model".
           4. Gets model size with class method "get_model_size".
           5. Gets the simulation times with class method "get_simu_times".
           6. Reads and applies user defined parameters from "user_settings".
           7. Initializes the model with class method "init_model".
        """

        # Load the library mascaret.so
        path = os.path.dirname(os.path.realpath(__file__))
        libmascdir = os.path.join(path, 'lib')
#        os.putenv("LD_LIBRARY_PATH",libmascdir + ":" + os.environ["LD_LIBRARY_PATH"]) 
        os.system("export LD_LIBRARY_PATH=" + libmascdir + ":$LD_LIBRARY_PATH")
        print ('toto',os.environ["LD_LIBRARY_PATH"])
        self.load_mascaret(libmascdir)

        # Create an instance of MASCARET
        self.create_model()

        # Read model files
        self.file_model(settings)

        # Get model size
        self.get_model_size()

        # Get the simulation times
        self.get_simu_times() 

        # Read and apply user defined parameters
        self.user_defined(user_settings)

        # Initialize model
        self.init_model()

    def load_mascaret(self, libmascdir):
        """Loads the Mascaret library situated in the directory "libmascdir"
        """
        self.logger.info('Using MascaretApi')
        libmascaret = libmascdir+'/mascaret.so'
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


    def get_model_size(self):
        '''Gets model size (number of nodes). 
           Uses C_GET_TAILLE_VAR_MASCARET.'''
        var_name = ctypes.c_char_p(b'Model.X')
        nb_nodes = ctypes.c_int()
        il_temp1 = ctypes.c_int()
        il_temp2 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(self.id_masc, var_name, 0,
                                                       ctypes.byref(nb_nodes),
                                                       ctypes.byref(il_temp1),
                                                       ctypes.byref(il_temp2))
        self.nb_nodes = nb_nodes
        if error != 0:
            self.logger.error("Error while getting size var in model  #{}, {}"
                              .format(self.id_masc, self.error_message()))
            return -1
        else:
            self.logger.debug('Get nb of nodes OK, size_X={}'
                              .format(nb_nodes.value))
            return error, nb_nodes

    def init_model(self):
        '''Initializes the model from constant values ("init_cst" in "user_settings" 
           along with "Q_cst" and "Z_cst" values) or from file.lig in "settings". 
           Uses Mascaret Api C_INIT_LIGNE_MASCARET or C_INIT_ETAT_MASCARET.'''
        if 'init_cst' in self.user_settings:
            # Initialize Mascaret Model from values
            Q = self.user_settings['init_cst']['Q_cst'] * self.nb_nodes.value
            Z = self.user_settings['init_cst']['Z_cst'] * self.nb_nodes.value
            Q_c = (ctypes.c_double * self.nb_nodes.value)(Q)
            Z_c = (ctypes.c_double * self.nb_nodes.value)(Z)
            error = self.libmascaret.C_INIT_LIGNE_MASCARET(
                self.id_masc, ctypes.byref(Q_c), ctypes.byref(Z_c), self.nb_nodes)
            if error != 0:
                self.logger.error("Error while initialising the state of MASCARET: {}"
                                  .format(self.error_message()))
            else:
                self.logger.debug(
                        'State constant initialisation successfull from constant value...OK')
        else:
            # Initialize Mascaret Model from file
            print ('file lig', self.settings['files']['lig'])
            print ('file lig', type(self.settings['files']['lig']) )
            init_file_name_c = (ctypes.c_char_p)(*[self.settings['files']['lig']])
            error = self.libmascaret.C_INIT_ETAT_MASCARET(
                                      self.id_masc, init_file_name_c, self.iprint)
            if error != 0:
                self.logger.error("Error while initialising the state of Mascaret from .lig: {}"
                                  .format(self.error_message()))
            else:
                self.logger.debug(
                    'State initialisation successfull from lig...OK')

    def get_simu_times(self):
        '''Gets the simulation times from .xcas in "settings". 
           Uses Mascaret Api C_GET_DOUBLE_MASCARET.'''
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
        
    def create_model(self):       
        '''Creates an instance of Mascaret. 
           Uses Mascaret Api C_CREATE_MASCARET.''' 
        id_masc = ctypes.c_int()
        error = self.libmascaret.C_CREATE_MASCARET(ctypes.byref(id_masc))
        if error != 0:
            self.logger.error("Error while creating a MASCARET model: {}"
                              .format(self.error_message()))
        else:
            self.id_masc = ctypes.c_int(id_masc.value)
        # .opt and .lis are writen only if iprint = 1 at the import AND the calcul steps
        self.iprint = 1

    def file_model(self, settings):
        '''Reads model files from "settings" which is a JSON file (.xcas, .geo, .lig, .loi, .dtd). 
           Uses Mascaret Api C_IMPORT_MODELE_MASCARET.'''
        with open(settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.settings = json.loads(file, encoding="utf-8", object_pairs_hook=OrderedDict)
        self.nb_bc = None
        file_type = []
        file_name = []
        for val in self.settings['files'].items():
            if not isinstance(val[1], list):
                file_name.append(val[1].encode('utf8'))
                self.settings['files'][val] = val[1].encode('utf8')
                file_type.append(val[0].encode('utf8'))
            else:
                for sub in val[1]:
                    file_name.append(sub.encode('utf8'))
                    self.settings['files'][val] = val[1].encode('utf8')
                    file_type.append(val[0].encode('utf8'))
#        for val in self.settings['files'].items():
#            if not isinstance(val[1], list):
#                file_name.append(val[1].encode('utf8'))
#                file_type.append(val[0].encode('utf8'))
#            else:
#                for sub in val[1]:
#                    file_name.append(sub.encode('utf8'))
#                    file_type.append(val[0].encode('utf8'))

        # Import a model
        L_file = len(file_name)
        file_name_c = (ctypes.c_char_p * L_file)(*file_name)
        file_type_c = (ctypes.c_char_p * L_file)(*file_type)
        error = self.libmascaret.C_IMPORT_MODELE_MASCARET(self.id_masc, file_name_c,
                                                  file_type_c, L_file, self.iprint)
        if error != 0:
            self.logger.error("Error while importing a MASCARET model: {}"
                          .format(self.error_message()))
        print(file_name)
        print(file_type)

    def __del__(self):
        '''Deletes a model.'''
        error = self.libmascaret.C_DELETE_MASCARET(self.id_masc)
        if error != 0:
            self.logger.error("Error while deleting the instantiation #{}"
                              .format(self.id_masc, self.error_message()))

    def __repr__(self):
        '''Framework for the method 'print' '''
        string = 'MODEL FILES:\n'
        string += ' -- xcas: {}\n'
        string += ' -- geo: {}\n'
        string += ' -- res: {}\n'
        string += ' -- listing: {}\n'
        string += ' -- damocle: {}\n'
        string += ' -- lig: {}\n'
        string += ' -- loi:\n'
        for file1 in self.settings['files']['loi']:
            string += '         {}\n'
        string += '\nUSER SETTINGS:\n'
        if 'Q_BC' in self.user_settings:
            string += ' -- Change the upstream flow rate:\n'
            string += '       > Index: {}\n'
            string += '       > Value: {}\n'
        if 'Ks' in self.user_settings:
            string += ' -- Change the friction coefficient:\n'
            string += '       > By zone: {}\n'
            string += '       > Index: {}\n'
            string += '       > Value: {}\n'
            string += '       > Zone index: {}\n'
        if 'MC' in self.user_settings:
            string += ' -- Monte-Carlo settings:\n'
            string += '       > Ks distribution: {}\n'
            string += '       > Ks parameter 1: {}\n'
            string += '       > Ks parameter 2: {}\n'
            string += '       > Number of simulations: {}\n'
        if 'misc' in self.user_settings:
            string += ' -- Miscellaneous:\n'
            string += '       > Print  boundary conditions: {}\n'
            string += '       > Output index: {}'

        src1 = list(itertools.chain.from_iterable([ v.values() if isinstance(v,dict) else [v] for v in self.settings['files'].values()]))
        src2 = list(itertools.chain.from_iterable([ v.values() if isinstance(v,dict) else [v] for v in self.user_settings.values()]))
        src = list(itertools.chain.from_iterable(v for v in [src1,src2]))
        src_ = []
        for v in src:
            if isinstance(v, list):
                for w in v:
                    src_.append(w)
            else:
                src_.append(v)
        return string.format(*src_)

    def run_mascaret(self): 
        """Runs Mascaret simulation. 
           Uses Mascaret Api C_CALCUL_MASCARET."""

        self.empty_opt()         

        error = self.libmascaret.C_CALCUL_MASCARET(self.id_masc, self.t0,
                                                   self.tend, self.dt, self.iprint)
        if error != 0:
            self.logger.error("Error running Mascaret: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Running Mascaret OK')

        return self.state(self.user_settings['misc']['index_outstate']).value

    def __call__(self, saveall=False): 

        '''Runs the application from "user_settings'''

        settings = self.user_settings

        if 'MC' in settings:

            if 'Ne' in settings['MC']:
                N = settings['MC']['Ne']
            else:
                N = 1

            nx = ('distKs' in settings['MC']) + ('distQ' in settings['MC'])
    
            X = np.zeros((settings['MC']['Ne'],nx))

            h = []

            for i in range(N):

                j = -1
            
                if 'distKs' in settings['MC']:
                    j += 1
                    if settings['MC']['distKs'] == "G": 
                        Ks = np.random.normal(settings['MC']['muKs'], settings['MC']['sigmaKs'])               
                    elif settings['MC']['distKs'] == "U":
                        Ks = np.random.uniform(settings['MC']['minKs'], settings['MC']['maxKs']) 
                    X[i,j] = Ks
                    if settings['Ks']['zone']:
                        self.zone_friction_minor = {'ind_zone': settings['Ks']['ind_zone'], 'value': Ks}
                    else:
                        self.friction_minor = {'idx': settings['Ks']['idx'], 'value': Ks} 

                if 'distQ' in settings['MC']:
                    j += 1
                    if settings['MC']['distQ'] == "G": 
                        Q = np.random.normal(settings['MC']['muQ'], settings['MC']['sigmaQ'])               
                    elif settings['MC']['distQ'] == "U":
                        Q = np.random.uniform(settings['MC']['minQ'], settings['MC']['maxQ'])
                    X[i,j] = Q
                    self.bc_qt = {'idx': settings['Q_BC']['idx'], 'value': Q} 

                h.append(self.run_mascaret())

                if saveall:
                   os.rename('ResultatsOpthyca.opt', 'ResultatsOpthyca_'+str(i)+'.opt')

        else:

            h = self.run_mascaret()

        self.results = h
        self.DOE = X

        return h


    def user_defined(self, user_settings):
        '''Reads user parameters from "user_settings" and applies values for Q_BC ("Q_BC={'idx','value'}")
            and Ks ("Ks={'zone','idx','value','ind_zone'}") (the Ks for 1 point or 1 zone). 
            Uses the class methods "zone_friction_minor", "friction_minor" and "bc_qt".
        '''
        with open(user_settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.user_settings = json.loads(file, encoding="utf-8", object_pairs_hook=OrderedDict)
        print ('fichiers mascaret', self.user_settings)
        if 'Q_BC' in self.user_settings:
            self.bc_qt = self.user_settings['Q_BC']
        if 'Ks' in self.user_settings:
            if self.user_settings['Ks']['zone'] is True:
                self.zone_friction_minor = self.user_settings['Ks']
            else:
                self.friction_minor = self.user_settings['Ks']

    def error_message(self):
        """Error message wrapper."""
        err_mess_c = ctypes.POINTER(ctypes.c_char_p)()
        error = self.libmascaret.C_GET_ERREUR_MASCARET(self.id_masc,
                                                       ctypes.byref(err_mess_c))
        if error != 0:
            return 'Error could not be retrieved from MASCARET...'
        return ctypes.string_at(err_mess_c)

    def info_all_bc(self):
        """Returns numbers and names of all boundary conditions. 
           Uses Mascaret Api C_GET_NOM_CONDITION_LIMITE_MASCARET."""
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
        '''Gets boundary conditions Qt. 
           Uses Mascaret Api C_GET_TAILLE_VAR_MASCARET and C_GET_DOUBLE_MASCARET.'''
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

        if self.user_settings['misc']['info_bc'] is True:
            if self.nb_bc is None:
                self.info_all_bc()
            for k in range(self.nb_bc):
                self.logger.info("Loi Q: {} {} {}".format(self.l_name_all_bc[k],
                                                          self.l_num_all_bc[k],
                                                          bc_qt[k, :]))

        return bc_qt

    @bc_qt.setter
    def bc_qt(self, q_bc):
        '''Sets boundary condition Qt using "self.bc_qt={'idx','value'}". 
           Uses Mascaret Api C_GET_TAILLE_VAR_MASCARET and C_SET_DOUBLE_MASCARET.'''
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
        '''Gets indices of the beginning and end of all the friction zones. 
           Uses Mascaret Api C_GET_TAILLE_VAR_MASCARET and C_GET_INT_MASCARET.
        '''
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

        return error, l_ind_beg_zone, l_ind_end_zone

    @property
    def zone_friction_minor(self):
        '''Gets minor friction coefficient at zone "self.ind_zone". 
           Uses class attributes "ind_zone_frot" and "friction_minor".'''
        error, l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot
        Ind_BegZone = l_ind_beg_zone[self.ind_zone]
        Ind_EndZone = l_ind_end_zone[self.ind_zone]

        zone_friction = []

        for index in range(Ind_BegZone, Ind_EndZone + 1):
            zone_friction.append(self.friction_minor)

        #self.ks_idx = self.user_settings['Ks']['idx']

        if error != 0:
            self.logger.error("Error getting friction minor on zone: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Get Zone KS OK')

        return zone_friction

    @zone_friction_minor.setter
    def zone_friction_minor(self, Ks):
        '''Changes minor friction coefficient at zone "Ks['ind_zone']" using "Ks['value']". 
           Uses class attribute "ind_zone_frot" and class method "friction_minor".'''
        ind_zone, value = Ks['ind_zone'], Ks['value']
        error, l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot#  self.get_indzonefrot
        Ind_BegZone = l_ind_beg_zone[ind_zone]
        Ind_EndZone = l_ind_end_zone[ind_zone]
        self.ind_zone = ind_zone
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
        """Gets minor friction coefficient at index "self.ks_idx". 
           Uses Mascaret Api C_GET_TAILLE_VAR_MASCARET and C_GET_DOUBLE_MASCARET."""
        var_name = ctypes.c_char_p(b'Model.FricCoefMainCh')
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1),
            ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('size Model.FricCoefMinCh = {} {} {}'
                          .format(size1.value, size2.value, size3.value))
        Ks_c = ctypes.c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, ctypes.byref(Ks_c))
        if error != 0:
            self.logger.error("Error setting friction minor: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Ks old value= {}'.format(Ks_c.value))

        return Ks_c.value

    @friction_minor.setter
    def friction_minor(self, Ks):
        '''Changes minor friction coefficient Ks at index "Ks['idx']" with "Ks['value']". 
           Uses Mascaret Api C_SET_DOUBLE_MASCARET.'''
        var_name = ctypes.c_char_p(b'Model.FricCoefMainCh')
        Ks_c = ctypes.c_double(Ks['value'])
        self.logger.debug('Ks_c = {}'.format(Ks_c))
        self.logger.debug('Ks new value= {}'.format(Ks_c.value))
        self.ks_idx = Ks['idx']
        error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, Ks_c)

        if error != 0:
            self.logger.error("Error setting friction minor: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Change KS OK')

    def state(self, index):
        '''Gets state at given index in "user_settings['misc']['index_outstate']". 
           Uses Mascaret Api C_GET_TAILLE_VAR_MASCARET and C_GET_DOUBLE_MASCARET.'''
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

    def read_opt(self, filename='ResultatsOpthyca.opt'):

        '''Reads the results file 'ResultatsOpthyca.opt' '''

        with open(filename,'rb') as myfile:
            data = myfile.read().replace('"', '')
        data = np.genfromtxt(cStringIO.StringIO(data), delimiter=';', skip_header=14)
        self.opt = data

    def plot_opt(self, xlab='Curvilinear abscissa (m)', ylab1='Water level (m)', ylab2='Flow rate (m3/s)',
                 title='Water level along the open-channel at final time'):
        
        '''Plots results contained in the results file 'ResultatsOpthyca.opt' '''
 
        if not hasattr(self, 'opt'):
            self.read_opt()

        nb = int(max(self.opt[:,2]))
        x = self.opt[-nb:-1,3]
        level = self.opt[-nb:-1,5]
        bathy = self.opt[-nb:-1,4]
        flowrate = self.opt[-nb:-1,-1]

        fig, ax1 = plt.subplots()
        ax1.plot(x, bathy, color='black')
        ax1.plot(x, level, color='blue')
        ax1.fill_between(x, bathy, level, facecolor='blue', alpha=0.5)
        ax1.set_xlabel(xlab)
        ax1.set_ylabel(ylab1, color='blue')
        ax1.tick_params('y', colors='blue')
        y_formatter = tick.ScalarFormatter(useOffset=False)
        ax1.yaxis.set_major_formatter(y_formatter)
        ax2 = ax1.twinx()
        ax2.plot(x, flowrate, color='red')
        ax2.set_ylabel(ylab2, color='red')
        ax2.tick_params('y', colors='red')
        ax2.yaxis.set_major_formatter(y_formatter)
        plt.title(title)
        plt.show()

    def empty_opt(self):
        open("ResultatsOpthyca.opt", 'w').close()
