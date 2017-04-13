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
from io import StringIO

logging.basicConfig(level=logging.DEBUG)


class MascaretApi(object):

    """Mascaret API."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings, user_settings):
        """Constructor.

        1. Loads the Mascaret library with :meth:`MascaretApi.load_mascaret`,
        2. Creates an instance of Mascaret with :meth:`MascaretApi.create_model`,
        3. Reads model files from "settings" with :meth:`MascaretApi.file_model`,
        4. Gets model size with :meth:`MascaretApi.model_size`,
        5. Gets the simulation times with :meth:`MascaretApi.simu_times`,
        6. Reads and applies user defined parameters from ``user_settings``,
        7. Initializes the model with :meth:`MascaretApi.init_model`.
        """
        self.logger.info('Using MascaretApi')
        # Load the library mascaret.so
        libmascaret = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   'lib/mascaret.so')
        self.load_mascaret(libmascaret)

        # Create an instance of MASCARET
        self.create_model()

        # Read model files
        self.file_model(settings)

        # Get model size
        self.nb_nodes = self.model_size

        # Get the simulation times
        self.simu_times

        # Read and apply user defined parameters
        self.user_defined(user_settings)

        # Initialize model
        self.init_model()

    def load_mascaret(self, libmascaret):
        """Load Mascaret library.

        :param str libmascaret: path to the library
        """
        ld_library = os.environ['LD_LIBRARY_PATH']
        self.logger.debug('LD_LIBRARY_PATH: {}'.format(ld_library))
        self.logger.info('Loading {}...'.format(libmascaret))
        if sys.platform.startswith('linux') or sys.platform.startswith('darwin'):
            try:
                self.libmascaret = ctypes.CDLL(libmascaret)
            except Exception as tb:
                self.logger.exception("Unable to load: mascaret.so. Check the "
                                      "environment variable LIBMASCARET: {}"
                                      .format(tb))
                raise SystemExit
            else:
                self.logger.info('Library loaded.')
        else:
            self.logger.error('Unsupported OS. Only macOS or Unix!')
            raise SystemExit

    @property
    def model_size(self):
        """Get model size (number of nodes).

        Uses :meth:`C_GET_TAILLE_VAR_MASCARET`.

        :return: Size of the model
        :rtype: int
        """
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
            return nb_nodes

    def init_model(self):
        """Initialize the model from constant values.

        ``init_cst`` in :attr:`user_settings` along with ``Q_cst`` and
        ``Z_cst`` values or from :file:`file.lig` in :attr:`settings`. Uses
        Mascaret Api :meth:`C_INIT_LIGNE_MASCARET` or :meth:`C_INIT_ETAT_MASCARET`.
        """
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
            init_file_name_c = (ctypes.c_char_p)(*[self.settings['files']['lig']])
            error = self.libmascaret.C_INIT_ETAT_MASCARET(
                self.id_masc, init_file_name_c, self.iprint)
            if error != 0:
                self.logger.error("Error while initialising the state of Mascaret from .lig: {}"
                                  .format(self.error_message()))
            else:
                self.logger.debug(
                    'State initialisation successfull from lig...OK')

    @property
    def simu_times(self):
        """Get the simulation times from :file:`.xcas` in :attr:`settings`.

        Uses Mascaret Api :meth:`C_GET_DOUBLE_MASCARET`.

        :return: time step, initial time and final time
        :rtype: tuple(float)
        """
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

        return (self.dt.value, self.t0.value, self.tend.value)

    def create_model(self):
        """Create an instance of Mascaret.

        Uses Mascaret Api :meth:`C_CREATE_MASCARET`.
        """
        id_masc = ctypes.c_int()
        error = self.libmascaret.C_CREATE_MASCARET(ctypes.byref(id_masc))
        if error != 0:
            self.logger.error("Error while creating a MASCARET model: {}"
                              .format(self.error_message()))
        else:
            self.id_masc = ctypes.c_int(id_masc.value)
        # .opt and .lis written only if iprint = 1 at import AND calcul steps
        self.iprint = 1

    def file_model(self, settings):
        """Read model files from :file:`settings` which is a *JSON* file.

        (.xcas, .geo, .lig, .loi, .dtd)
        Uses Mascaret Api :meth:`C_IMPORT_MODELE_MASCARET`.

        :param str settings: path of *JSON* settings file
        """
        with open(settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.settings = json.loads(
                file, encoding="utf-8", object_pairs_hook=OrderedDict)
        self.nb_bc = None
        # Convert all values of settings from str to bytes
        file_type = []
        file_name = []
        for key_val in self.settings['files'].items():
            try:
                value = key_val[1].encode('utf8')
                file_name.append(value)
                self.settings['files'][key_val[0]] = value
                file_type.append(key_val[0].encode('utf8'))
            except AttributeError:  # In case of a list, loop over it
                for i, sub in enumerate(key_val[1]):
                    sub_value = sub.encode('utf8')
                    file_name.append(sub_value)
                    self.settings['files'][key_val[0]][i] = sub_value
                    file_type.append(key_val[0].encode('utf8'))

        # Import a model
        L_file = len(file_name)
        file_name_c = (ctypes.c_char_p * L_file)(*file_name)
        file_type_c = (ctypes.c_char_p * L_file)(*file_type)
        error = self.libmascaret.C_IMPORT_MODELE_MASCARET(self.id_masc, file_name_c,
                                                          file_type_c, L_file, self.iprint)
        if error != 0:
            self.logger.error("Error while importing a MASCARET model: {}"
                              .format(self.error_message()))
        else:
            self.logger.info("Model imported with:\n-> file_name: {}\n-> file_type: {}"
                             .format(file_name, file_type))

    def __del__(self):
        """Delete a model."""
        error = self.libmascaret.C_DELETE_MASCARET(self.id_masc)
        if error != 0:
            self.logger.error("Error while deleting the instantiation #{}:\n{}"
                              .format(self.id_masc, self.error_message()))
        else:
            self.logger.debug("Model #{} deleted.".format(self.id_masc))

    def __repr__(self):
        """Class informations based on settings."""
        string = ("MODEL FILES:\n"
                  " -- xcas: {}\n"
                  " -- geo: {}\n"
                  " -- res: {}\n"
                  " -- listing: {}\n"
                  " -- damocle: {}\n"
                  " -- lig: {}\n"
                  " -- loi:\n")
        for file1 in self.settings['files']['loi']:
            string += '         {}\n'
        string += '\nUSER SETTINGS:\n'
        if 'Q_BC' in self.user_settings:
            string += (" -- Change the upstream flow rate:\n"
                       "       > Index: {}\n"
                       "       > Value: {}\n")
        if 'Ks' in self.user_settings:
            string += (" -- Change the friction coefficient:\n"
                       "       > By zone: {}\n"
                       "       > Index: {}\n"
                       "       > Value: {}\n"
                       "       > Zone index: {}\n")
        if 'MC' in self.user_settings:
            string += (" -- Monte-Carlo settings:\n"
                       "       > Ks distribution: {}\n"
                       "       > Ks parameter 1: {}\n"
                       "       > Ks parameter 2: {}\n"
                       "       > Number of simulations: {}\n")
        if 'misc' in self.user_settings:
            string += (" -- Miscellaneous:\n"
                       "       > Print  boundary conditions: {}\n"
                       "       > Output index: {}")

        src1 = list(itertools.chain.from_iterable([v.values() if isinstance(
            v, dict) else [v] for v in self.settings['files'].values()]))
        src2 = list(itertools.chain.from_iterable([v.values() if isinstance(
            v, dict) else [v] for v in self.user_settings.values()]))
        src = list(itertools.chain.from_iterable(v for v in [src1, src2]))
        src_ = []
        for v in src:
            if isinstance(v, list):
                for w in v:
                    src_.append(w)
            else:
                src_.append(v)
        return string.format(*src_)

    def run_mascaret(self):
        """Run Mascaret simulation.

        Use Mascaret Api :meth:`C_CALCUL_MASCARET`.

        :return: water level at :attr:`index_outstate`
        :rtype: double
        """
        self.empty_opt()

        self.logger.info('Running Mascaret...')
        error = self.libmascaret.C_CALCUL_MASCARET(self.id_masc, self.t0,
                                                   self.tend, self.dt, self.iprint)
        if error != 0:
            self.logger.error("Error running Mascaret: {}"
                              .format(self.error_message()))
        else:
            self.logger.info('Running Mascaret OK')

        return self.state(self.user_settings['misc']['index_outstate']).value

    def __call__(self, saveall=False):
        """Run the application using :attr:`user_settings`.

        :param bool saveall: Change the default name of the Results file
        """
        settings = self.user_settings

        if 'MC' in settings:
            try:
                n = settings['MC']['Ne']
            except KeyError:
                n = 1

            nx = ('distKs' in settings['MC']) + ('distQ' in settings['MC'])
            self.doe = np.empty((settings['MC']['Ne'], nx))

            if 'distKs' in settings['MC']:
                ks = self.doe[:, 0]
                if settings['MC']['distKs'] == "G":
                    ks[:] = np.random.normal(
                        settings['MC']['muKs'], settings['MC']['sigmaKs'], n)
                elif settings['MC']['distKs'] == "U":
                    ks[:] = np.random.uniform(
                        settings['MC']['minKs'], settings['MC']['maxKs'], n)

            if 'distQ' in settings['MC']:
                if 'distKs' in settings['MC']:
                    q = self.doe[:, 1]
                else:
                    q = self.doe[:, 0]
                if settings['MC']['distQ'] == "G":
                    q[:] = np.random.normal(
                        settings['MC']['muQ'], settings['MC']['sigmaQ'], n)
                elif settings['MC']['distQ'] == "U":
                    q[:] = np.random.uniform(
                        settings['MC']['minQ'], settings['MC']['maxQ'], n)

            h = np.empty(n)

            for i in range(n):
                self.logger.info('Iteration #{}'.format(n))
                if 'distKs' in settings['MC']:
                    if settings['Ks']['zone']:
                        self.zone_friction_minor = {
                            'ind_zone': settings['Ks']['ind_zone'], 'value': ks[i]}
                    else:
                        self.friction_minor = {
                            'idx': settings['Ks']['idx'], 'value': ks[i]}

                if 'distQ' in settings['MC']:
                    self.bc_qt = {'idx': settings['Q_BC']['idx'], 'value': q[i]}

                h[i] = self.run_mascaret()

                if saveall:
                    os.rename('ResultatsOpthyca.opt',
                              'ResultatsOpthyca_' + str(i) + '.opt')

        else:

            h = self.run_mascaret()

        self.results = h

        return h

    def user_defined(self, user_settings):
        """Read user parameters from :file:`user_settings`` and apply values.

        Look for ``Q_BC`` (``Q_BC={'idx','value'}``) and ``Ks`` (``Ks={'zone','idx','value',
        'ind_zone'}``) (the ``Ks`` for 1 point or 1 zone).
        Use :meth:`zone_friction_minor`, :meth:`friction_minor` and :meth:`bc_qt`.

        :param str user_settings: Path of the *JSON* settings file
        """
        with open(user_settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.user_settings = json.loads(
                file, encoding="utf-8", object_pairs_hook=OrderedDict)
        if 'Q_BC' in self.user_settings:
            self.bc_qt = self.user_settings['Q_BC']
        if 'Ks' in self.user_settings:
            if self.user_settings['Ks']['zone'] is True:
                self.zone_friction_minor = self.user_settings['Ks']
            else:
                self.friction_minor = self.user_settings['Ks']

    def error_message(self):
        """Error message wrapper.

        :return: Error message
        :rtype: str
        """
        err_mess_c = ctypes.POINTER(ctypes.c_char_p)()
        error = self.libmascaret.C_GET_ERREUR_MASCARET(self.id_masc,
                                                       ctypes.byref(err_mess_c))
        if error != 0:
            return 'Error could not be retrieved from MASCARET...'
        return ctypes.string_at(err_mess_c)

    def info_all_bc(self):
        """Return numbers and names of all boundary conditions. 
        
        Use Mascaret Api :meth:`C_GET_NOM_CONDITION_LIMITE_MASCARET`.

        :return:
        :rtype: float, list(float), list(float)
        """
        # Rating curve do not count
        nb_bc = ctypes.c_int()
        errors = False
        error = self.libmascaret.C_GET_NB_CONDITION_LIMITE_MASCARET(
            self.id_masc, ctypes.byref(nb_bc))
        if error != 0:
            self.logger.error("Error getting the number of boundary conditions: {}"
                              .format(self.error_message()))
            errors = True
        else:
            self.nb_bc = nb_bc.value

        l_name_all_bc = []
        l_num_all_bc = []
        for k in range(nb_bc.value):
            NomCL = ctypes.POINTER(ctypes.c_char_p)()
            NumLoi = ctypes.c_int()
            error = self.libmascaret.C_GET_NOM_CONDITION_LIMITE_MASCARET(
                self.id_masc, k + 1, ctypes.byref(NomCL), ctypes.byref(NumLoi))
            if error != 0:
                self.logger.error("Error at index {} getting the name of boundary conditions: {}"
                                  .format(k, self.error_message()))
                errors = True
            l_name_all_bc.append(ctypes.string_at(NomCL))
            l_num_all_bc.append(NumLoi.value)

        if not errors:
            self.l_name_all_bc = l_name_all_bc
            self.l_num_all_bc = l_num_all_bc
            self.logger.debug('Get BC info OK')

        return nb_bc, l_name_all_bc, l_num_all_bc

    @property
    def bc_qt(self):
        """Get boundary conditions Qt.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and :meth:`C_GET_DOUBLE_MASCARET`.

        :return: boundary conditions for Qt
        :rtype: list(float)
        """
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
        errors = False
        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            error = self.libmascaret.C_GET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))
            if error != 0:
                self.logger.error("Error at indices: ({}, {}) getting discharge: {}"
                                  .format(k, kk, self.error_message()))
                errors = True
            else:
                bc_qt[k, kk] = q_bc_c.value

        if not errors:
            self.logger.debug('Get BC Q(t) OK')

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
        """Set boundary condition Qt.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and :meth:`C_SET_DOUBLE_MASCARET`.

        :param dict q_bc: Boundary conditions on Qt ``{'idx','value'}``
        """
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

        errors = False
        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            q_bc_c.value = new_tab_q_bc[k, kk]
            error = self.libmascaret.C_SET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))
            if error != 0:
                self.logger.error("Error at indices: ({}, {}) setting discharge: {}"
                                  .format(k, kk, self.error_message()))
                errors = True

        if not errors:
            self.logger.debug('Change Q OK')

    @property
    def ind_zone_frot(self):
        """Get indices of the beginning and end of all the friction zones. 
        
        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and :meth:`C_GET_INT_MASCARET`.

        :return: Index of beginning and end
        :rtype: list(int)
        """
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()

        var_name = ctypes.c_char_p(b'Model.FrictionZone.FirstNode')
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        if error != 0:
            self.logger.error("Error getting number of friction zone at first node: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Number of Friction Zones at first node: {}'.format(size1.value))

        l_ind_beg_zone = []
        errors = False
        for k in range(size1.value):
            ind_beg_zone_c = ctypes.c_int()
            error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_beg_zone_c))
            if error != 0:
                self.logger.error("Error at index: {} getting first node friction zone: {}"
                                  .format(k, self.error_message()))
                errors = True
            else:
                l_ind_beg_zone.append(ind_beg_zone_c.value)

        var_name = ctypes.c_char_p(b'Model.FrictionZone.LastNode')
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        if error != 0:
            self.logger.error("Error getting number friction zone at last node: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Number of Friction Zones at last node: {}'.format(size1.value))

        l_ind_end_zone = []
        for k in range(size1.value):
            ind_end_zone_c = ctypes.c_int()
            error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_end_zone_c))
            if error != 0:
                self.logger.error("Error at index: {} getting last node friction zone: {}"
                                  .format(k, self.error_message()))
                errors = True
            else:
                l_ind_end_zone.append(ind_end_zone_c.value)

        if not errors:
            self.logger.debug('Get list index for all friction zones OK.')

        return l_ind_beg_zone, l_ind_end_zone

    @property
    def zone_friction_minor(self):
        """Get minor friction coefficient at zone :attr:`ind_zone`.

        Use :attr:`ind_zone_frot` and :attr:`friction_minor`.

        :return: Friction coefficient at zone
        :rtype: list(float)
        """
        l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot
        Ind_BegZone = l_ind_beg_zone[self.ind_zone]
        Ind_EndZone = l_ind_end_zone[self.ind_zone]

        zone_friction = []
        for index in range(Ind_BegZone, Ind_EndZone + 1):
            zone_friction.append(self.friction_minor)

        self.logger.debug('Get Zone KS OK')

        return zone_friction

    @zone_friction_minor.setter
    def zone_friction_minor(self, Ks):
        """Change minor friction coefficient at zone.

        Use :attr:`ind_zone_frot` and :meth:`friction_minor`.

        :param dict Ks: Friction coeffcient at zone ``{'ind_zone','value'}``
        """
        ind_zone, value = Ks['ind_zone'], Ks['value']
        l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot
        Ind_BegZone = l_ind_beg_zone[ind_zone]
        Ind_EndZone = l_ind_end_zone[ind_zone]
        self.ind_zone = ind_zone
        for index in range(Ind_BegZone, Ind_EndZone + 1):
            self.logger.debug(index)
            self.friction_minor = {'idx': index, 'value': value}

        self.logger.debug('Change Zone KS OK')

    @property
    def friction_minor(self):
        """Get minor friction coefficient at index :attr:`.ks_idx`.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and
        :meth:`C_GET_DOUBLE_MASCARET`.

        :return: Minor friction coefficient
        :rtype: float
        """
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
        """Changes minor friction coefficient.

        Use Mascaret Api :meth:`C_SET_DOUBLE_MASCARET`.

        :param dict Ks: Minor friction coefficient ``{'idx','value'}``
        """
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
        """Get state at given index in :attr:`user_settings['misc']['index_outstate']`.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and
        :meth:`C_GET_DOUBLE_MASCARET`.

        :param float index: Index to get the state from
        :return: State at a given index
        :rtype: float
        """
        var_name = ctypes.c_char_p(b'State.Z')

        itemp0 = ctypes.c_int()
        itemp1 = ctypes.c_int()
        itemp2 = ctypes.c_int()
        error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(self.id_masc, var_name, 0, ctypes.byref(itemp0),
                                                           ctypes.byref(itemp1), ctypes.byref(itemp2))
        if error != 0:
            self.logger.error("Error getting state: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('itemp {} {} {}'
                              .format(itemp0.value, itemp1.value, itemp2.value))

        Z_res_c = ctypes.c_double()
        error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, index, 0, 0, ctypes.byref(Z_res_c))
        if error != 0:
            self.logger.error("Error getting state: {}"
                              .format(self.error_message()))
        else:
            self.logger.debug('Get state OK')

        return Z_res_c

    def read_opt(self, filename='ResultatsOpthyca.opt'):
        """Read the results :file:`ResultatsOpthyca.opt`.

        :param str filename: path of the results file
        :return: Opt data
        :rtype: np.array
        """
        with open(filename, 'rb') as myfile:
            opt_data = myfile.read().decode('utf8').replace('"', '')

        opt_data = np.genfromtxt(StringIO(opt_data),
                                 delimiter=';', skip_header=14)

        return opt_data

    def empty_opt(self):
        """Hack to be able to re-launch Mascaret."""
        with open("ResultatsOpthyca.opt", 'w'):
            self.logger.debug('Cleaning results to launch a new run.')

    def plot_opt(self, xlab='Curvilinear abscissa (m)', ylab1='Water level (m)',
                 ylab2='Flow rate (m3/s)', title='Water level along the open-channel at final time'):
        """Plots results contained in the results file ``ResultatsOpthyca.opt``.

        :param str xlab: label x
        :param str ylab1: label y1
        :param str ylab2: label y2
        :param str title: Title
        """
        opt_data = self.read_opt()

        nb = int(max(opt_data[:, 2]))
        x = opt_data[-nb:-1, 3]
        level = opt_data[-nb:-1, 5]
        bathy = opt_data[-nb:-1, 4]
        flowrate = opt_data[-nb:-1, -1]

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
        fig.tight_layout()
        fig.savefig('./waterlevel.pdf', transparent = True, bbox_inches='tight')
        plt.close('all')
