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
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib.patches import Polygon
from ...utils import multi_eval

logging.basicConfig(level=logging.DEBUG)


class MascaretApi(object):

    """Mascaret API."""

    logger = logging.getLogger(__name__)

    def __setattr__(self, name, value):
        """Detect errors.

        Overwright attribute setter to detect API errors.
        If :attr:`error` is not set null, an error is raised and the programme
        is terminated.

        :param str name: name of the attribute
        :param ... value: value to assign
        """
        object.__setattr__(self, name, value)
        if (name is 'error') and (value is not 0):
            self.logger.error("API error:\n{}".format(self.error_message()))
            raise SystemExit

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
        self.logger.error("Getting size var in model  #{}..."
                          .format(self.id_masc.value))
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(self.id_masc, var_name, 0,
                                                           ctypes.byref(nb_nodes),
                                                           ctypes.byref(il_temp1),
                                                           ctypes.byref(il_temp2))
        self.logger.debug('Get nb of nodes: size_X={}.'
                              .format(nb_nodes.value))
        return nb_nodes

    def init_model(self):
        """Initialize the model from constant values.

        ``init_cst`` in :attr:`user_settings` along with ``Q_cst`` and
        ``Z_cst`` values or from :file:`file.lig` in :attr:`settings`. Uses
        Mascaret Api :meth:`C_INIT_LIGNE_MASCARET` or
        :meth:`C_INIT_ETAT_MASCARET`.
        """
        if 'init_cst' in self.user_settings:
            # Initialize Mascaret Model from values
            q = self.user_settings['init_cst']['Q_cst'] * self.nb_nodes.value
            z = self.user_settings['init_cst']['Z_cst'] * self.nb_nodes.value
            q_c = (ctypes.c_double * self.nb_nodes.value)(q)
            z_c = (ctypes.c_double * self.nb_nodes.value)(z)
            self.logger.debug('Initilizing MASCARET from constant value...')
            self.error = self.libmascaret.C_INIT_LIGNE_MASCARET(
                self.id_masc, ctypes.byref(q_c), ctypes.byref(z_c), self.nb_nodes)
            self.logger.debug(
                'State constant initialisation successfull from constant value.')
        else:
            # Initialize Mascaret Model from file
            init_file_name_c = (ctypes.c_char_p)(*[self.settings['files']['lig']])
            self.logger.debug('Initializing MASCARET from lig...')
            self.error = self.libmascaret.C_INIT_ETAT_MASCARET(
                self.id_masc, init_file_name_c, self.iprint)
            self.logger.debug('State initialisation successfull from lig.')

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
        self.logger.debug('Getting time step...')
        self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.dt))
        self.logger.debug('Time step: dt={}.'.format(self.dt.value))
        # t0
        self.t0 = ctypes.c_double()
        var_name = ctypes.c_char_p(b'Model.InitTime')
        self.logger.debug('Getting initial time...')
        self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.t0))
        self.logger.debug('Initial time: t0={}.'.format(self.t0.value))
        # tend
        self.tend = ctypes.c_double()
        var_name = ctypes.c_char_p(b'Model.MaxCompTime')
        self.logger.debug('Getting end time...')
        self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, 0, 0, 0, ctypes.byref(self.tend))
        self.logger.debug('End time: tend={}.'.format(self.tend.value))

        return (self.dt.value, self.t0.value, self.tend.value)

    def create_model(self):
        """Create an instance of Mascaret.

        Uses Mascaret Api :meth:`C_CREATE_MASCARET`.
        """
        id_masc = ctypes.c_int()
        self.logger.debug('Creating a model...')
        self.error = self.libmascaret.C_CREATE_MASCARET(ctypes.byref(id_masc))
        self.logger.debug('Model created.')
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
        len_file = len(file_name)
        file_name_c = (ctypes.c_char_p * len_file)(*file_name)
        file_type_c = (ctypes.c_char_p * len_file)(*file_type)
        self.logger.debug('Importing a model...')
        self.error = self.libmascaret.C_IMPORT_MODELE_MASCARET(self.id_masc, file_name_c,
                                                          file_type_c, len_file, self.iprint)
        self.logger.info("Model imported with:\n-> file_name: {}\n-> file_type: {}."
                             .format(file_name, file_type))

    def __del__(self):
        """Delete a model."""
        self.logger.debug('Deleting instance #{}...'.format(self.id_masc.value))
        self.error = self.libmascaret.C_DELETE_MASCARET(self.id_masc)
        self.logger.debug("Model #{} deleted.".format(self.id_masc.value))

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

    @multi_eval
    def run_mascaret(self, x=None, flag=None, saveall=False):
        """Run Mascaret simulation.

        Use Mascaret Api :meth:`C_CALCUL_MASCARET`.
        If :arg:`x` if not None, ``Ks`` and ``Q`` are modified before running.
        When :attr:`flag` is None, both parameters are modified. Thus :arg:`x`
        needs to be set accordingly. If the flag is set to ``Ks``, then only
        this parameter is considered.

        :param list x: inputs [Ks, Q]
        :param str flag: None, 'Ks' or 'Q'
        :param bool saveall: Change the default name of the Results file
        :return: water level at :attr:`index_outstate`
        :rtype: double
        """
        if x is not None:
            if (flag is None) or (flag is 'Ks'):
                if self.user_settings['Ks']['zone']:
                    self.zone_friction_minor = {
                        'ind_zone': self.user_settings['Ks']['ind_zone'],
                            'value': x[0]}
                else:
                    self.friction_minor = {'idx': self.user_settings['Ks']['idx'],
                                           'value': x[0]}
            elif flag is 'Q':
                self.bc_qt = {'idx': self.user_settings['Q_BC']['idx'],
                              'value': x[0]}

            if flag is None:
                self.bc_qt = {'idx': self.user_settings['Q_BC']['idx'],
                              'value': x[1]}

        self.empty_opt()
        self.logger.info('Running Mascaret...')
        self.error = self.libmascaret.C_CALCUL_MASCARET(self.id_masc, self.t0,
                                                   self.tend, self.dt, self.iprint)
        self.logger.info('Mascaret ran.')

        if saveall:
            os.rename('ResultatsOpthyca.opt',
                      'ResultatsOpthyca_' + str(x).replace(' ', '-') + '.opt')

        return self.state(self.user_settings['misc']['index_outstate']).value

    def __call__(self, x=None, saveall=False):
        """Run the application using :attr:`user_settings`.

        :param list x: inputs [Ks, Q]
        :param bool saveall: Change the default name of the Results file
        """
        if ('MC' in self.user_settings) and (x is None):
            self.logger.info('Performing a Monte-Carlo analysis...')
            try:
                n = self.user_settings['MC']['Ne']
            except KeyError:
                n = 1

            nx = ('distKs' in self.user_settings['MC']) + ('distQ' in self.user_settings['MC'])
            self.doe = np.empty((self.user_settings['MC']['Ne'], nx))

            if 'distKs' in self.user_settings['MC']:
                ks = self.doe[:, 0]
                flag = 'Ks'
                if self.user_settings['MC']['distKs'] == "G":
                    ks[:] = np.random.normal(
                        self.user_settings['MC']['muKs'], self.user_settings['MC']['sigmaKs'], n)
                elif self.user_settings['MC']['distKs'] == "U":
                    ks[:] = np.random.uniform(
                        self.user_settings['MC']['minKs'], self.user_settings['MC']['maxKs'], n)

            if 'distQ' in self.user_settings['MC']:
                if 'distKs' in self.user_settings['MC']:
                    q = self.doe[:, 1]
                    flag = None
                else:
                    q = self.doe[:, 0]
                    flag='Q'
                if self.user_settings['MC']['distQ'] == "G":
                    q[:] = np.random.normal(
                        self.user_settings['MC']['muQ'], self.user_settings['MC']['sigmaQ'], n)
                elif self.user_settings['MC']['distQ'] == "U":
                    q[:] = np.random.uniform(
                        self.user_settings['MC']['minQ'], self.user_settings['MC']['maxQ'], n)

            self.logger.debug('Design of Experiment:\n{}'.format(self.doe))

            h = self.run_mascaret(self.doe, flag=flag, saveall=True)

        else:
            self.logger.info('Performing a single MASCARET simulation...')
            h = self.run_mascaret(x=x, saveall=saveall)

        self.results = h

        return h

    def user_defined(self, user_settings):
        """Read user parameters from :file:`user_settings`` and apply values.

        Look for ``Q_BC`` (``Q_BC={'idx','value'}``) and ``Ks``
        (``Ks={'zone','idx','value', 'ind_zone'}``) (the ``Ks`` for 1 point or
        1 zone). Use :meth:`zone_friction_minor`, :meth:`friction_minor` and
        :meth:`bc_qt`.

        :param str user_settings: Path of the *JSON* settings file
        """
        with open(user_settings, 'rb') as file:
            file = file.read().decode('utf8')
            self.user_settings = json.loads(
                file, encoding="utf-8", object_pairs_hook=OrderedDict)
        if 'Q_BC' in self.user_settings:
            self.bc_qt = self.user_settings['Q_BC']
        if 'Ks' in self.user_settings:
            if self.user_settings['Ks']['zone']:
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
        self.logger.debug('Getting the number of boundary conditions...')
        self.error = self.libmascaret.C_GET_NB_CONDITION_LIMITE_MASCARET(
            self.id_masc, ctypes.byref(nb_bc))
        self.nb_bc = nb_bc.value
        self.logger.debug('Number of boundary conditions: {}.'.format(self.nb_bc))

        l_name_all_bc = []
        l_num_all_bc = []
        self.logger.debug('Getting name of the boundary conditions...')
        for k in range(nb_bc.value):
            name_all_bc = ctypes.POINTER(ctypes.c_char_p)()
            n_law = ctypes.c_int()
            self.error = self.libmascaret.C_GET_NOM_CONDITION_LIMITE_MASCARET(
                self.id_masc, k + 1, ctypes.byref(name_all_bc), ctypes.byref(n_law))
            l_name_all_bc.append(ctypes.string_at(name_all_bc))
            l_num_all_bc.append(n_law.value)

        self.l_name_all_bc = l_name_all_bc
        self.l_num_all_bc = l_num_all_bc
        self.logger.debug('BC info get.')

        return nb_bc, l_name_all_bc, l_num_all_bc

    @property
    def bc_qt(self):
        """Get boundary conditions Qt.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET`
        and :meth:`C_GET_DOUBLE_MASCARET`.

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
        self.logger.debug('Getting the size of Model.Graph.Discharge...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Size Model.Graph.Discharge= {} {} {}.'
                          .format(size1.value, size2.value, size3.value))

        bc_qt = np.ones((size1.value, size2.value), float)
        self.logger.debug('Getting discharge values...')
        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))
            bc_qt[k, kk] = q_bc_c.value

        self.logger.debug('BC Q(t) get.')

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

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET`
        and :meth:`C_SET_DOUBLE_MASCARET`.

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
        self.logger.debug('Getting the size of Model.Graph.Discharge...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Size Model.Graph.Discharge= {} {} {}.'
                          .format(size1.value, size2.value, size3.value))

        self.logger.debug('Getting discharge values...')
        for k, kk in itertools.product(range(size1.value), range(size2.value)):
            q_bc_c = ctypes.c_double()
            num_bc_c = ctypes.c_int(k + 1)
            indextime_bc_c = ctypes.c_int(kk + 1)
            q_bc_c.value = new_tab_q_bc[k, kk]
            self.error = self.libmascaret.C_SET_DOUBLE_MASCARET(
                self.id_masc, var_name, num_bc_c, indextime_bc_c, 0, ctypes.byref(q_bc_c))

        self.logger.debug('BC Q(t) set.')

    @property
    def ind_zone_frot(self):
        """Get indices of the beginning and end of all the friction zones.

        Use Mascaret Api :meth:`C_GET_TAILLE_VAR_MASCARET` and
        :meth:`C_GET_INT_MASCARET`.

        :return: Index of beginning and end
        :rtype: list(int)
        """
        size1 = ctypes.c_int()
        size2 = ctypes.c_int()
        size3 = ctypes.c_int()

        var_name = ctypes.c_char_p(b'Model.FrictionZone.FirstNode')
        self.logger.debug('Getting the size of Model.FrictionZone.FirstNode...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Number of Friction Zones at first node: {}.'.format(size1.value))

        l_ind_beg_zone = []
        self.logger.debug('Getting friction indices at the beginning...')
        for k in range(size1.value):
            ind_beg_zone_c = ctypes.c_int()
            self.error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_beg_zone_c))
            l_ind_beg_zone.append(ind_beg_zone_c.value)
        self.logger.debug('Friction indices at the beginning get.')

        var_name = ctypes.c_char_p(b'Model.FrictionZone.LastNode')
        self.logger.debug('Getting the size of Model.FrictionZone.LastNode...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1), ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Number of Friction Zones at last node: {}.'.format(size1.value))

        l_ind_end_zone = []
        self.logger.debug('Getting friction indices at the end...')
        for k in range(size1.value):
            ind_end_zone_c = ctypes.c_int()
            self.error = self.libmascaret.C_GET_INT_MASCARET(
                self.id_masc, var_name, k + 1, 0, 0, ctypes.byref(ind_end_zone_c))
            l_ind_end_zone.append(ind_end_zone_c.value)
        self.logger.debug('Friction indices at the end get.')

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
        idx_beg_zone = l_ind_beg_zone[self.ind_zone]
        idx_end_zone = l_ind_end_zone[self.ind_zone]

        zone_friction = []
        for index in range(idx_beg_zone, idx_end_zone + 1):
            self.ks_idx = index
            zone_friction.append(self.friction_minor)

        self.logger.debug('Zone Ks get.')

        return zone_friction

    @zone_friction_minor.setter
    def zone_friction_minor(self, ks):
        """Change minor friction coefficient at zone.

        Use :attr:`ind_zone_frot` and :meth:`friction_minor`.

        :param dict ks: Friction coeffcient at zone ``{'ind_zone','value'}``
        """
        ind_zone, value = ks['ind_zone'], ks['value']
        l_ind_beg_zone, l_ind_end_zone = self.ind_zone_frot
        idx_beg_zone = l_ind_beg_zone[ind_zone]
        idx_end_zone = l_ind_end_zone[ind_zone]
        self.ind_zone = ind_zone
        for index in range(idx_beg_zone, idx_end_zone + 1):
            self.logger.debug(index)
            self.friction_minor = {'idx': index, 'value': value}

        self.logger.debug('Zone Ks set.')

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
        self.logger.debug('Getting the size of Model.FricCoefMainCh...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(size1),
            ctypes.byref(size2), ctypes.byref(size3))
        self.logger.debug('Size of Model.FricCoefMinCh = {} {} {}.'
                          .format(size1.value, size2.value, size3.value))
        ks_c = ctypes.c_double()
        self.logger.debug('Getting friction values.')
        self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, ctypes.byref(ks_c))
        self.logger.debug('Ks value= {}.'.format(ks_c.value))

        return ks_c.value

    @friction_minor.setter
    def friction_minor(self, ks):
        """Changes minor friction coefficient.

        Use Mascaret Api :meth:`C_SET_DOUBLE_MASCARET`.

        :param dict ks: Minor friction coefficient ``{'idx','value'}``
        """
        var_name = ctypes.c_char_p(b'Model.FricCoefMainCh')
        ks_c = ctypes.c_double(ks['value'])
        self.logger.debug('ks_c = {}'.format(ks_c.value))
        self.logger.debug('Ks new value= {}'.format(ks))
        self.ks_idx = ks['idx']
        self.logger.debug('Setting Ks value...')
        self.error = self.libmascaret.C_SET_DOUBLE_MASCARET(
            self.id_masc, var_name, self.ks_idx, 0, 0, ks_c)
        self.logger.debug('Ks changed.')

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
        self.logger.debug('Getting the size of State.Z...')
        self.error = self.libmascaret.C_GET_TAILLE_VAR_MASCARET(
            self.id_masc, var_name, 0, ctypes.byref(itemp0), ctypes.byref(itemp1), ctypes.byref(itemp2))
        self.logger.debug('itemp= {} {} {}.'
                          .format(itemp0.value, itemp1.value, itemp2.value))

        z_res_c = ctypes.c_double()
        self.logger.debug('Getting the value of State.Z...')
        self.error = self.libmascaret.C_GET_DOUBLE_MASCARET(
            self.id_masc, var_name, index, 0, 0, ctypes.byref(z_res_c))
        self.logger.debug('State get.')

        return z_res_c

    def read_opt(self, filename='ResultatsOpthyca.opt'):
        """Read the results :file:`ResultatsOpthyca.opt`.

        :param str filename: path of the results file
        :return: Opt data
        :rtype: np.array
        """
        with open(filename, 'rb') as myfile:
            opt_data = myfile.read().decode('utf8').replace('"', '')

        opt_data = np.genfromtxt(BytesIO(opt_data.encode('utf8')),
                                 delimiter=';', skip_header=14)

        return opt_data

    def empty_opt(self):
        """Hack to be able to re-launch Mascaret."""
        with open("ResultatsOpthyca.opt", 'w'):
            self.logger.debug('Cleaning results to launch a new run.')

    def plot_opt(self, xlab='Curvilinear abscissa (m)', ylab1='Water level (m)',
                 ylab2='Flow rate (m3/s)', title='Water level along the open-channel at final time'):
        """Plots results contained in the results file :file:`ResultatsOpthyca.opt`.

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
        fig.savefig('./waterlevel.pdf', transparent=True, bbox_inches='tight')
        plt.close('all')
