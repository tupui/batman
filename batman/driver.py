# coding: utf8
"""
Driver Class
============

Defines all methods used to interact with other classes.

:Example:

::

    >> from batman import Driver
    >> driver = Driver(settings, script_path, output_path)
    >> driver.sampling_pod(update=False)
    >> driver.write_pod()
    >> driver.prediction(write=True)
    >> driver.write_model()
    >> driver.uq()
    >> driver.visualization()

"""
import logging
import os
import pickle
from copy import copy
from concurrent import futures
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from .pod import Pod
from .space import (Space, dists_to_ot)
from .surrogate import SurrogateModel
from .tasks import (SnapshotIO, ProviderPlugin, ProviderFile)
from .uq import UQ
from .visualization import (response_surface, Kiviat3D)
from .functions.utils import multi_eval


class Driver(object):
    """Driver class."""

    logger = logging.getLogger(__name__)
    # Structure of the output directory
    fname_tree = {
        'snapshots': 'snapshots',
        'space': 'space',
        'data': 'data.dat',
        'pod': 'surrogate/pod',
        'surrogate': 'surrogate',
        'predictions': 'predictions',
        'uq': 'uq',
        'visualization': 'visualization',
    }
    # Data provider for snapshots
    provider_class = {
        'plugin': ProviderPlugin,
        'file': ProviderFile,
    }

    def __init__(self, settings, fname):
        """Initialize Driver.

        From settings, init snapshot, space [POD, surrogate].

        :param dict settings: settings.

        :param str fname: output folder path.
        """
        self.settings = settings
        self.fname = fname
        try:
            os.makedirs(self.fname)
        except OSError:
            pass  # directory exists already

        # Space
        if 'resampling' in self.settings['space']:
            resamp_size = self.settings['space']['resampling']['resamp_size']
        else:
            resamp_size = 0
        if 'init_size' in self.settings['space']['sampling']:
            init_size = self.settings['space']['sampling']['init_size']
        else:  # when providing DoE as a list
            init_size = self.settings['space']['sampling']
        try:
            duplicate = self.settings['space']['sampling']['method'] == 'saltelli'
        except (KeyError, TypeError):
            duplicate = False

        self.space = Space(self.settings['space']['corners'],
                           init_size,
                           nrefine=resamp_size,
                           plabels=self.settings['snapshot']['plabels'],
                           duplicate=duplicate)

        # Asynchronous job manager
        self.async_pool = futures.ThreadPoolExecutor(
            max_workers=self.settings['snapshot']['max_workers'])

        # Snapshot Management
        args = settings['snapshot'].get('io', {})
        self.snapshot_io = SnapshotIO(parameter_names=settings['snapshot']['plabels'],
                                      feature_names=settings['snapshot']['flabels'],
                                      **args)
        provider_type = settings['snapshot']['provider']['type']
        self.logger.info('Select data provider type "{}"'.format(provider_type))
        self.provider = self.provider_class[provider_type](
            self.async_pool,
            self.snapshot_io,
            settings['snapshot']['provider'])
        self.snapshot_counter = 0

        # Sampling initialisation
        self.to_compute_points = copy(self.provider.known_points)
        if self.to_compute_points:
            # use points that were automatically discovered by the provider
            for point in self.to_compute_points:
                self.space += point
            self.to_compute_points = list(self.space)
        else:
            # generate points according to settings
            space_provider = self.settings['space']['sampling']
            if isinstance(space_provider, list):
                # a list of points is provided
                self.logger.info('Reading list of points from the settings.')
                self.space += space_provider
                self.to_compute_points = list(self.space)
            elif isinstance(space_provider, dict):
                # use sampling method
                distributions = space_provider.get('distributions')
                discrete = self.settings['space']['sampling'].get('discrete')
                self.to_compute_points = self.space.sampling(
                    space_provider['init_size'],
                    space_provider['method'],
                    distributions,
                    discrete)
            else:
                self.logger.error('Bad space provider.')
                raise SystemError

        # Pod
        if 'pod' in self.settings:
            settings_ = {'tolerance': self.settings['pod']['tolerance'],
                         'dim_max': self.settings['pod']['dim_max'],
                         'corners': self.settings['space']['corners'],
                         'nsample': self.space.doe_init,
                         'nrefine': resamp_size}
            self.pod = Pod(**settings_)
            self.pod.space.duplicate = duplicate
        else:
            self.pod = None
            self.logger.info('No POD is computed.')

        self.data = None

        # Surrogate model
        if 'surrogate' in self.settings:
            settings_ = {}
            if self.settings['surrogate']['method'] == 'pc':
                dists = self.settings['space']['sampling']['distributions']
                dists = dists_to_ot(dists)

                settings_ = {'strategy': self.settings['surrogate']['strategy'],
                             'degree': self.settings['surrogate']['degree'],
                             'distributions': dists,
                             'sample': self.space[:]}
            elif self.settings['surrogate']['method'] == 'evofusion':
                settings_ = {'cost_ratio': self.settings['surrogate']['cost_ratio'],
                             'grand_cost': self.settings['surrogate']['grand_cost']}
            elif self.settings['surrogate']['method'] == 'kriging':
                if 'kernel' in self.settings['surrogate']:
                    kernel = self.settings['surrogate']['kernel']
                    try:
                        kernel = eval(kernel, {'__builtins__': None},
                                      kernels.__dict__)
                    except (TypeError, AttributeError):
                        self.logger.error('Scikit-Learn kernel unknown.')
                        raise SystemError
                    settings_ = {'kernel': kernel}

                settings_.update({
                    'noise': self.settings['surrogate'].get('noise', False),
                    'global_optimizer': self.settings['surrogate'].get('global_optimizer', True)
                })

            self.surrogate = SurrogateModel(self.settings['surrogate']['method'],
                                            self.settings['space']['corners'],
                                            **settings_)
            if self.settings['surrogate']['method'] == 'pc':
                self.space.empty()
                sample = self.surrogate.predictor.sample
                self.space += sample
                if not self.provider.known_points:
                    self.to_compute_points = sample[:len(self.space)]
        else:
            self.surrogate = None
            self.logger.info('No surrogate is computed.')

    def sampling(self, points=None, update=False):
        """Create snapshots.

        Generates or retrieve the snapshots [and then perform the POD].

        :param :class:`Space` points: points to perform the sample from.
        :param bool update: perform dynamic or static computation.
        """
        if points is None:
            points = self.to_compute_points

        # Generate snapshots
        if isinstance(points, dict):
            snapshot_points = points.items()
        else:
            snapshot_root = os.path.join(self.fname, self.fname_tree['snapshots'])
            snapshot_points = [(point,
                                os.path.join(snapshot_root,
                                             str(i + self.snapshot_counter)))
                               for i, point in enumerate(points)]
        snapshots = [self.provider.snapshot(p, d) for p, d in snapshot_points]
        self.snapshot_counter += len(snapshots)

        # Fit the Surrogate [and POD]
        if self.pod is not None:
            if update:
                self.surrogate.space.empty()
                [self.pod.update(snapshot) for snapshot in snapshots]
            else:
                self.pod.decompose(snapshots)
            self.data = self.pod.VS()
            points = self.pod.space

        else:
            if snapshots:
                snapdata = np.vstack([snap.data for snap in snapshots])
                if update:
                    snapdata = np.vstack([self.data, snapdata])
                    if len(snapdata) != len(self.space):  # no resampling
                        snapdata = self.data
                        for snapshot in snapshots:
                            self.space += snapshot.point
                            snapdata = np.vstack([snapdata, snapshot.data])
                self.data = snapdata
            points = self.space

        try:  # if surrogate
            self.surrogate.fit(points, self.data, pod=self.pod)
        except AttributeError:
            pass

    def resampling(self):
        """Resampling of the parameter space.

        Generate new samples if quality and number of sample are not satisfied.
        From a new sample, it re-generates the POD.

        """
        self.logger.info("\n----- Resampling parameter space -----")
        method = self.settings['space']['resampling']['method']
        extremum = self.settings['space']['resampling'].get('extremum')
        hybrid = self.settings['space']['resampling'].get('hybrid')
        discrete = self.settings['space']['sampling'].get('discrete')
        delta_space = self.settings['space']['resampling'].get('delta_space', 0.08)
        q2_criteria = self.settings['space']['resampling'].get('q2_criteria')
        pdf = self.settings.get('uq', {}).get('pdf')

        while len(self.space) < self.space.max_points_nb:
            self.logger.info("-> New iteration")

            if (method != 'optimization') and (q2_criteria is not None):
                quality, point_loo = self.surrogate.estimate_quality()
                if quality >= q2_criteria:
                    break
            else:
                point_loo = None

            new_point = self.space.refine(self.surrogate,
                                          method,
                                          point_loo=point_loo,
                                          delta_space=delta_space, dists=pdf,
                                          hybrid=hybrid, discrete=discrete,
                                          extremum=extremum)

            try:
                self.sampling(new_point, update=True)
            except ValueError:
                break

            if method == 'optimization':
                self.space.optimization_results(extremum=extremum)

    def write(self):
        """Write DOE, Surrogate [and POD] to disk."""
        path = os.path.join(self.fname, self.fname_tree['space'])
        try:
            os.makedirs(path)
        except OSError:
            pass
        self.space.write(path)
        if self.surrogate is not None:
            path = os.path.join(self.fname, self.fname_tree['surrogate'])
            try:
                os.makedirs(path)
            except OSError:
                pass
            self.surrogate.write(path)
        if self.pod is not None:
            path = os.path.join(self.fname, self.fname_tree['pod'])
            try:
                os.makedirs(path)
            except OSError:
                pass
            self.pod.write(path)
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.fname, self.fname_tree['data'])
            with open(path, 'wb') as fdata:
                pickler = pickle.Pickler(fdata)
                pickler.dump(self.data)
            self.logger.debug('Wrote data to {}'.format(path))

    def read(self):
        """Read Surrogate [and POD] from disk."""
        path = os.path.join(self.fname, self.fname_tree['space'])
        self.space.read(os.path.join(path, 'space.dat'))
        if self.surrogate is not None:
            self.surrogate.read(os.path.join(self.fname,
                                             self.fname_tree['surrogate']))
            self.data = self.surrogate.data
        if self.pod is not None:
            self.pod.read(os.path.join(self.fname, self.fname_tree['pod']))
            self.pod.space = self.space
            self.surrogate.pod = self.pod
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.fname, self.fname_tree['data'])
            with open(path, 'rb') as fdata:
                unpickler = pickle.Unpickler(fdata)
                self.data = unpickler.load()
            self.logger.debug('Data read from {}'.format(path))

    def restart(self):
        """Restart process."""
        # Surrogate [and POD] has already been computed
        self.logger.info('Restarting from previous computation...')
        to_compute_points = self.space[:]
        self.read()  # Reset space with actual computations
        self.snapshot_counter = len(self.space)

        if self.snapshot_counter < len(to_compute_points):
            # will add new points to be processed
            # [static or dynamic pod is finished]
            self.to_compute_points = [p for p in to_compute_points
                                      if p not in self.space]
        else:
            # automatic resampling has to continue from
            # the processed points
            self.to_compute_points = []

    def prediction(self, points, write=False):
        """Perform a prediction.

        :param points: point(s) to predict.
        :type points: :class:`space.point.Point` or array_like (n_samples, n_features).
        :param bool write: whether to write snapshots.
        :return: Result.
        :rtype: array_like (n_samples, n_features).
        :return: Standard deviation.
        :rtype: array_like (n_samples, n_features).
        """
        results, sigma = self.surrogate(points)
        if write:
            root_path = os.path.join(self.fname, self.fname_tree['predictions'])
            try:
                points[0][0]
            except TypeError:
                points = [points]
            for i, (data, point) in enumerate(zip(results, points)):
                path = os.path.join(root_path, 'Newsnap{}'.format(i))
                try:
                    os.makedirs(path)
                except OSError:
                    pass
                self.snapshot_io.write_point(path, point)
                self.snapshot_io.write_data(path, data)
        return results, sigma

    def uq(self):
        """Perform UQ analysis."""
        args = {}
        args['fname'] = os.path.join(self.fname, self.fname_tree['uq'])
        args['space'] = self.space
        args['indices'] = self.settings['uq']['type']
        args['plabels'] = self.settings['snapshot']['plabels']
        args['dists'] = self.settings['uq']['pdf']
        args['nsample'] = self.settings['uq']['sample']

        if self.pod is not None:
            args['data'] = self.pod.mean_snapshot + np.dot(self.pod.U, self.data.T).T
        else:
            args['data'] = self.data

        args['test'] = self.settings['uq'].get('test')
        args['xdata'] = self.settings.get('visualization', {}).get('xdata')
        args['xlabel'] = self.settings.get('visualization', {}).get('xlabel')
        args['flabel'] = self.settings.get('visualization', {}).get('flabel')

        analyse = UQ(self.surrogate, **args)

        if self.surrogate is None:
            self.logger.warning("No surrogate model, be sure to have a "
                                "statistically significant sample to trust "
                                "following results.")
        analyse.sobol()
        analyse.error_propagation()

    def visualization(self):
        """Apply visualisation options."""
        p_len = len(self.settings['space']['corners'][0])

        # In case of POD, data need to be converted from modes to snapshots.
        if self.pod is not None:
            data = self.pod.mean_snapshot + np.dot(self.pod.U, self.data.T).T
        else:
            data = self.data

        output_len = np.asarray(data).shape[1]

        self.logger.info('Creating response surface...')
        args = {}
        if 'visualization' in self.settings:
            # xdata for output with dim > 1
            if ('xdata' in self.settings['visualization']) and (output_len > 1):
                args['xdata'] = self.settings['visualization']['xdata']
            elif output_len > 1:
                args['xdata'] = np.linspace(0, 1, output_len)

            # Plot Doe if doe option is True
            if ('doe' in self.settings['visualization']) and\
                    self.settings['visualization']['doe']:
                args['doe'] = self.space

            # Display resampling if resampling option is true
            if ('resampling' in self.settings['visualization']) and\
                    self.settings['visualization']['resampling']:
                args['resampling'] = self.settings['space']['resampling']['resamp_size']
            else:
                args['resampling'] = 0

            args['ticks_nbr'] = self.settings.get('visualization', {}).get('ticks_nbr', 10)
            args['contours'] = self.settings.get('visualization', {}).get('contours')
            args['range_cbar'] = self.settings.get('visualization', {}).get('range_cbar')
            args['axis_disc'] = self.settings.get('visualization', {}).get('axis_disc')
        else:
            args['xdata'] = np.linspace(0, 1, output_len) if output_len > 1 else None

        try:
            args['bounds'] = self.settings['visualization']['bounds']
            for i, _ in enumerate(args['bounds'][0]):
                if (args['bounds'][0][i] < self.settings['space']['corners'][0][i])\
                        or (args['bounds'][1][i] > self.settings['space']['corners'][1][i]):
                    args['bounds'] = self.settings['space']['corners']
                    self.logger.warning("Specified bounds for visualisation are "
                                        "wider than space corners. Default value used.")
        except KeyError:
            args['bounds'] = self.settings['space']['corners']

        # Data based on surrogate model (function) or not
        if 'surrogate' in self.settings:
            args['fun'] = self.func
        else:
            args['sample'] = self.space
            args['data'] = data

        try:
            args['plabels'] = self.settings['visualization']['plabels']
        except KeyError:
            args['plabels'] = self.settings['snapshot']['plabels']

        if len(self.settings['snapshot']['flabels']) < 2:
            try:
                args['flabel'] = self.settings['visualization']['flabel']
            except KeyError:
                args['flabel'] = self.settings['snapshot']['flabels'][0]

        path = os.path.join(self.fname, self.fname_tree['visualization'])
        try:
            os.makedirs(path)
        except OSError:
            pass

        if p_len < 5:
            # Creation of the response surface(s)
            args['fname'] = os.path.join(path, 'Response_Surface')
            response_surface(**args)

        else:
            # Creation of the Kiviat image
            args['fname'] = os.path.join(path, 'Kiviat.pdf')
            args['sample'] = self.space
            args['data'] = data
            if 'range_cbar' not in args:
                args['range_cbar'] = None
            if 'ticks_nbr' not in args:
                args['ticks_nbr'] = 10
            if 'kiviat_fill' not in args:
                args['kiviat_fill'] = True
            kiviat = Kiviat3D(args['sample'], args['bounds'], args['data'],
                              plabels=args['plabels'],
                              range_cbar=args['range_cbar'])
            kiviat.plot(fname=args['fname'], flabel=args['flabel'],
                        ticks_nbr=args['ticks_nbr'], fill=args['kiviat_fill'])

            # Creation of the Kiviat movie:
            args['fname'] = os.path.join(path, 'Kiviat.mp4')
            rate = 400
            kiviat.f_hops(frame_rate=rate, fname=args['fname'],
                          flabel=args['flabel'], fill=args['kiviat_fill'],
                          ticks_nbr=args['ticks_nbr'])

    @multi_eval
    def func(self, coords):
        """Evaluate the surrogate at a given point.

        This function calls the surrogate to compute a prediction.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float.
        """
        f_eval, _ = self.surrogate(coords)
        return f_eval[0]
