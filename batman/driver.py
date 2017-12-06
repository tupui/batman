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
from concurrent import futures
import numpy as np
import sklearn.gaussian_process.kernels as kernels
from .pod import Pod
from .space import Space
from .surrogate import SurrogateModel
from .tasks import (SnapshotTask, Snapshot, SnapshotProvider)
from .uq import UQ
from .visualization import response_surface, Kiviat3D
from .functions.utils import multi_eval


class Driver(object):
    """Driver class."""

    logger = logging.getLogger(__name__)
    # Structure of the output directory
    fname_tree = {
        'snapshots': 'snapshots',
        'space': 'space.dat',
        'data': 'data.dat',
        'pod': 'surrogate/pod',
        'surrogate': 'surrogate',
        'predictions': 'predictions',
        'uq': 'uq',
        'visualization': 'visualization',
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
            pass
        self.snapshot_counter = 0

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
                           plabels=self.settings['snapshot']['io']['parameter_names'],
                           duplicate=duplicate)

        # Snapshots
        Snapshot.initialize(self.settings['snapshot']['io'])
        self.provider = SnapshotProvider(self.settings['snapshot']['provider'])

        if self.provider.is_job:
            # compute relative path to snapshot files
            data_files = []
            for files in self.settings['snapshot']['io']['filenames'].values():
                data_files = [os.path.join(self.provider['data-directory'], f)
                              for f in files]
            SnapshotTask.initialize(self.provider, data_files)

            # snapshots generation manager
            self.snapshooter = futures.ThreadPoolExecutor(
                max_workers=self.settings['snapshot']['max_workers'])

        if self.provider.is_file:
            # get the point from existing snapshot files,
            self.logger.info('Reading points from a list of snapshots files.')

            self.to_compute_points = []

            for path in self.provider:
                point = Snapshot.read_point(path)
                self.space += point
                self.to_compute_points.append(path)
        else:
            space_provider = self.settings['space']['sampling']
            if isinstance(space_provider, list):
                # a list of points is provided
                self.logger.info('Reading list of points from the settings.')
                self.to_compute_points = space_provider
                self.space += space_provider
            elif isinstance(space_provider, dict):
                # use sampling method
                distributions = space_provider['distributions']\
                    if 'distributions' in space_provider else None

                discrete = self.settings['space']['sampling']['discrete']\
                    if 'discrete' in self.settings['space']['sampling'] else None

                self.to_compute_points = self.space.sampling(space_provider['init_size'],
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
            if self.settings['surrogate']['method'] == 'pc':
                dists = self.settings['space']['sampling']['distributions']
                try:
                    dists = [eval('ot.' + dist, {'__builtins__': None},
                                  {'ot': __import__('openturns')})
                             for dist in dists]
                except (TypeError, AttributeError):
                    self.logger.error('OpenTURNS distribution unknown.')
                    raise SystemError

                settings_ = {'strategy': self.settings['surrogate']['strategy'],
                             'degree': self.settings['surrogate']['degree'],
                             'distributions': dists,
                             'n_sample': self.settings['space']['sampling']['init_size']}
            elif self.settings['surrogate']['method'] == 'evofusion':
                settings_ = {'cost_ratio': self.settings['surrogate']['cost_ratio'],
                             'grand_cost': self.settings['surrogate']['grand_cost']}
            elif self.settings['surrogate']['method'] == 'kriging':
                if 'kernel' not in self.settings['surrogate']:
                    settings_ = {}
                else:
                    kernel = self.settings['surrogate']['kernel']
                    try:
                        kernel = eval(kernel, {'__builtins__': None},
                                      kernels.__dict__)
                    except (TypeError, AttributeError):
                        self.logger.error('Scikit-Learn kernel unknown.')
                        raise SystemError
                    settings_ = {'kernel': kernel}
                if 'noise' in self.settings['surrogate']:
                    settings_.update({'noise': self.settings['surrogate']['noise']})
            else:
                settings_ = {}

            self.surrogate = SurrogateModel(self.settings['surrogate']['method'],
                                            self.settings['space']['corners'],
                                            **settings_)
            if self.settings['surrogate']['method'] == 'pc':
                self.space.empty()
                sample = self.surrogate.predictor.sample
                self.space += sample
                if not self.provider.is_file:
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
        # snapshots generation
        if self.provider.is_file:
            snapshots = points
        else:
            snapshots = []
            for point in points:
                if not self.provider.is_job:
                    # snapshots are in memory
                    path = None
                else:
                    # snapshots are on disk
                    path = os.path.join(self.fname,
                                        self.fname_tree['snapshots'],
                                        str(self.snapshot_counter))
                    self.snapshot_counter += 1

                if self.provider.is_function:
                    # create a snapshot on disk or in memory
                    snapshot = Snapshot(point, self.provider(point))
                    snapshots += [Snapshot.convert(snapshot, path=path)]
                elif self.provider.is_job:
                    # create a snapshot task
                    t = SnapshotTask(point, path)
                    snapshots += [self.snapshooter.submit(t.run)]

        # Fit the Surrogate [and POD]
        if self.pod is not None:
            if update:
                self.surrogate.space.empty()
                if self.provider.is_job:
                    for snapshot in futures.as_completed(snapshots):
                        self.pod.update(snapshot.result())
                else:
                    for snapshot in snapshots:
                        self.pod.update(snapshot)
            else:
                if self.provider.is_job:
                    _snapshots = []
                    for snapshot in futures.as_completed(snapshots):
                        _snapshots += [snapshot.result()]
                    snapshots = _snapshots
                self.pod.decompose(snapshots)

            self.data = self.pod.VS()
            points = self.pod.space
        else:
            if self.provider.is_job:
                _snapshots = []
                for snapshot in futures.as_completed(snapshots):
                    _snapshots += [snapshot.result()]
                snapshots = _snapshots

            if snapshots:
                snapshots_ = [Snapshot.convert(snapshot) for snapshot in snapshots]
                snapshots = np.vstack([snapshot.data for snapshot in snapshots_])

                if update:
                    snapshots = np.vstack([self.data, snapshots])
                    if len(snapshots) != len(self.space):  # no resampling
                        snapshots = self.data
                        for snapshot in snapshots_:
                            self.space += snapshot.point
                            if snapshot.point in self.space:
                                snapshots = np.vstack([snapshots,
                                                       snapshot.data])

                self.data = snapshots

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
        while len(self.space) < self.space.max_points_nb:
            self.logger.info("-> New iteration")
            quality, point_loo = self.surrogate.estimate_quality()
            # quality = 0.5
            # point_loo = [-1.1780625, -0.8144629629629629]
            if quality >= self.settings['space']['resampling']['q2_criteria']:
                break

            pdf = self.settings['uq']['pdf'] if 'uq' in self.settings else None
            hybrid = self.settings['space']['resampling']['hybrid']\
                if 'hybrid' in self.settings['space']['resampling'] else None

            discrete = self.settings['space']['sampling']['discrete']\
                if 'discrete' in self.settings['space']['sampling'] else None
            delta_space = self.settings['space']['resampling']['delta_space']
            method = self.settings['space']['resampling']['method']

            new_point = self.space.refine(self.surrogate,
                                          method,
                                          point_loo=point_loo,
                                          delta_space=delta_space, dists=pdf,
                                          hybrid=hybrid, discrete=discrete)

            try:
                self.sampling(new_point, update=True)
            except ValueError:
                break

            if self.settings['space']['resampling']['method'] == 'optimization':
                self.space.optimization_results()

    def write(self):
        """Write Surrogate [and POD] to disk."""
        if self.surrogate is not None:
            path = os.path.join(self.fname, self.fname_tree['surrogate'])
            try:
                os.makedirs(path)
            except OSError:
                pass
            self.surrogate.write(path)
        else:
            path = os.path.join(self.fname, self.fname_tree['space'])
            self.space.write(path)
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
        if self.surrogate is not None:
            self.surrogate.read(os.path.join(self.fname,
                                             self.fname_tree['surrogate']))
            self.space[:] = self.surrogate.space[:]
            self.data = self.surrogate.data
        else:
            path = os.path.join(self.fname, self.fname_tree['space'])
            self.space.read(path)
        if self.pod is not None:
            self.pod.read(os.path.join(self.fname, self.fname_tree['pod']))
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
        :param bool write: write a snapshot or not.
        :return: Result.
        :rtype: lst(:class:`tasks.snapshot.Snapshot`) or array_like (n_samples, n_features).
        :return: Standard deviation.
        :rtype: array_like (n_samples, n_features).
        """
        if write:
            output = os.path.join(self.fname, self.fname_tree['predictions'])
        else:
            output = None

        return self.surrogate(points, path=output)

    def uq(self):
        """Perform UQ analysis."""
        args = {}
        args['fname'] = os.path.join(self.fname, self.fname_tree['uq'])
        args['space'] = self.space
        args['indices'] = self.settings['uq']['type']
        args['plabels'] = self.settings['snapshot']['io']['parameter_names']
        args['dists'] = self.settings['uq']['pdf']
        args['nsample'] = self.settings['uq']['sample']

        if self.pod is not None:
            args['data'] = self.pod.mean_snapshot + np.dot(self.pod.U, self.data.T).T
        else:
            args['data'] = self.data

        try:
            args['test'] = self.settings['uq']['test']
        except KeyError:
            pass

        try:
            args['xdata'] = self.settings['visualization']['xdata']
            args['xlabel'] = self.settings['visualization']['xlabel']
        except KeyError:
            pass

        try:
            args['flabel'] = self.settings['visualization']['flabel']
        except KeyError:
            pass

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

            try:
                args['ticks_nbr'] = self.settings['visualization']['ticks_nbr']
            except KeyError:
                pass
            try:
                args['contours'] = self.settings['visualization']['contours']
            except KeyError:
                pass
            try:
                args['range_cbar'] = self.settings['visualization']['range_cbar']
            except KeyError:
                pass
            try:
                args['axis_disc'] = self.settings['visualization']['axis_disc']
            except KeyError:
                pass
        else:
            args['xdata'] = np.linspace(0, 1, output_len)\
                if output_len > 1 else None

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
            args['plabels'] = self.settings['snapshot']['io']['parameter_names']

        if len(self.settings['snapshot']['io']['variables']) < 2:
            try:
                args['flabel'] = self.settings['visualization']['flabel']
            except KeyError:
                args['flabel'] = self.settings['snapshot']['io']['variables'][0]

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
