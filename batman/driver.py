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

"""
import logging
import os
import numpy as np
import pickle
from concurrent import futures
import openturns as ot
from collections import OrderedDict
from .pod import Pod
from .space import (Space, FullSpaceError, AlienPointError, UnicityError)
from .surrogate import (SurrogateModel, PC)
from .tasks import (SnapshotTask, Snapshot, SnapshotProvider)
from .uq import UQ


class Driver(object):

    """Driver class."""

    logger = logging.getLogger(__name__)
    output_tree = {
        'snapshots': 'snapshots',
        'space': 'space.dat',
        'data': 'data.dat',
        'pod': 'surrogate/pod',
        'surrogate': 'surrogate',
        'predictions': 'predictions',
        'uq': 'uq',
    }
    '''Structure of the output directory.'''

    def __init__(self, settings, output):
        """Initialize Driver.

        From settings, init snapshot, space and POD.

        :param dict settings: settings.
        :param str script: settings path.
        :param str output: output path.
        """
        self.settings = settings
        self.output = output
        try:
            os.makedirs(self.output)
        except OSError:
            pass
        self.snapshot_counter = 0

        # Space
        self.space = Space(self.settings)

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

            self.to_compute_points = OrderedDict()

            for path in self.provider:
                point = Snapshot.read_point(path)
                try:
                    self.space += point
                except (AlienPointError, UnicityError, FullSpaceError) as tb:
                    self.logger.warning("Ignoring: {}".format(tb))
                else:
                    self.to_compute_points[point] = path
        else:
            space_provider = self.settings['space']['sampling']
            if isinstance(space_provider, list):
                # a list of points is provided
                self.logger.info('Reading list of points from the settings.')
                self.to_compute_points = space_provider
                self.space += space_provider
            elif isinstance(space_provider, dict):
                # use sampling method
                self.to_compute_points = self.space.sampling(space_provider['init_size'])
            else:
                self.logger.error('Bad space provider.')
                raise SystemError

        # Pod
        try:
            self.pod = Pod(self.settings)
        except KeyError:
            self.pod = None
            self.logger.info('No POD is computed.')

        # Surrogate model
        try:
            if self.settings['surrogate']['method'] == 'pc':
                dists = self.settings['space']['sampling']['method']
                dists = [eval("ot." + dist) for dist in dists]
                settings_ = {'strategy': self.settings['surrogate']['strategy'],
                             'degree': self.settings['surrogate']['degree'],
                             'distributions': dists,
                             'n_sample': self.settings['space']['sampling']['init_size']}
                pc = PC(**settings_)
                self.space.empty()
                try:
                    self.space += pc.sample
                except (AlienPointError, UnicityError, FullSpaceError) as tb:
                    self.logger.warning("Ignoring: {}".format(tb))
                finally:
                    self.to_compute_points = pc.sample[:len(self.space)]
            else:
                settings_ = {}

            self.surrogate = SurrogateModel(self.settings['surrogate']['method'],
                                            self.settings['space']['corners'],
                                            **settings_)
        except KeyError:
            self.surrogate = None
            self.logger.info('No surrogate is computed.')

    def sampling(self, points=None, update=False):
        """Create snapshots.

        Generates or retrieve the snapshots [and then perform the POD].

        :param :class:`Space` points: points to perform the sample from
        :param bool update: perform dynamic or static computation

        """
        if points is None:
            points = self.to_compute_points
        # snapshots generation
        if self.provider.is_file:
            snapshots = points.values()
        else:
            snapshots = []
            for p in points:
                if not self.provider.is_job:
                    # snapshots are in memory
                    path = None
                else:
                    # snapshots are on disk
                    path = os.path.join(self.output,
                                        self.output_tree['snapshots'],
                                        str(self.snapshot_counter))
                    self.snapshot_counter += 1

                if self.provider.is_function:
                    # create a snapshot on disk or in memory
                    s = Snapshot(p, self.provider(p))
                    snapshots += [Snapshot.convert(s, path=path)]
                elif self.provider.is_job:
                    # create a snapshot task
                    t = SnapshotTask(p, path)
                    snapshots += [self.snapshooter.submit(t.run)]

        # Fit the Surrogate [and POD]
        if self.pod is not None:
            if update:
                self.surrogate.space.empty()
                if self.provider.is_job:
                    for s in futures.as_completed(snapshots):
                        self.pod.update(s.result())
                else:
                    for s in snapshots:
                        self.pod.update(s)
            else:
                if self.provider.is_job:
                    _snapshots = []
                    for s in futures.as_completed(snapshots):
                        _snapshots += [s.result()]
                    snapshots = _snapshots
                self.pod.decompose(snapshots)

            self.data = self.pod.VS()

            try:  # if surrogate
                self.surrogate.fit(self.pod.points, self.data, pod=self.pod)
            except AttributeError:
                pass
        else:
            if self.provider.is_job:
                _snapshots = []
                for s in futures.as_completed(snapshots):
                    _snapshots += [s.result()]
                snapshots = _snapshots

            if snapshots:
                snapshots_ = [Snapshot.convert(s) for s in snapshots]
                snapshots = np.vstack([s.data for s in snapshots_])

                if update:
                    snapshots = np.vstack([self.data, snapshots])
                    if len(snapshots) != len(self.space):  # no resampling
                        snapshots = self.data
                        for snapshot in snapshots_:
                            try:
                                self.space += snapshot.point
                                snapshots = np.vstack([snapshots, snapshot.data])
                            except (AlienPointError, UnicityError, FullSpaceError) as tb:
                                self.logger.warning("Ignoring: {}".format(tb))

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
        while len(self.space) < self.space.max_points_nb:
            quality, point_loo = self.surrogate.estimate_quality()
            # quality = 0.5
            # point_loo = [-1.1780625, -0.8144629629629629]
            if quality >= self.settings['space']['resampling']['q2_criteria']:
                break

            try:
                new_point = self.space.refine(self.surrogate, point_loo)
            except FullSpaceError:
                break

            self.sampling(new_point, update=True)

            if self.settings['space']['resampling']['method'] == 'optimization':
                self.space.optimization_results()

    def write(self):
        """Write Surrogate [and POD] to disk."""
        if self.surrogate is not None:
            path = os.path.join(self.output, self.output_tree['surrogate'])
            try:
                os.makedirs(path)
            except OSError:
                pass
            self.surrogate.write(path)
        else:
            path = os.path.join(self.output, self.output_tree['space'])
            self.space.write(path)
        if self.pod is not None:
            path = os.path.join(self.output, self.output_tree['pod'])
            try:
                os.makedirs(path)
            except OSError:
                pass
            self.pod.write(path)
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.output, self.output_tree['data'])
            with open(path, 'wb') as f:
                pickler = pickle.Pickler(f)
                pickler.dump(self.data)
            self.logger.debug('Wrote data to {}'.format(path))

    def read(self):
        """Read Surrogate [and POD] from disk."""
        if self.surrogate is not None:
            self.surrogate.read(os.path.join(self.output,
                                             self.output_tree['surrogate']))
            self.space[:] = self.surrogate.space[:]
            self.data = self.surrogate.data
        else:
            path = os.path.join(self.output, self.output_tree['space'])
            self.space.read(path)
        if self.pod is not None:
            self.pod.read(os.path.join(self.output, self.output_tree['pod']))
            self.surrogate.pod = self.pod
        elif (self.pod is None) and (self.surrogate is None):
            path = os.path.join(self.output, self.output_tree['data'])
            with open(path, 'rb') as f:
                unpickler = pickle.Unpickler(f)
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

        :param :class:`space.point.Point` points: point(s) to predict
        :param bool write: write a snapshot or not
        :return: Result
        :rtype: lst(:class:`tasks.snapshot.Snapshot`) or np.array(n_points, n_features)
        :return: Standard deviation
        :rtype: lst(np.array)
        """
        if write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None

        return self.surrogate(points, path=output)

    def uq(self):
        """Perform UQ analysis."""
        output = os.path.join(self.output, self.output_tree['uq'])

        if self.pod is not None:
            data = self.pod.mean_snapshot + np.dot(self.pod.U, self.data.T).T
        else:
            data = self.data

        analyse = UQ(self.settings, self.surrogate,
                     self.space, data, output)

        if self.surrogate is not None:
            analyse.sobol()
        analyse.error_propagation()
