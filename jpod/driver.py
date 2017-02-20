# coding: utf8
"""
Driver Class
============

Defines all methods used to interact with other classes.

:Example:

::

    >> from jpod import Driver
    >> driver = Driver(settings, script_path, output_path)
    >> driver.sampling_pod(update=False)
    >> driver.write_pod()
    >> driver.prediction(write=True)
    >> driver.write_model()
    >> driver.uq()

"""
import logging
import os
import sys

from concurrent import futures

from collections import OrderedDict
from . import mpi
from .pod import Pod
from .space import (Space, FullSpaceError, AlienPointError, UnicityError)
from .tasks import (PodServerTask, SnapshotTask, Snapshot, SnapshotProvider)
from .uq import UQ


class Driver(object):

    """Driver class."""

    logger = logging.getLogger(__name__)
    output_tree = {
        'snapshots': 'snapshots',
        'pod': 'pod',
        'predictions': 'predictions',
        'uq': 'uq',
    }
    '''Structure of the output directory.'''

    def __init__(self, settings, output):
        """Initialize Driver.

        From settings, init snapshot, space and POD.

        :param dict settings: settings
        :param str script: settings path
        :param str output: output path

        """
        self.settings = settings
        self.output = output
        self.external_pod = None
        self.pod = None
        self.snapshot_counter = 0

        # Snapshots
        Snapshot.initialize(self.settings['snapshot']['io'])
        self.provider = SnapshotProvider(self.settings['snapshot']['provider'])

        if self.provider.is_job:
            # compute relative path to snapshot files
            data_files = []
            for files in self.settings['snapshot']['io']['filenames'].values():
                for f in files:
                    data_files += [
                        os.path.join(self.provider['data-directory'], f)]
            SnapshotTask.initialize(self.provider, data_files)

            # snapshots generation manager
            self.snapshooter = futures.ThreadPoolExecutor(
                max_workers=self.settings['snapshot']['max_workers'])

        # Space
        self.space = Space(self.settings)

        if self.provider.is_file:
            # get the point from existing snapshot files,
            self.logger.info('Reading points from a list of snapshots files.')

            self.initial_points = OrderedDict()

            for path in self.provider:
                point = Snapshot.read_point(path)
                try:
                    self.space += point
                except (AlienPointError, UnicityError, FullSpaceError) as tb:
                    self.logger.warning("Ignoring: {}".format(tb))
                else:
                    self.initial_points[point] = path

        else:
            space_provider = self.settings['space']['sampling']
            if isinstance(space_provider, list):
                # a list of points is provided
                self.logger.info('Reading list of points from the settings.')
                self.initial_points = space_provider
                self.space += self.initial_points
            elif isinstance(space_provider, dict):
                # use point sampling
                self.initial_points = self.space.sampling(space_provider['init_size'])
            else:
                self.logger.error('Bad space provider.')
                raise SystemError

        # Pod
        if self.settings['pod']['server'] is not None:
            self.logger.info('Using external pod.')
            script = './settings.json'
            # get the pod server running and connect to its through its proxy
            self.external_pod = PodServerTask(self.settings['pod']['server']['port'],
                                              self.settings['pod']['server']['python'],
                                              script, self.output)
            self.external_pod.run()
            self.pod = self.external_pod.proxy.Pod(self.settings,
                                                   self.settings['snapshot']['io'])
        else:
            self.pod = Pod(self.settings)

    def sampling_pod(self, update):
        """Call private method _pod_processing."""
        self._pod_processing(self.initial_points, update)

    def resampling_pod(self):
        """Resampling of the POD.

        Generate new samples if quality and number of sample are not satisfied.
        From a new sample, it re-generates the POD.

        """
        max_points = self.settings['space']['sampling']['init_size'] + self.settings['space']['resampling']['resamp_size']
        while len(self.pod.points) < max_points:
            # quality, point_loo = self.pod.estimate_quality()
            quality = 0.5
            point_loo = [2.606125, 1.2379444444444445]
            # point_loo = [-1.1780625, -0.8144629629629629, -2.63886]

            if quality >= self.settings['space']['resampling']['q2_criteria']:
                break

            try:
                new_point = self.space.refine(self.pod, point_loo)
            except FullSpaceError:
                break

            self._pod_processing(new_point, True)

    def _pod_processing(self, points, update):
        """POD processing.

        Generates or retrieve the snapshots and then perform the POD.

        :param :class:`Space` points: points to perform the POD from
        :param bool update: perform dynamic or static computation

        """
        # snapshots generation
        snapshots = []
        for p in points:
            if self.provider.is_file:
                snapshots += [points[p]]
            else:
                if self.external_pod is None \
                   and not self.provider.is_job:
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

        # compute the pod
        if update:
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

    def write_pod(self):
        """Write POD to file."""
        self.pod.write(os.path.join(self.output, self.output_tree['pod']))

    def read_pod(self, path=None):
        """Read POD from file."""
        path = path or os.path.join(self.output, self.output_tree['pod'])
        self.pod.read(path)

    def restart(self):
        """Restart process."""
        # POD has already been computed previously
        self.logger.info('Restarting pod.')
        # read the pod data
        self.pod.read(os.path.join(self.output, self.output_tree['pod']))
        # points that have been already processed
        processed_points = self.pod.points
        self.snapshot_counter = len(processed_points)

        if len(processed_points) < self.initial_points.size:
            # static or dynamic pod is finished,
            # we add new points to be processed
            self.initial_points = [p for p in self.initial_points
                                   if p not in processed_points]
        else:
            # automatic resampling has to continue from
            # the processed points
            self.initial_points = []
            self.space.empty()
            self.space += processed_points

    def prediction(self, write=False):
        """Perform a prediction."""
        if self.external_pod is not None or write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None

        self.pod.predict(self.settings['surrogate']['method'],
                         self.settings['surrogate']['predictions'], output)

    def prediction_without_computation(self, write=False):
        """Perform a prediction using an existing model read from file."""
        if self.external_pod is not None or write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None
        model = self.read_model()
        self.pod.predict_without_computation(
            model, self.settings['surrogate']['predictions'], output)

    def write_model(self):
        """Write model to file."""
        self.pod.write_model(
            os.path.join(
                self.output,
                self.output_tree['pod']))

    def read_model(self, path=None):
        """Read model from file."""
        path = path or os.path.join(self.output, self.output_tree['pod'])
        return self.pod.read_model(path)

    def uq(self):
        """Perform UQ analysis."""
        output = os.path.join(self.output, self.output_tree['uq'])
        analyse = UQ(self.pod, self.settings, output)
        analyse.sobol()
        analyse.error_propagation()

    def __del__(self):
        """Driver destructor."""
        # terminate pending tasks
        if mpi.myid == 0 and self.external_pod is not None:
            self.logger.info('Terminating the external pod.')
            self.external_pod.terminate()
