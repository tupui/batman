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

from concurrent import futures

from collections import OrderedDict
from . import mpi
from .pod import (Snapshot, Pod)
from .space import (Space, FullSpaceError, AlienPointError)
from .tasks import (PodServerTask, SnapshotTask)
from .uq import UQ


class SnapshotProvider():

    """Utility class to make the code more readable.

    This is how the provider type is figured out.
    """

    def __init__(self, provider):
        self.provider = provider

    @property
    def is_file(self):
        return isinstance(self.provider, list)

    @property
    def is_job(self):
        return isinstance(self.provider, dict)

    @property
    def is_function(self):
        return callable(self.provider)

    def __getitem__(self, key):
        return self.provider[key]

    def __call__(self, *args, **kwargs):
        return self.provider(*args, **kwargs)


class Driver():

    """Driver class."""

    output_tree = {
        'snapshots': 'snapshots',
        'pod': 'pod',
        'predictions': 'predictions',
        'uq': 'uq',
    }
    '''Structure of the output directory.'''

    def __init__(self, settings, script, output):
        """Initialize Driver.

        From settings, init snapshot, space and POD.

        :param dict settings: settings
        :param str script: settings path
        :param str output: output path

        """
        self.settings = settings
        '''JPOD settings'''

        self.output = output
        '''Path to output directory.'''

        self.logger = logging.getLogger(__name__)
        '''Console and file logger.'''

        self.external_pod = None
        '''External pod task handle.'''

        self.pod = None
        '''POD processing, either local or external.'''

        self.snapshooter = None
        '''Snapshots generation manager.'''

        self.provider = None
        '''Snapshot provider, it generates a snapshot.'''

        self.space = None
        '''Parameter space.'''

        self.initial_points = None
        '''Points in the parameter space for the static pod.'''

        self.snapshot_counter = 0
        '''Counter for numbering the snapshots.'''

        # snapshot computation
        self._init_snapshot()

        # parameter space and points
        self._init_space()

        # Init pod
        self._init_pod(script)

    def _init_snapshot(self):
        """Initialize snapshots  :class:`SnapshotTask`."""
        Snapshot.initialize(self.settings['snapshot']['io'])

        # snapshot generation
        self.provider = SnapshotProvider(self.settings['snapshot']['provider'])

        if self.provider.is_job:
            # compute relative path to snapshot files
            data_files = []
            for files in self.settings['snapshot']['io']['filenames'].values():
                for f in files:
                    data_files += [
                        os.path.join(
                            self.provider['data-directory'],
                            f)]
            SnapshotTask.initialize(self.provider, data_files)

            # snapshots generation manager
            self.snapshooter = futures.ThreadPoolExecutor(
                max_workers=self.settings['snapshot']['max_workers'])

    def _init_space(self):
        """Initialize :class:`Space` by creating the DOE."""
        # space
        self.space = Space(self.settings)

        # initial points
        if self.provider.is_file:
            # get the point from existing snapshot files,
            # the points outside the space are ignored
            self.logger.info('Reading points from a list of snapshots files.')

            self.initial_points = OrderedDict()

            for path in self.provider:
                point = Snapshot.read_point(path)
                try:
                    self.space.add([point])
                except AlienPointError:
                    self.logger.info(
                        'Ignoring snapshot\n\t{}\n\tbecause its point {} is outside the space.'.format(path,point))
                else:
                    self.initial_points[point] = path

        else:
            space_provider = self.settings['space']['provider']
            if isinstance(space_provider, list):
                # a list of points is provided
                self.logger.info('Reading list of points from the settings.')
                self.initial_points = space_provider
                self.space.add(self.initial_points)
            elif isinstance(space_provider, dict):
                # use point sampling
                self.initial_points = self.space.sampling(space_provider['method'],
                                                          space_provider['size'])
            else:
                self.logger.error('Bad space provider.')
                raise SystemError

    def _init_pod(self, script):
        """Initialize :class:`Pod`."""
        if self.settings['pod']['server'] is not None:
            if mpi.size > 1:
                raise Exception(
                    'When using the external pod, the driver must be sequential.')

            self.logger.info('Using external pod.')
            # get the pod server running and connect to its through its proxy
            self.external_pod = PodServerTask(self.settings['pod']['server']['port'],
                                              self.settings['pod']['server']['python'],
                                              script, self.output)
            self.external_pod.run()
            # self.external_pod._after_run()
            self.pod = self.external_pod.proxy.Pod(self.settings['pod']['tolerance'],
                                                   self.settings['pod']['dim_max'],
                                                   self.settings['snapshot']['io'])
        else:
            # directly instantiate the pod,
            # the snapshot class is initialized as a by product
            self.pod = Pod(self.settings['pod']['tolerance'],
                           self.settings['pod']['dim_max'],
                           self.settings['space']['corners'])

    def sampling_pod(self, update):
        """Call private method _pod_processing."""
        self._pod_processing(self.initial_points, update)

    def resampling_pod(self):
        """Resampling of the POD.

        Generate new samples if quality and number of sample are not satisfied.
        From a new sample, it re-generates the POD.

        """
        while len(self.pod.points) < self.settings['space']['size_max']:
            quality, point_loo = self.pod.estimate_quality()
            if quality >= self.settings['pod']['quality']:
                break

            try:
                new_point = self.space.refine(self.pod, point_loo)
            except FullSpaceError:
                break

            self._pod_processing(new_point, True)

    def _pod_processing(self, points, update):
        """POD processing.

        Generates or retrieve the snapshots and then perform the POD.

        :param :class:`SpaceBase` points: points to perform the POD from
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
            self.space.add(processed_points)

    def prediction(self, write=False):
        """Perform a prediction."""
        if self.external_pod is not None or write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None

        self.pod.predict(self.settings['prediction']['method'],
                         self.settings['prediction']['points'], output)

    def prediction_without_computation(self, write=False):
        """Perform a prediction using an existing model read from file."""
        if self.external_pod is not None or write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None
        model = self.read_model()
        self.pod.predict_without_computation(
            model, self.settings['prediction']['points'], output)

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
