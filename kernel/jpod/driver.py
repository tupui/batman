import logging
import os
import shutil
import subprocess

from concurrent import futures

from misc import OrderedDict
import mpi
import numpy as N
from pod import Snapshot, Pod
from space import Space, FullSpaceError, AlienPointError
from tasks import PodServerTask, SnapshotTask, Task
from uq import UQ
try:
    subprocess.Popen.terminate
except AttributeError:
    # fix for python < 2.6
    # the following missing functions comes from python2.6 subprocess module
    import signal

    def subprocess_send_signal(self, sig):
        """Send a signal to the process
        """
        os.kill(self.pid, sig)

    def subprocess_terminate(self):
        """Terminate the process with SIGTERM
        """
        self.send_signal(signal.SIGTERM)

    subprocess.Popen.send_signal = subprocess_send_signal
    subprocess.Popen.terminate = subprocess_terminate


# force numpy to raise an exception on floating-point errors
N.seterr(all='raise', under='warn')


class SnapshotProvider(object):
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


class Driver(object):
    """docstring for Driver"""

    output_tree = {
        # 'snapshot-template' : 'snapshot-template',
        'snapshots': 'snapshots',
        'pod': 'pod',
        'predictions': 'predictions',
    }
    '''Structure of the output directory.'''

    def __init__(self, snapshot_settings, space_settings, output, plot=False):
        self.pod_quality = None
        '''POD automatic resampling quality.'''

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
        self._init_snapshot(snapshot_settings)

        # parameter space and points
        self._init_space(space_settings, plot)

    def _init_snapshot(self, settings):
        """docstring for _init_snapshot"""
        Snapshot.initialize(settings['io'])

        # snapshot generation
        self.provider = SnapshotProvider(settings['provider'])

        if self.provider.is_job:
            # compute relative path to snapshot files
            data_files = []
            for files in settings['io']['filenames'].values():
                for f in files:
                    data_files += [
                        os.path.join(
                            self.provider['data-directory'],
                            f)]

            SnapshotTask.initialize(
                self.provider['context'],
                self.provider['command'],
                self.provider['script'],
                self.provider['timeout'],
                data_files,
                self.provider['private-directory'],
                self.provider['clean'])

            # snapshots generation manager
            self.snapshooter = futures.ThreadPoolExecutor(
                max_workers=settings['max_workers'])

    def _init_space(self, settings, plot):
        # space
        self.space = Space(settings['corners'], settings.get('size_max'), settings.get('delta_space'),
                           plot)

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
                        'Ignoring snapshot\n\t%s\n\tbecause its point %s is outside the space.',
                        path,
                        point)
                else:
                    self.initial_points[point] = path

        else:
            space_provider = settings['provider']
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
                raise TypeError('Bad space provider.')

    def __del__(self):
        """docstring for __del__"""
        # terminate pending tasks
        if mpi.myid == 0 \
           and self.external_pod is not None:
            self.logger.info('Terminating the external pod.')
            self.external_pod.terminate()

    def _pod_processing(self, points, update):
        """docstring for fname"""
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

    def init_pod(self, settings, script):
        self.pod_quality = settings.pod['quality']

        if settings.pod['server'] is not None:
            if mpi.size > 1:
                raise Exception(
                    'When using the external pod, the driver must be sequential.')

            self.logger.info('Using external pod.')
            # get the pod server running and connect to its through its proxy
            self.external_pod = PodServerTask(settings.pod['server']['port'],
                                              settings.pod['server']['python'],
                                              script, self.output)
            self.external_pod.run()
            # self.external_pod._after_run()
            self.pod = self.external_pod.proxy.Pod(settings.pod['tolerance'],
                                                   settings.pod['dim_max'],
                                                   settings.snapshot['io'])
        else:
            # directly instantiate the pod,
            # the snapshot class is initialized as a by product
            self.pod = Pod(settings.pod['tolerance'], settings.pod['dim_max'])

    def fixed_sampling_pod(self, update):
        """docstring for static_pod"""
        if self.pod is None:
            raise Exception(
                "driver's pod has not been initialized, call init_pod first.")
        self._pod_processing(self.initial_points, update)

    def automatic_resampling_pod(self):
        """docstring for static_pod"""
        if self.pod is None:
            raise Exception(
                "driver's pod has not been initialized, call init_pod first.")

        while True:
            quality, point = self.pod.estimate_quality()

            if quality <= self.pod_quality:
                break

            try:
                new_points = self.space.refine_around(
                    point)  # FIXME: restart !
            except FullSpaceError:
                break

            self._pod_processing(new_points, True)

    def write_pod(self):
        """docstring for static_pod"""
        self.pod.write(os.path.join(self.output, self.output_tree['pod']))

    def read_pod(self, path=None):
        """docstring for static_pod"""
        path = path or os.path.join(self.output, self.output_tree['pod'])
        self.pod.read(path)

    def prediction(self, settings, write=False):
        if self.external_pod is not None \
           or write:
            output = os.path.join(self.output, self.output_tree['predictions'])
        else:
            output = None

        return self.pod.predict(settings['method'], settings['points'], output)
    
    def uq(self, settings):
	print "HELLLLLOOOOOOOO"
        analyse = UQ(settings)	
	analyse.sobol()        

    def restart(self):
        self.logger.info('Restarting pod.')
        # read the pod data
        self.pod.read(os.path.join(self.output, self.output_tree['pod']))
        # points that have been already processed
        processed_points = self.pod.points

        if set(processed_points).issubset(self.initial_points):
            # static or dynamic pod is not finished, the remaining points have
            # to be processed
            self.initial_points = [p for p in self.initial_points
                                   if p not in processed_points]
        else:
            # static or dynamic pod is done,
            # the eventual automatic resampling has to continue from the processed points
            # FIXME: space needs the refiner structure!
            self.initial_points = []
            self.space.empty()
            self.space.add(processed_points)


# def catch_and_clean(method):
#     def wrapped_method(self, *args, **kwargs):
#         try:
#             method(*args, **kwargs)
#         finally:
#             self.finalize()
#     return wrapped_method
#
#
# class SafeDriver(Driver):
#
#
#     """docstring for SafeDriver"""
#     def finalize(self):
#         # terminate pending tasks
#         if mpi.myid == 0 \
#            and self.external_pod is not None:
#             self.logger.info('Terminating the external pod.')
#             self.external_pod.terminate()
