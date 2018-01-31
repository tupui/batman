# coding: utf-8
"""
This module defines a specialized Provider class.

The ProviderFile class handles jobs that consist in
executing an external job that produce a data file.
"""
from collections import OrderedDict
import os
import shutil
import logging
import subprocess as sp
import numpy as np
from .snapshot import Snapshot
from ..space import Point


class ProviderFile(object):
    """A Provider class that build snapshots whose data come from a file."""

    logger = logging.getLogger(__name__)

    def __init__(self, executor, io_manager, job_settings):
        """Initialize the provider.

        :param executor: a task pool executor.
        :param io_manager: defines snapshots as files.
        :param dict job_settings: specify a job for building snapshot data
            with the following:

            - **command** (str): command to execute.
            - **context_directory** (str): directory from wich to get jobs resources.
            - **coupling_directory** (str): default is ``batman-coupling``.
            - **clean** (bool): default is ``False``.

        :type executor: :class:`concurrent.futures.Executor`
        :type io_manager: :class:`SnapshotIO`
        """
        self._io = io_manager
        self._executor = executor

        # automatic data file generation
        try:
            self._job = {
                'command': job_settings['command'],
                'context_directory': job_settings['context_directory'],
                'coupling_directory': job_settings.get('coupling_directory',
                                                       'batman-coupling'),
                'clean': job_settings.get('clean', False),
            }
            self.logger.debug('Job for snapshot file creation: {}'.format(self._job))
        except KeyError:
            self._job = None
            self.logger.warning(
                'No job were specified for snapshot file creation. '
                'BATMAN will crash if a non existing snapshot is requested.'
            )

        # automatic discovery of existing points
        self._known_points = OrderedDict()
        discover = job_settings.get('discover_from')
        if discover is not None:
            for root, _, files in os.walk(discover):
                if (self._io.point_filename in files) and (self._io.data_filename in files):
                    # found a point
                    try:
                        point = self._io.read_point(root)
                    except KeyError:
                        self.logger.debug('Ignored bad formatted point in {}'.format(root))
                    else:
                        self._known_points[point] = root
                        self.logger.debug('Discovered point {} in {}'.format(point, root))

    @property
    def known_points(self):
        """Dictionnary binding known snapshots with their location."""
        return self._known_points

    def snapshot(self, point, snapshot_dir):
        """Snapshot bound to an asynchronous job that read data from a file.

        :param point: the point in parameter space at which to provide a
          snapshot.
        :param str snapshot_dir: the directory containing the snapshot data
          files.

        :type point: :class:`batman.space.Point`.
        :return: A snapshot.
        :rtype: :class:`Snapshot`.
        """
        self.logger.debug('Request snapshot for point {}'.format(point))
        return Snapshot(point, self._executor.submit(self._load_data, point, snapshot_dir))

    def _load_data(self, point, snapshot_dir):
        """Load data from a file. Build it if not exist.

        :param point: point in parameter space.
        :param str snapshot_dir: directory containing files to read.
        :type point: :class:`batman.space.Point`.
        :rtype: :class:`numpy.ndarray`
        """
        point = Point(point)
        try:
            # link current location to actual snapshot location
            os.symlink(self._known_points[point], snapshot_dir)

        except OSError:
            # current location is actual snapshot location
            pass

        except KeyError:
            # point is not known

            data_filepath = os.path.join(snapshot_dir, self._io.data_filename)
            point_filepath = os.path.join(snapshot_dir, self._io.point_filename)

            if not (os.path.isfile(data_filepath) and os.path.isfile(point_filepath)):
                # snapshot files must be created
                if self._job is None:
                    self.logger.error('Cannot build requested snapshot data for point {} !'
                                      .format(point))
                    raise SystemExit

                # "Alfred, please prepare the batmobile"
                work_dir = os.path.join(snapshot_dir, '.wkdir')
                self._job_initialize(point, work_dir)
                self._job_execute(point, work_dir)
                self._job_finalize(point, work_dir, snapshot_dir)

            # record current location as actual snapshot location
            self._known_points[point] = snapshot_dir

        # read data file
        data = self._io.read_data(snapshot_dir)
        self.logger.debug('Read data for point {} from directory {}'.format(point, snapshot_dir))
        return data

    def _job_initialize(self, point, work_dir):
        """Setup job execution.

        Create and populate:
        - work-directory from context-directory (use symbolic links)
        - coupling subdirectory

        :param point: point in parameter space.
        :param str work_dir: directory to populate with job script and resource
          files.
        :type point: :class:`Point`
        """
        coupling_dir = os.path.join(work_dir, self._job['coupling_directory'])

        # copy-link the content of 'context_dir' to 'snapshot_dir'
        # wkdir shall not exist at all
        os.makedirs(coupling_dir)
        context_dir = os.path.abspath(self._job['context_directory'])
        for root, dirs, files in os.walk(context_dir):
            local = root.replace(context_dir, work_dir)
            for d in dirs:
                os.makedirs(os.path.join(local, d))
            for f in files:
                os.symlink(os.path.join(root, f), os.path.join(local, f))
        self._io.write_point(coupling_dir, point)
        self.logger.debug('Point {} :: Prepared workdir in {}'.format(point, work_dir))
        self.logger.debug('Point {} :: Coupling directory is {}'.format(point, coupling_dir))

    def _job_execute(self, point, work_dir):
        """Execute job.

        :param point: point in parameter space.
        :param str work_dir: directory from which to launch the job.
        :type point: :class:`Point`
        :raises :exc:`subprocess.CalledProcessError`
        """
        cmd = self._job['command'].split()
        job = sp.Popen(cmd, cwd=work_dir)
        self.logger.debug('Point {} :: Starting job in {}'.format(point, work_dir))
        ret = job.wait()
        if ret != 0:
            raise sp.CalledProcessError(ret, self._job['command'])

    def _job_finalize(self, point, work_dir, snapshot_dir):
        """Finalize job execution.

        - move snapshot data from coupling subdir to snapshot dir
        - clean workdir

        :param point: point in parameter space.
        :param str work_dir: directory containing job output.
        :param str snapshot_dir: directory in which to put snapshot files.
        :type point: :class:`Point`
        """
        coupling_dir = os.path.join(work_dir, self._job['coupling_directory'])

        # move data to snapshot directory
        os.rename(os.path.join(coupling_dir, self._io.point_filename),
                  os.path.join(snapshot_dir, self._io.point_filename))
        os.rename(os.path.join(coupling_dir, self._io.data_filename),
                  os.path.join(snapshot_dir, self._io.data_filename))
        self.logger.debug('Point {} :: Moved snapshot from {} to {}'
                          .format(point, coupling_dir, snapshot_dir))

        if self._job['clean']:
            # remove workdir
            shutil.rmtree(work_dir)
            self.logger.debug('Point {} :: Removed workdir {}'.format(point, work_dir))
        else:
            # keep a symlink to snapshot data in coupling directory
            os.symlink(os.path.join(snapshot_dir, self._io.point_filename),
                       os.path.join(coupling_dir, self._io.point_filename))
            os.symlink(os.path.join(snapshot_dir, self._io.data_filename),
                       os.path.join(coupling_dir, self._io.data_filename))
            self.logger.debug('Point {} :: Set symlinks from {} to {}'
                              .format(point, snapshot_dir, work_dir))
