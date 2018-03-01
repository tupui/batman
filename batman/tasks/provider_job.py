# coding: utf-8
"""
Data Provider: Build snapshots through a 3rd-party program
==========================================================

This provider builds its data using a shell command.

The command executes a 3rd-party program.
It can be anything from a small exotic perl command
to a heavy CFD simulation.
Coupling is done through files.
"""
import os
import shutil
import logging
import tempfile
import subprocess as sp
import numpy as np
from .sample_cache import SampleCache
from ..space import Sample
from ..input_output import formater


class ProviderJob(object):
    """Provides Snapshots built through a 3rd-party program"""

    logger = logging.getLogger(__name__)

    def __init__(self, plabels, flabels, command, context_directory,
                 coupling_directory='batman-coupling',
                 psizes=None, fsizes=None,
                 executor=None, clean=False,
                 discover_pattern=None, save_dir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """Initialize the provider.

        :param list(str) plabels: input parameter names.
        :param list(str) flabels: output feature names.
        :param str command: command to be executed for computing new snapshots.
        :param str context_directory: store every ressource required for executing a job.
        :param str coupling_directory: subdirectory in which input/output files are placed.
        :param list(int) psizes: number of components of parameters.
        :param list(int) fsizes: number of components of output features.
        :param executor: Pool executor for asynchronous jobs.
        :param bool clean: whether to remove working directories.
        :param str discover_pattern: UNIX-style patterns for directories with pairs
            of sample files to import.
        :param str save_dir: path to a directory for saving known snapshots.
        :param str space_fname: name of space file to write.
        :param str data_fname: name of data file to write.
        :param str space_format: space file format.
        :param str data_format: data file format.

        :type executor: :py:class:`concurrent.futures.Executor`
        """
        if executor is not None:
            self._executor = executor

        # job specification
        self._job = {
            'command': command,
            'context_directory': context_directory,
            'coupling_directory': coupling_directory,
            'input_file': space_fname,
            'input_format': space_format,
            'output_file': data_fname,
            'output_format': data_format,
            'clean': clean,
        }
        self.logger.debug('Job specification: {}'.format(self._job))

        # discover existing snapshots
        self._cache = SampleCache(plabels, flabels, psizes, fsizes, save_dir,
                                  space_fname, space_format,
                                  data_fname, data_format)
        if discover_pattern:
            self._cache.discover(discover_pattern)
            self._cache.save()

        # choose a workdir
        if save_dir is not None:
            self._workdir = save_dir
        else:
            self._tmp = tempfile.TemporaryDirectory()
            self._workdir = self._tmp.name
            self._job['clean'] = True

    @property
    def plabels(self):
        """Names of space parameters"""
        return self._cache.plabels

    @property
    def flabels(self):
        """Names of data features"""
        return self._cache.flabels

    @property
    def psizes(self):
        """Shape of space parameters"""
        return self._cache.psizes

    @property
    def fsizes(self):
        """Shape of data features"""
        return self._cache.fsizes

    @property
    def known_points(self):
        """List of points whose associated data is already known"""
        return self._cache.space

    def require_data(self, points):
        """Return samples for requested points.

        Data for unknown points if generated through an external job.

        :return: samples for requested points (carry both space and data)
        :rtype: :class:`Sample`
        """
        if np.size(points) == 0:
            return self._cache[:0]
        points = np.atleast_2d(points)
        self.logger.debug('Requested Snapshots for points {}'.format(points))

        # locate results in cache
        idx = self._cache.locate(points)

        new_points = points[idx >= len(self._cache)]
        if len(new_points) > 0:
            # build new samples
            new_idx = idx[idx >= len(self._cache)]
            try:
                mapper = self._executor.map
            except AttributeError:
                samples, failed = self.build_data(new_points, new_idx)
                self._cache += samples
            else:
                ret_list = list(mapper(self.build_data, new_points, new_idx))
                self._cache = sum([ret for ret, err in ret_list], self._cache)
                failed = sum([err for ret, err in ret_list], [])
            self._cache.save()

            # check for failed jobs
            if len(failed) > 0:
                failed_points = [tuple(point) for point, err in failed]
                self.logger.error('Jobs failed for points {}'.format(failed_points))
                err = failed[0][1]
                raise sp.CalledProcessError(err.returncode, err.cmd)

        return self._cache[idx]

    def build_data(self, points, sample_id=None):
        """Compute data for requested points.

        Ressources for executing a job are copied from
        the context directory to a work directory.
        The shell command is executed from this directory.
        The command shall find its inputs and place its outputs
        in the coupling sub-directory, inside the work directory.

        :return: samples for requested points (carry both space and data)
        :rtype: :class:`Sample`
        """
        self.logger.debug('Build new Snapshots for points {}'.format(points))
        sample = Sample(plabels=self.plabels, flabels=self.flabels,
                        psizes=self.psizes, fsizes=self.fsizes,
                        pformat=self._job['input_format'],
                        fformat=self._job['output_format'])

        points = np.atleast_2d(points)
        sample_id = np.atleast_1d(sample_id) if sample_id is not None else range(points)
        failed = []
        for i, point in zip(sample_id, points):
            # start job
            work_dir = os.path.join(self._workdir, str(i))
            self._job_initialize(point, work_dir)
            try:
                self._job_execute(point, work_dir)
            except sp.CalledProcessError as err:
                failed.append((point, err))
                continue
            # get result
            sample_dir = os.path.join(work_dir, self._job['coupling_directory'])
            space_fname = os.path.join(sample_dir, self._job['input_file'])
            data_fname = os.path.join(sample_dir, self._job['output_file'])
            sample.read(space_fname, data_fname)
            if self._job['clean']:
                shutil.rmtree(work_dir)
        return sample, failed

    def _job_initialize(self, point, work_dir):
        """Setup job execution.

        Create and populate:
        - work-directory from context-directory (use symbolic links)
        - coupling subdirectory

        :param array-like point: point in parameter space.
        :param str work_dir: directory to populate with job script and resource files.
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

        # create input_file
        point_file = os.path.join(coupling_dir, self._job['input_file'])
        point_formater = formater(self._job['input_format'])
        point_formater.write(point_file, point, self.plabels)

        self.logger.debug('Point {} - Prepared workdir in {}'.format(point, work_dir))
        self.logger.debug('Point {} - Coupling directory is {}'.format(point, coupling_dir))

    def _job_execute(self, point, work_dir):
        """Execute job.

        :param array-like point: point in parameter space.
        :param str work_dir: directory from which to launch the job.
        :raises :exc:`subprocess.CalledProcessError`
        """
        cmd = self._job['command'].split()
        job = sp.Popen(cmd, cwd=work_dir)
        self.logger.debug('Point {} - Starting job in {}'.format(point, work_dir))
        ret = job.wait()
        if ret != 0:
            raise sp.CalledProcessError(ret, self._job['command'])
