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
import logging
try:
    from backports import tempfile
except ImportError:
    import tempfile
import os
import copy
import shutil
import subprocess as sp
import numpy as np
from .local_executor import LocalExecutor
from .remote_executor import MasterRemoteExecutor
from .sample_cache import SampleCache
from ..space import Sample


class ProviderJob:
    """Provides Snapshots built through a 3rd-party program."""

    logger = logging.getLogger(__name__)

    def __init__(self, plabels, flabels, command, context_directory,
                 psizes=None, fsizes=None,
                 coupling=None, hosts=None,
                 pool=None, clean=False,
                 discover_pattern=None, save_dir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """Initialize the provider.

        :param list(str) plabels: input parameter names.
        :param list(str) flabels: output feature names.
        :param str command: command to be executed for computing new snapshots.
        :param str context_directory: store every ressource required for
          executing a job.
        :param list(int) psizes: number of components of parameters.
        :param list(int) fsizes: number of components of output features.
        :param dict coupling: Definition of the snapshots IO files:

            - **coupling_directory** (str) -- sub-directory in
              ``context_directory`` that will contain input parameters and
              output file.
            - **input_fname** (str) -- basename for files storing the point
              coordinates ``plabels``.
            - **input_format** (str) -- ``json`` (default), ``csv``, ``npy``,
              ``npz``.
            - **output_fname** (str) -- basename for files storing values
              associated to ``flabels``.
            - **output_format** (str) -- ``json`` (default), ``csv``, ``npy``,
              ``npz``.

        :param list(dict) hosts: Definition of the remote HOSTS if any:

            - **hostname** (str) -- Remote host to connect to.
            - **remote_root** (str) -- Remote folder to create and store data.
            - **username** (str) -- username.
            - **password** (str) -- password.

        :param pool: pool executor.
        :type pool: :class:`concurrent.futures`.xxx.xxx.Executor.
        :param bool clean: whether to remove working directories.
        :param str discover_pattern: UNIX-style patterns for directories with
          pairs of sample files to import.
        :param str save_dir: path to a directory for saving known snapshots.
        :param str space_fname: name of space file to write.
        :param str data_fname: name of data file to write.
        :param str space_format: space file format.
        :param str data_format: data file format.
        """
        # discover existing snapshots
        self._cache = SampleCache(plabels, flabels, psizes, fsizes, save_dir,
                                  space_fname, space_format,
                                  data_fname, data_format)
        if discover_pattern:
            self._cache.discover(discover_pattern)
            self._cache.save()

        self.safe_saved = False

        # job specification
        self._job = {
            'command': command,
            'context_directory': context_directory,
            'coupling_directory': 'batman-coupling',
            'input_fname': space_fname,
            'input_sizes': self.psizes,
            'input_labels': self.plabels,
            'input_format': space_format,
            'output_fname': data_fname,
            'output_sizes': self.fsizes,
            'output_labels': self.flabels,
            'output_format': data_format,
            'clean': clean,
        }
        if coupling is not None:
            self._job.update(coupling)
        self.logger.debug('Job specification: {}'.format(self._job))

        # execution
        if save_dir is not None:
            workdir = save_dir
        else:
            _tmp = tempfile.TemporaryDirectory()
            self._job['clean'] = True
            workdir = _tmp.name

        self.backupdir = os.path.join(workdir, '.backup')
        try:
            os.makedirs(self.backupdir)
        except OSError:
            self.logger.warning('Was not able to create backup directory')
        finally:
            self._cache_backup = copy.deepcopy(self._cache)

        if pool is not None:
            self._pool = pool

        if hosts is not None:
            self._executor = MasterRemoteExecutor(local_root=workdir,
                                                  job=self._job, hosts=hosts)
        else:
            self._executor = LocalExecutor(local_root=workdir, **self._job)

    @property
    def plabels(self):
        """Names of space parameters."""
        return self._cache.plabels

    @property
    def flabels(self):
        """Names of data features."""
        return self._cache.flabels

    @property
    def psizes(self):
        """Shape of space parameters."""
        return self._cache.psizes

    @property
    def fsizes(self):
        """Shape of data features."""
        return self._cache.fsizes

    @property
    def known_points(self):
        """List of points whose associated data is already known."""
        return self._cache.space

    def require_data(self, points):
        """Return samples for requested points.

        Data for unknown points if generated through an external job.

        :param array_like points: points to compute (n_points, n_features).
        :return: samples for requested points (carry both space and data)
        :rtype: :class:`Sample`
        """
        self.safe_saved = False

        if np.size(points) == 0:
            return self._cache[:0]  # return empty container
        points = np.atleast_2d(points)
        self.logger.debug('Requested Snapshots for points {}'.format(points))

        # locate results in cache
        idx = self._cache.locate(points)

        new_points = points[idx >= len(self._cache)]
        if len(new_points) > 0:
            # build new samples
            new_idx = idx[idx >= len(self._cache)]
            try:
                mapper = self._pool.map
            except AttributeError:  # no pool
                samples, failed = self.build_data(new_points, new_idx)
                self._cache += samples
            else:
                ret_list = list(mapper(self.build_data, new_points, new_idx))
                self._cache = sum([ret for ret, err in ret_list], self._cache)
                failed = sum([err for ret, err in ret_list], [])

            # safelly save sample space and data
            try:
                self._cache.save()
                self.safe_saved = True
            except OSError:
                self.logger.error('Failed to save sample')
                raise SystemExit

            # check for failed jobs
            if failed:
                failed_points = [tuple(point) for point, err in failed]
                self.logger.error('Jobs failed for points {}'.format(failed_points))
                err = failed[0][1]
                self.logger.error(err.stderr)
                raise sp.CalledProcessError(err.returncode, err.cmd,
                                            output=err.stdout, stderr=err.stderr)

        return self._cache[idx]

    def build_data(self, points, sample_id=None):
        """Compute data for requested points.

        Ressources for executing a job are copied from
        the context directory to a work directory.
        The shell command is executed from this directory.
        The command shall find its inputs and place its outputs
        in the coupling sub-directory, inside the work directory.

        :param array_like points: points to compute (n_points, n_features).
        :param list sample_id: points indices in the points list.
        :return: samples for requested points (carry both space and data)
          and failed if any.
        :rtype: :class:`Sample`, list([point, err])
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
            try:
                snapshot = self._executor.snapshot(point, i)
            except sp.CalledProcessError as err:
                failed.append((point, err))
                continue

            sample += snapshot

            # backup sample space and data
            self._cache_backup += sample
            self._cache_backup.save(self.backupdir)
        return sample, failed

    def __del__(self):
        """Remove backup directory."""
        if self.safe_saved:
            shutil.rmtree(self.backupdir)
