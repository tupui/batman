# -*- coding: utf-8 -*-
"""
SnapshotTask Class
==================

An object of this class corresponds to a snapshot.
The initialize classmethod is used to define common characteristics.
It is only after that a :class:`SnapshotTask` object can be created.

:Example:

::

    >> from tasks import SnapshotTask
    >> SnapshotTask.initialize(provider, data_files)
    >> task = SnapshotTask(point, path)
    >> task.run()

"""
import os
import re
import time
import logging
import shutil
from ..misc import clean_path
import subprocess
from .snapshot import Snapshot

opj = os.path.join


class SnapshotTask(object):

    """SnapshotTask class."""

    logger = logging.getLogger(__name__)

    started_file = 'job-started'
    finished_file = 'job-finished'
    touch = 'touch'

    dummy_line = '^(\s*#|\n)'
    '''Regular expression used that match a non executable statement in a shell script.'''

    info_line = '# State file insertion <<<<<<<<<<<<<<<<<<<<<<<<<\n'
    '''Commented lines to be inserted in the script before and after state file creation statements.'''

    period = None
    '''Period of time in seconds between state files existence checking.'''

    initialized = False
    '''Switch to check that the class has been initialized.'''

    # these attributes are settings common to all objects
    context = None
    '''Path to the directory which contains the files required for running the snapshot producer.'''

    private_directory = None
    '''Directory inside the task working_directory where batman stuff to keep will be located.'''

    command = None
    '''Command line used for executing the script.'''

    script = None
    '''Script that will run the snapshot producer.'''

    timeout = None
    '''Period of time to wait before a task is considered to be failed.'''

    data_files = None
    '''List of path to data files that defines a snapshot, paths must be relative to the context directory.'''

    clean_working_directory = False
    '''Clean the working directory after a task is terminated.'''

    @classmethod
    def _reset(cls):
        """Reset class attributes settings, for unit testing purposes."""
        cls.private_directory = None
        cls.context = None
        cls.command = None
        cls.script = None
        cls.timeout = None
        cls.data_files = None
        cls.initialized = False
        cls.clean_working_directory = False

    @classmethod
    def initialize(cls, provider, data_files):
        """Initialize the settings common to all objects."""
        if not os.path.isdir(provider['context']):
            cls.logger.error(
                'Cannot find the context directory: {}'.format(provider['context']))
            raise SystemError
        else:
            cls.context = clean_path(provider['context'])

        if not os.path.isfile(provider['script']):
            cls.logger.error(
                'Cannot find script file: {}'.format(provider['script']))
            raise SystemError
        else:
            cls.script = clean_path(provider['script'])

        cls.private_directory = provider['private-directory']
        cls.command = provider['command']
        cls.timeout = provider['timeout']
        cls.period = cls.timeout / 100.
        cls.data_files = data_files
        cls.clean_working_directory = provider['clean']

    def __init__(self, point, working_directory):
        """Create a task the will produce a snapshot.

        :param point: a point object
        :param working_directory: path to the directory in which the task will be run.
        """
        cls = self.__class__

        self.working_directory = clean_path(working_directory)
        self.point = point
        self.private_directory = opj(self.working_directory,
                                     cls.private_directory)
        self.started_file = opj(self.private_directory, cls.started_file)
        self.finished_file = opj(self.private_directory, cls.finished_file)
        self.script = opj(self.private_directory,
                          os.path.basename(cls.script))

    def run(self):
        """Return the result of the task.

        If the task has already been completed, just return its result, otherwise try to run it first.
        """
        cls = self.__class__
        result = self._before_run()

        command = cls.command.split() + [self.script]
        wkdir = self.working_directory
        # check if the task has already been run
        if result is None:
            self.handle = subprocess.Popen(command, cwd=wkdir)
            cls.logger.info('Launched : %s', ' '.join(command))
            result = self._after_run()
        else:
            cls.logger.info('Task is up to date and was not run.')

        return result

    def _before_run(self):
        """Prepare the run.

        The `working_directory` is created as a replica of the `context` directory tree with symbolic links. This is where the snapshot producer will be run, by issuing the command line generated with `command` and `script`. In addition to the files located in the directory `context`, the snapshot producer needs the parameters or the coordinates of the point in the parameters space bound to the snapshot to be created, this file is created here. In order to be informed about the state of the snapshot computation, without relying on querying a third party tool, the producer script is modified in order to create dummy state files in `working_directory`. A file which indicates that the script is starting (see `started_file`) is created right before any executable statements of the script and similarly right after all executable statements (see `finished_file`).
        """
        cls = self.__class__

        # check whether the task needs to be run
        task_already_done = True
        for f in cls.data_files:
            f = opj(self.private_directory, os.path.basename(f))
            if not os.path.isfile(f):
                task_already_done = False
                break

        if task_already_done:
            return self.private_directory

        else:
            # there must be no working directory
            if os.path.isdir(self.working_directory):
                cls.logger.error(
                    'Working directory already exists:\n{}'.format(self))
                raise SystemError

            # prepare the working directory
            self._copytree_with_symlinks(cls.context, self.working_directory)

            # create the private directory
            os.makedirs(self.private_directory)

            # write the point's coordinates
            Snapshot.write_point(self.point, self.private_directory)

            # add state files creation to the script
            self._state_files_hook()

            return None

    def _wait_for_completion(self):
        """Check if the task is completed."""
        cls = self.__class__

        # initial time for checking timeout
        start_time = time.time()

        while True:
            # check for finished state file
            if os.path.isfile(self.finished_file):
                break

            # otherwise check timeout
            elif time.time() - start_time > cls.timeout:
                # the task is taking too long
                # first see if it has started at all
                if os.path.isfile(self.started_file):
                    msg = 'The job started but did not finish'
                else:
                    msg = 'The job has not started'
                cls.logger.error('{}:\n{}'.format(msg, self))
                raise SystemError

            # elif os.path.isfile(self.started_file):
            #     start_time = os.path.getctime(self.started_file)
            #     life_span = time.time() - start_time
            #     if life_span > self.__class__.timeout: ...
            #     raise TaskTimeoutError('')

            # finally wait a few seconds
            else:
                time.sleep(cls.period)

    def _after_run(self):
        """Wait for the job completion and return the snapshot directory."""
        cls = self.__class__

        self._wait_for_completion()

        # move the expected snapshot files in the private directory
        for f in cls.data_files:
            f_original = opj(self.working_directory, f)
            if not os.path.isfile(f_original):
                cls.logger.error('Missing data file: {}'.format(f))
                raise SystemError
            else:
                f_moved = opj(self.private_directory, os.path.basename(f))
                os.rename(f_original, f_moved)

        # clean up working directory but the private directory
        if self.clean_working_directory:
            for f in os.listdir(self.working_directory):
                f = opj(self.working_directory, f)
                if f != self.private_directory:
                    try:
                        shutil.rmtree(f)
                    except OSError:
                        os.remove(f)

        return self.private_directory

    def __str__(self):
        cls = self.__class__
        s = 'command line : ' + cls.command + '\n'
        s += 'run form : ' + str(os.getcwd()) + '\n'
        s += 'point : ' + str(self.point) + '\n'
        s += 'context : ' + self.context + '\n'
        s += 'working directory : ' + self.working_directory
        return s

    def _state_files_hook(self):
        """Modify the script to create the state files."""
        cls = self.__class__

        # file creation command lines
        touch_started = cls.touch + ' ' + self.started_file + '\n'
        touch_finished = cls.touch + ' ' + self.finished_file + '\n'

        # wrap with info strings
        touch_started = cls.info_line + touch_started + cls.info_line
        touch_finished = cls.info_line + touch_finished + cls.info_line

        # insert started state file creation
        with open(cls.script, 'r') as f:
            script = f.readlines()
            for i, line in enumerate(script):
                if not re.match(cls.dummy_line, line):
                    script.insert(i, touch_started)
                    break

        # append finished state file creation
        script.append(touch_finished)

        # create the hooked script
        with open(self.script, 'w') as f:
            f.writelines(script)

    def _copytree_with_symlinks(self, original, copy):
        """Make a copy of a directory tree with symbolic links.

        :param original: path to the directory to be copied.
        :param copy    : path to the copy location.
        """
        os.makedirs(copy)
        for root, dirs, files in os.walk(original):
            local = root.replace(original, copy)
            for d in dirs:
                os.makedirs(opj(local, d))
            for f in files:
                os.symlink(opj(root, f), opj(local, f))
