import os
import re
import time
import logging
import shutil
from misc import clean_path
from task import Task
from space import Point
from pod import Snapshot

opj = os.path.join


class TaskTimeoutError(Exception):
    pass


class TaskFailed(Exception):
    pass




class SnapshotTask(Task):
    """docstring for SnapshotTask"""

    # these attributes should be independent of external settings
    started_file  = 'job-started'
    '''Name of the empty file which indicates that the task script has been started.'''

    finished_file = 'job-finished'
    '''Name of the empty file which indicates that the task script has finished.'''

    touch = 'touch'
    '''Shell program to create an empty file, used to create the script state files.'''

    dummy_line = '^(\s*#|\n)'
    '''Regular expression used that match a non executable statement in a shell script.'''

    info_lines = ['# start jpod state file insertion <<<<<<<<<<<<<<<<<<<<<<<<<\n',
                  '# end jpod state file insertion <<<<<<<<<<<<<<<<<<<<<<<<<<<\n']
    '''Commented lines to be inserted in the script before and after state file creation statements.'''

    period = 1
    '''Period of time in seconds between state files existence checking.'''

    initialized = False
    '''Switch to check that the class has been initialized.'''


    # these attributes are settings common to all objects
    context = None
    '''Path to the directory which contains the files required for running the snapshot producer.'''

    private_directory = None
    '''Directory inside the task working_directory where jpod stuff to keep will be located.'''

    command = None
    '''Command line used for executing the script.'''

    script = None
    '''Script that will run the snapshot producer.'''

    timeout = None
    '''Period of time to wait before a task is considered to be failed.'''

    data_files = None
    '''List of path to data files that defines a snapshot, paths must be relative to the context directory.'''


    @classmethod
    def _reset(cls):
        """Reset class attributes settings, for unit testing purposes."""
        cls.private_directory = None
        cls.context    = None
        cls.command    = None
        cls.script     = None
        cls.timeout    = None
        cls.data_files = None
        cls.initialized = False


    @classmethod
    def initialize(cls, context, command, script, timeout, data_files,
                   private_directory):
        """Initialize the settings common to all objects."""
        if not os.path.isdir(context):
            raise ValueError('cannot find the context directory \'%s\'.'%(context))
        else:
            cls.context = clean_path(context)

        if not isinstance(private_directory, str):
            raise ValueError('private_directory must be a string.')
        else:
            cls.private_directory = private_directory

        if not isinstance(command, str):
            raise ValueError('command must be a string.')
        else:
            cls.command = command

        if not os.path.isfile(script):
            raise ValueError('cannot find script file.')
        else:
            cls.script = clean_path(script)

        try:
            timeout = float(timeout)
            if timeout <= 0:
                raise TypeError
        except TypeError:
            raise ValueError('period must be a positive number')
        else:
            cls.timeout = timeout

        bad = False
        if not isinstance(data_files, list):
            bad = True
        else:
            for f in data_files:
                if not isinstance(f, str):
                    bad = True
                    break
        if bad:
            raise ValueError('data files must be a list of strings.')
        else:
            cls.data_files = data_files

        cls.initialized = True


    def __init__(self, point, working_directory):
        """Create a task the will produce a snapshot.

        :param point: a point object.
        :param working_directory: path to the directory in which the task will be run.
        """
        cls = self.__class__

        # check the class has been initialized
        if not cls.initialized:
            raise Exception(cls.__name__ + ' class has not been initialized.')

        # check the Snapshot class has been initialized as we need its
        # point_filename attribute for writing a point to disk
        if not Snapshot.initialized:
            raise Exception('Snapshot class has not been initialized.')

        if not isinstance(working_directory, str):
            raise TypeError('working_directory must be a string')
        else:
            self.working_directory = clean_path(working_directory)

        if not isinstance(point, Point):
            raise TypeError('point must be a Point object.')
        else:
            self.point = point

        self.private_directory = opj(self.working_directory,
                                     cls.private_directory)
        self.started_file      = opj(self.private_directory, cls.started_file)
        self.finished_file     = opj(self.private_directory, cls.finished_file)
        self.script            = opj(self.private_directory, 
                                     os.path.basename(cls.script))

        # the task must be run from the working directory
        super(SnapshotTask, self).__init__(cls.command.split() + [self.script],
                                           cwd=self.working_directory)


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
                msg  = str(self) + '\n'
                msg += 'working directory already exists'
                raise TaskFailed(msg)

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
                msg = str(self) + '\n'
                if os.path.isfile(self.started_file):
                    msg += 'the job started but did not finish.'
                else:
                    msg += 'the job has not started.'
                raise TaskTimeoutError(msg)

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
                raise TaskFailed('missing data file : %s'%f)
            else:
                f_moved = opj(self.private_directory, os.path.basename(f))
                os.rename(f_original, f_moved)

        # clean up working directory but the private directory
        for f in os.listdir(self.working_directory):
            f = opj(self.working_directory, f)
            if f != self.private_directory:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

        return self.private_directory


    def __str__(self):
        s  = super(SnapshotTask, self).__str__() + '\n'
        s += 'point : ' + str(self.point) + '\n'
        s += 'context : ' + self.__class__.context + '\n'
        s += 'working directory : ' + self.working_directory
        return s


    def _state_files_hook(self):
        """Modify the script to create the state files."""
        cls = self.__class__

        # file creation command lines
        touch_started  = cls.touch + ' ' + self.started_file + '\n'
        touch_finished = cls.touch + ' ' + self.finished_file + '\n'

        # wrap with info strings
        touch_started  = cls.info_lines[0] + touch_started  + cls.info_lines[1]
        touch_finished = cls.info_lines[0] + touch_finished + cls.info_lines[1]

        # insert started state file creation
        lines = open(cls.script).readlines()
        for i, line in enumerate(lines):
            if not re.match(cls.dummy_line, line):
                lines.insert(i, touch_started)
                break

        # append finished state file creation
        lines.append(touch_finished)

        # create the hooked script
        f = open(self.script, 'w')
        for l in lines:
            f.write(l)
        f.close()


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
