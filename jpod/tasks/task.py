import os
import logging
import subprocess


class Task(object):
    """A class to handle subprocesses.

    A subclass of subprocess.Popen for which additional things can be performed before and after both running and terminating the process.
    """

    logger = logging.getLogger(__name__)


    def __init__(self, args, **kwargs):
        self.args   = args
        '''Command line to be executed.'''

        self.kwargs = kwargs
        '''Keyword arguments for subprocess.Popen.'''

        # TODO: do we want to be able to capture stdout and stderr?
        # self.kwargs.update({
        #     'stdout' : subprocess.PIPE,
        #     'stderr' : subprocess.PIPE})

        self.handle = None
        '''Subprocess handle.'''


    def __str__(self):
        s  = 'command line : '
        s += ' '.join(self.args) + '\n'
        s += 'run form : '
        s += self.kwargs.get('cwd', os.getcwd())
        return s


    def _before_run(self):
        """Method to be run right before executing run.

        It must return None if the task has not been run, otherwise whatever _after_run is supposed to return.
        """
        pass


    def run(self):
        """Return the result of the task.

        If the task has already been completed, just return its result, otherwise try to run it first.
        """
        result = self._before_run()

        # check if the task has already been run
        if result is None:
            self.handle = subprocess.Popen(self.args, **self.kwargs)
            self.logger.info('Launched : %s', ' '.join(self.args))
            result = self._after_run()
        else:
            self.logger.info('Task is up to date and was not run.')

        return result


    def _after_run(self):
        """Method to be run a after executing run."""
        pass


    def _before_terminate(self):
        """Method to be run right before executing terminate."""
        pass


    def terminate(self):
        """Terminate the task."""
        self._before_terminate()
        self.handle.terminate()
        self.logger.info('Terminated : %s', ' '.join(self.args))
        self._after_terminate()


    def _after_terminate(self):
        """Method to be run a after executing terminate."""
        pass
