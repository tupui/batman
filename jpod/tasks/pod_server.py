import os
import re
import logging
from .snapshot import SnapshotTask
import rpyc
import socket
import numpy as N


class PodServerTask(SnapshotTask):
    """Task subclass to manage the pod server."""


    def __init__(self, port, python, script, directory):
        self.proxy = None
        '''Proxy used for shutting down the server.'''

        self.port = port
        '''Port to use for connection.'''

        self.connection = None
        '''Rpyc connection handle.'''

        # check output directory
        if not os.path.isdir(directory):
            raise ValueError('no such directory : ' + directory)

        # start server task
        dirname = os.path.dirname(__file__)
        pod_server_path = os.path.join(dirname, '..', 'pod_server.py')
        pod_server_path = os.path.normpath(pod_server_path)
        pod_server_path = os.path.abspath(pod_server_path)
        command = python.split() + [pod_server_path, str(port), script]
        super(PodServerTask, self).__init__(command, cwd=directory)


    def _after_run(self):
        """Wait for the server to be up."""
        while True: # TODO: timeout
            try:
                channel = rpyc.Channel(rpyc.SocketStream.connect(
                                       'localhost', self.port, timeout = 3.))
                self.connection = rpyc.Connection(rpyc.VoidService, channel,
                                                  config = {'allow_pickle' : True,
                                                            'allow_public_attrs' : True})
            except socket.error:
                # server not yet running
                pass
            else:
                try:
                    self.proxy = self.connection.root
                except EOFError:
                    # busy socket
                    pass
                else:
                    return


    def _before_terminate(self):
        """Shutdown the server."""
        if self.connection is not None:
            self.connection.close()
