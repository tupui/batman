# coding: utf8
"""
Executor: remote computations
=============================

This executor perform remotelly the generation of the sample.
:class:`MasterRemoteExecutor`
"""
import logging
import os
import stat
import shutil
import tarfile
import getpass
import threading
import subprocess as sp
import itertools
import numpy as np
import paramiko
from paramiko.ssh_exception import (PasswordRequiredException,
                                    AuthenticationException,
                                    SSHException)
from batman.input_output import formater


class MasterRemoteExecutor:
    """Master Remote executor."""

    logger = logging.getLogger(__name__)

    def __init__(self, local_root, job, hosts):
        """Initialize master remote executor.

        Store one instance of :class:`RemoteExecutor` per host.
        An internal counter handle load balancing between hosts.

        :param str local_root: Local folder to create and store data.
        :param dict job: Parametrization of the jobs:

            - **local_root** (str) -- Local folder to create and store data.
            - **command** (str) -- command to be executed for computing new
              snapshots.
            - **context_directory** (str) -- store every ressource required for
              executing a job.
            - **coupling_directory** (str) -- sub-directory in
              ``context_directory`` that will contain input parameters and
              output file.
            - **input_fname** (str) -- basename for files storing the point
              coordinates ``plabels``.
            - **input_format** (str) -- ``json`` (default), ``csv``, ``npy``,
              ``npz``.
            - **input_labels** (list(str)) -- input parameter names.
            - **input_sizes** (list(int)) -- number of components of
              parameters.
            - **output_fname** (str) -- basename for files storing values
              associated to ``flabels``.
            - **output_format** (str) -- ``json`` (default), ``csv``, ``npy``,
              ``npz``.
            - **output_labels** (list(str)) -- output feature names.
            - **output_sizes** (list(int)) -- number of components of output
              features.
            - **clean** (bool) -- whether to remove working directories.

        :param list(dict) hosts: Parametrization of each host:

            - **hostname** (str) -- Remote host to connect to.
            - **remote_root** (str) -- Remote folder to create and store data.
            - **username** (str) -- username.
            - **password** (str) -- password.
        """
        self.n_hosts = len(hosts)
        if 'weight' in hosts[0]:
            self.load_balancing = np.array([host.pop('weight') for host in hosts])
            self.load_balancing = self.load_balancing / sum(self.load_balancing)
        else:
            self.load_balancing = itertools.cycle(range(self.n_hosts))

        self.hosts = []
        for host in hosts:
            host.update(job)
            self.hosts.append(RemoteExecutor(local_root=local_root, **host))

    def snapshot(self, point, sample_id):
        """Compute a snapshot remotelly.

        Depending on the internal counter, distribute the computation on hosts.

        :param array_like point: point to compute (n_features,).
        :param list sample_id: points indices in the points list.
        :return: concatenation of point and data for requested point
        :rtype: array_like
        """
        if isinstance(self.load_balancing, np.ndarray):
            host = np.random.choice(self.n_hosts, p=self.load_balancing)
        else:
            host = self.load_balancing.__next__()

        return self.hosts[host].snapshot(point, sample_id)


class RemoteExecutor:
    """Remote executor."""

    logger = logging.getLogger(__name__)

    def __init__(self, hostname, remote_root, local_root, command, context_directory,
                 coupling_directory, input_fname, input_format, input_labels, input_sizes,
                 output_fname, output_format, output_labels, output_sizes,
                 username=None, password=None, clean=False):
        """Initialize remote executor.

        It connects to the *host* and opens *ssh* and *sftp* connections.
        A working directory is created and it contains everything from setup
        to outputs.

        :param str hostname: Remote host to connect to.
        :param str remote_root: Remote folder to create and store data.
        :param str local_root: Local folder to create and store data.
        :param str command: command to be executed for computing new snapshots.
        :param str context_directory: store every ressource required for
          executing a job.
        :param str coupling_directory: sub-directory in
          ``context_directory`` that will contain input parameters and
          output file.
        :param str input_fname: basename for files storing the point
          coordinates ``plabels``.
        :param str input_format: ``json`` (default), ``csv``, ``npy``,
          ``npz``.
        :param list(str) input_labels: input parameter names.
        :param list(int) input_sizes: number of components of parameters.
        :param str output_fname: basename for files storing values
          associated to ``flabels``.
        :param str output_format: ``json`` (default), ``csv``, ``npy``,
          ``npz``.
        :param list(str) output_labels: output feature names.
        :param list(int) output_sizes: number of components of output features.
        :param str username: username.
        :param str password: password.
        :param bool clean: whether to remove working directories.
        """
        # config
        ssh_config = paramiko.SSHConfig()
        try:
            with open(os.path.expanduser('~/.ssh/config'), 'r') as fd:
                ssh_config.parse(fd)
        except (IOError, OSError):
            pass
        user_config = ssh_config.lookup(hostname)
        args = {}
        for key in ['hostname', 'username', 'password', 'port']:
            if key in user_config:
                args[key] = user_config[key]

        if username is not None:
            args['username'] = username
        if password is not None:
            args['password'] = password
        if 'proxycommand' in user_config:
            del args['proxycommand']
            args['sock'] = paramiko.ProxyCommand(user_config['proxycommand'])
        hostname = args['hostname']
        username = username or getpass.getuser()

        args['key_filename'] = user_config.get('identityfile')

        # connection
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(**args)
        except (PasswordRequiredException, SSHException):
            prompt = "{}@{}'s password: ".format(username, hostname)
            args['allow_agent'] = False
            args['timeout'] = 3
            for _ in range(3):
                args['password'] = getpass.getpass(prompt=prompt)
                try:
                    self.ssh.connect(**args)
                except (AuthenticationException, SSHException):
                    print('Permission denied, please try again.')
                    continue
                break
            else:
                args['password'] = getpass.getpass(prompt=prompt)
                try:
                    self.ssh.connect(**args)
                except (AuthenticationException, SSHException):
                    raise SystemExit('Permission denied.')
        self.sftp = self.ssh.open_sftp()

        self.logger.info('Connected to remote host: {}'.format(hostname))

        # working directories
        self.exec_remote('mkdir -p {}'.format(remote_root))
        self._wkroot = self.sftp.normalize(remote_root)
        self._lwkroot = os.path.abspath(local_root)

        # context directories
        self.logger.info('Compressing and sending context directory...')
        self._lcontext = context_directory
        self._context = os.path.join(self._wkroot, 'context')
        tarname = os.path.join(self._lwkroot, 'context.tar.gz')
        with tarfile.open(tarname, 'w:gz') as tar:
            tar.add(self._lcontext, arcname='context')
        self.sftp.put(tarname, os.path.join(self._wkroot, 'context.tar.gz'))
        self.logger.debug('Context directory sent')

        self.logger.debug('Uncompressing context directory...')
        self.exec_remote('cd {} && tar xzf context.tar.gz'
                         ' && rm context.tar.gz'.format(self._wkroot))
        self.logger.info('Context directory on remote host')

        # job
        self._cmd = command
        self._coupling = coupling_directory
        self._input_file = input_fname
        self._input_sizes = input_sizes
        self._input_labels = input_labels
        self._input_formater = formater(input_format)
        self._output_file = output_fname
        self._output_sizes = output_sizes
        self._output_labels = output_labels
        self._output_formater = formater(output_format)
        self._clean = clean

        # threadsafety: SFTP session seems not to be threadsafe
        self._lock = threading.Lock()

    def __dell__(self):
        """Close both ssh and sftp connections."""
        if self._clean:
            self.exec_remote('rm -r {}'.format(self._wkroot))
        self.sftp.close()
        self.ssh.close()

    def exec_remote(self, cmd):
        """Execute a command on the HOST and check its completion."""
        _, cmd_out, cmd_err = self.ssh.exec_command(cmd)
        ret = cmd_err.channel.recv_exit_status()
        if ret != 0:
            raise sp.CalledProcessError(ret, cmd,
                                        output=cmd_out.read().decode(),
                                        stderr=cmd_err.read().decode())
        return cmd_out

    def walk(self, directory):
        """Directory and file generator.

        :param str directory: dir to list from.
        :return: directory, its sub-directories and files.
        :rtype: tuple(str)
        """
        content = self.sftp.listdir_attr(directory)
        files = [attr.filename for attr in content if not stat.S_ISDIR(attr.st_mode)]
        dirs = [attr.filename for attr in content if stat.S_ISDIR(attr.st_mode)]
        yield (directory, dirs, files)
        for d in dirs:
            for x in self.walk(os.path.join(directory, d)):
                yield x

    def snapshot(self, point, sample_id):
        """Compute a snapshot remotelly.

        :param array_like point: point to compute (n_features,).
        :param list sample_id: points indices in the points list.
        :return: concatenation of point and data for requested point
        :rtype: array_like
        """
        # build input file
        self.logger.debug('Building snapshot directory')
        lsnapdir = os.path.join(self._lwkroot, str(sample_id))
        os.makedirs(lsnapdir)
        infile = os.path.join(lsnapdir, self._input_file)
        self._input_formater.write(infile, point, self._input_labels, self._input_sizes)
        snapdir = os.path.join(self._wkroot, os.path.basename(lsnapdir))
        cpldir = os.path.join(snapdir, self._coupling)
        with self._lock:
            self.sftp.mkdir(snapdir)
            self.sftp.mkdir(cpldir)
            self.sftp.put(infile, os.path.join(cpldir, self._input_file))

            # link context
            for root, dirs, files in self.walk(self._context):
                local = root.replace(self._context, snapdir)
                for d in dirs:
                    self.sftp.mkdir(os.path.join(local, d))
                for f in files:
                    self.sftp.symlink(os.path.join(root, f), os.path.join(local, f))

        # execute command
        self.logger.debug('Executing command on remote host')
        self.exec_remote('cd {} && {}'.format(snapdir, self._cmd))

        # get result
        self.logger.debug('Getting results')
        with self._lock:
            outfile = os.path.join(lsnapdir, self._output_file)
            self.sftp.get(os.path.join(cpldir, self._output_file), outfile)
            data = self._output_formater.read(outfile, self._output_labels)

        # cleaning
        if self._clean:
            shutil.rmtree(lsnapdir)
            self.exec_remote('rm -r {}'.format(snapdir))

        return np.append(point, data)
