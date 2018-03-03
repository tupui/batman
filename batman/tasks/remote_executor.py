"""
[TODO] ssh
"""
import os
import stat
import shutil
import tarfile
import getpass
import threading
import numpy as np
import subprocess as sp
import paramiko
from paramiko.ssh_exception import (PasswordRequiredException,
                                    AuthenticationException,
                                    SSHException)
from batman.input_output import formater


class RemoteExecutor:
    """[TODO]"""

    def __init__(self, hostname, remote_root, local_root, command, context_directory,
                 coupling_directory, input_filename, input_format, input_labels, input_sizes,
                 output_filename, output_format, output_labels, output_sizes,
                 username=None, password=None, clean=True):
        """[TODO]
        """
        # config
        ssh_config = paramiko.SSHConfig()
        try:
            with open(os.path.expanduser('~/.ssh/config'), 'r') as fd:
                ssh_config.parse(fd)
        except (IOError, OSError):
            pass
        user_config = ssh_config.lookup(hostname)
        args = user_config.copy()
        if username is not None:
            args['username'] = username
        if password is not None:
            args['password'] = password
        if 'proxycommand' in user_config:
            del args['proxycommand']
            args['sock'] = paramiko.ProxyCommand(user_config['proxycommand'])
        hostname = args['hostname']
        username = username or getpass.getuser()

        # connection
        self.ssh = paramiko.SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(**args)
        except PasswordRequiredException:
            prompt = "{}@{}'s password: ".format(username, hostname)
            args['allow_agent'] = False
            args['timeout'] = 3
            for i in range(2):
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
                    print('Permission denied. You sucks !')
                    raise SystemExit
        self.sftp = self.ssh.open_sftp()

        # work directories
        out = self.exec('mkdir -p {}; mktemp -d --tmpdir="{}" bat.XXXXXXXX'
                        .format(remote_root, remote_root))
        wkroot = out.readlines()[-1].strip()
        self._wkroot = self.sftp.normalize(wkroot)
        self._lwkroot = os.path.abspath(local_root)

        # context directories
        self._lcontext = context_directory
        self._context = os.path.join(self._wkroot, 'context')
        tarname = os.path.join(self._lwkroot, 'context.tar.gz')
        with tarfile.open(tarname, 'w:gz') as tar:
            tar.add(self._lcontext, arcname='context')
        self.sftp.put(tarname, os.path.join(self._wkroot, 'context.tar.gz'))
        self.exec('cd {} && tar xaf context.tar.gz && rm context.tar.gz'.format(self._wkroot))

        # job
        self._cmd = command
        self._coupling = coupling_directory
        self._input_file = input_filename
        self._input_sizes = input_sizes
        self._input_labels = input_labels
        self._input_formater = formater(input_format)
        self._output_file = output_filename
        self._output_sizes = output_sizes
        self._output_labels = output_labels
        self._output_formater = formater(output_format)
        self._clean = clean

        # threadsafety: SFTP session seems not to be threadsafe
        self._lock = threading.Lock()

    def close(self):
        """[TODO]"""
        if self._clean:
            self.exec('rm -r {}'.format(self._wkroot))
        self.sftp.close()
        self.ssh.close()

    def exec(self, cmd):
        """[TODO]"""
        _, cmd_out, cmd_err = self.ssh.exec_command(cmd)
        ret = cmd_err.channel.recv_exit_status()
        if ret != 0:
            raise sp.CalledProcessError(ret, cmd,
                                        output=cmd_out.read().decode(),
                                        stderr=cmd_err.read().decode())
        return cmd_out

    def walk(self, directory):
        """[TODO] generator"""
        content = self.sftp.listdir_attr(directory)
        files = [attr.filename for attr in content if not stat.S_ISDIR(attr.st_mode)]
        dirs = [attr.filename for attr in content if stat.S_ISDIR(attr.st_mode)]
        yield (directory, dirs, files)
        for d in dirs:
            self.walk(os.path.join(directory, d))

    def snapshot(self, point, sample_id):
        """[TODO]"""
        # build input file
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
        self.exec('cd {} && {}'.format(snapdir, self._cmd))

        # get result
        with self._lock:
            outfile = os.path.join(lsnapdir, self._output_file)
            self.sftp.get(os.path.join(cpldir, self._output_file), outfile)
            data = self._output_formater.read(outfile, self._output_labels)

        # cleaning
        if self._clean:
            shutil.rmtree(lsnapdir)
            self.exec('rm -r {}'.format(snapdir))

        return np.append(point, data)

