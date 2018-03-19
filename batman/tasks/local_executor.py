# coding: utf-8
"""
Executor: Local computations
============================

This executor perform locally the generation of the sample.
"""
import os
import shutil
import subprocess as sp
import numpy as np
from batman.input_output import formater


class LocalExecutor:
    """Local exectuor."""

    def __init__(self, local_root, command, context_directory,
                 coupling_directory, input_fname, input_format, input_labels, input_sizes,
                 output_fname, output_format, output_labels, output_sizes, clean=False):
        """Initialize local executor.

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
        :param bool clean: whether to remove working directories.
        """
        # work directories
        try:
            os.makedirs(local_root)
        except OSError:
            pass
        self._wkroot = local_root

        # context directories
        self._context = os.path.abspath(context_directory)

        # job
        self._cmd = command.split()
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

    def snapshot(self, point, sample_id):
        """Compute a snapshot locally.

        :param array_like point: point to compute (n_features,).
        :param list sample_id: points indices in the points list.
        :return: concatenation of point and data for requested point
        :rtype: array_like
        """
        # build input file
        snapdir = os.path.join(self._wkroot, str(sample_id))
        cpldir = os.path.join(snapdir, self._coupling)
        os.makedirs(cpldir)
        infile = os.path.join(cpldir, self._input_file)
        self._input_formater.write(infile, point, self._input_labels, self._input_sizes)

        # link context
        for root, dirs, files in os.walk(self._context):
            local = root.replace(self._context, snapdir)
            for d in dirs:
                os.makedirs(os.path.join(local, d))
            for f in files:
                os.symlink(os.path.join(root, f), os.path.join(local, f))

        # execute command
        job = sp.Popen(self._cmd, cwd=snapdir, stdout=sp.PIPE, stderr=sp.PIPE)
        out, err = job.communicate()
        ret = job.wait()
        if ret != 0:
            raise sp.CalledProcessError(ret, ' '.join(self._cmd),
                                        output=out.decode(),
                                        stderr=err.decode())

        # get result
        outfile = os.path.join(cpldir, self._output_file)
        data = self._output_formater.read(outfile, self._output_labels)

        # cleaning
        if self._clean:
            shutil.rmtree(snapdir)

        return np.append(point, data)
