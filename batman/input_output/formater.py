"""
Built-in Inputs / Outputs
*************************

This module provides several formater objects
for reading and writing dataset with named fields.

File formats store variable names whenever it is possible.

The formaters are available from the :ref:`FORMATER`
dictionary whose keys are format names.
"""
from collections import namedtuple
import os
import json
import numpy as np


# Formater interface:
# - np.ndarray read(fname, varnames)
# - write(fname, np.ndarray, varnames)
Formater = namedtuple('Formater', ['read', 'write'])


# JSON

def json_read(fname, varnames):
    """Reader method for json file.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, n_variable).
    :rtype: numpy.ndarray
    """
    with open(fname, 'r') as fd:
        data = json.load(fd)
    data = list(zip(*[np.ravel(data[var]) for var in varnames]))
    return np.array(data)


def json_write(fname, dataset, varnames):
    """Write method for json file.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, n_variable).
    :param list(str) varnames: column names in dataset.
    """
    data = dict(zip(varnames, np.reshape(dataset, (-1, len(varnames))).T.tolist()))
    with open(fname, 'w') as fd:
        json.dump(data, fd)


# CSV

def csv_read(fname, varnames):
    """Reader method for csv file.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, n_variable).
    :rtype: numpy.ndarray
    """
    # 1st line of file is column names, can be a comment line
    data = np.genfromtxt(fname, delimiter=',', names=True)[varnames]
    return np.array([list(d) for d in np.atleast_1d(data)])


def csv_write(fname, dataset, varnames):
    """Write method for csv file.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, n_variable).
    :param list(str) varnames: column names in dataset.
    """
    data = np.reshape(dataset, (-1, len(varnames)))
    np.savetxt(fname, data, delimiter=',', header=','.join(varnames))


# NUMPY BINARY (uncompressed)

def npy_read(fname, varnames):
    """Reader method for numpy npy file.
    The uncompressed file contains exactly one dataset.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, n_variable).
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npy'
    data = np.load(fname)
    return data.reshape(-1, len(varnames))


def npy_write(fname, dataset, varnames):
    """Write method for numpy npy file.
    The uncompressed file contains exactly one dataset.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, n_variable).
    :param list(str) varnames: column names in dataset.
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npy'
    data = np.reshape(dataset, (-1, len(varnames)))
    np.save(fname, data)


# NUMPY BINARY (compressed)

def npz_read(fname, varnames):
    """Reader method for numpy npz file.
    The file may be compressed or not.
    file members are:
    - 'data': values to read as a 2D array
    - 'labels': names of data columns

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, n_variable).
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npz'
    with np.load(fname) as fd:
        label = fd['labels'].tolist()
        order = [label.index(v) for v in varnames]
        return fd['data'].reshape(-1, len(label))[:, order]


def npz_write(fname, dataset, varnames):
    """Write method for numpy npz file.
    The file is compressed.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, n_variable).
    :param list(str) varnames: column names in dataset.
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npz'
    data = np.reshape(dataset, (-1, len(varnames)))
    np.savez_compressed(fname, data=data, labels=varnames)


# Available formater instances

FORMATER = {
    'json': Formater(read=json_read, write=json_write),
    'csv': Formater(read=csv_read, write=csv_write),
    'npy': Formater(read=npy_read, write=npy_write),
    'npz': Formater(read=npz_read, write=npz_write),
}
