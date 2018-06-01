"""
Built-in Inputs / Outputs
*************************

This module provides several formater objects
for reading and writing dataset with named fields.

File formats store variables' name whenever it is possible.
If no variables' name are provided, it will try to load them all.

The formaters are available from the :ref:`FORMATER`
dictionary whose keys are format names.
"""
from collections import namedtuple
import os
import json
import numpy as np


# Formater interface:
# - np.ndarray read(fname, varnames)
# - write(fname, np.ndarray, varnames, sizes=None)
Formater = namedtuple('Formater', ['read', 'write'])


# JSON

def json_read(fname, varnames=None):
    """Reader method for json file.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, sum(sizes)).
    :rtype: numpy.ndarray
    """
    with open(fname, 'r') as fd:
        data = json.load(fd)
    dataset = []

    varnames = data.keys() if varnames is None else varnames

    for var in varnames:
        dset = np.atleast_1d(data[var])
        dset = dset.reshape(len(dset), -1)
        dataset.append(dset)
    dataset = np.concatenate(dataset, axis=1)
    return dataset


def json_write(fname, dataset, varnames, sizes=None):
    """Write method for json file.

    :param str fname: file to write.
    :param array-like dataset: an array of shape (n_entry, sum(sizes)).
    :param list(str) varnames: column names in dataset.
    :param list(int) sizes: size of each variable
    """
    if sizes is None:
        sizes = [1] * len(varnames)
    nsample = np.atleast_2d(dataset).shape[0]
    dataset = np.reshape(dataset, (nsample, sum(sizes)))
    offsets = np.append(0, np.cumsum(sizes)[:-1])

    data = {}
    for var, start, size in zip(varnames, offsets, sizes):
        data[var] = dataset[:, start:start+size].reshape(-1, size).tolist()
    with open(fname, 'w') as fd:
        json.dump(data, fd)


# CSV

def csv_read(fname, varnames=None):
    """Reader method for csv file.

    Header consists in the varnames and footer of their respective sizes.
    Both header and footer must start with `#` and be separated by comma.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, sum(sizes)).
    :rtype: numpy.ndarray
    """
    with open(fname, 'r') as fd:
        lines = fd.readlines()
    names = lines[0].strip('# \n').replace(' ', '').split(',')
    sizes = list(map(int, lines[-1].lstrip('#').split(',')))
    offsets = np.append(0, np.cumsum(sizes)[:-1])
    nsample = len(lines) - 2
    data = np.genfromtxt(fname, delimiter=',').reshape(nsample, sum(sizes))

    varnames = names if varnames is None else varnames

    index = [names.index(var) for var in varnames]
    offsets = [offsets[i] for i in index]
    sizes = [sizes[i] for i in index]
    dataset = [data[:, start:start+size] for start, size in zip(offsets, sizes)]
    dataset = np.concatenate(dataset, axis=1)
    return dataset


def csv_write(fname, dataset, varnames, sizes=None):
    """Write method for csv file.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, sum(sizes)).
    :param list(str) varnames: column names in dataset.
    :param list(int) sizes: size of each variable
    """
    if sizes is None:
        sizes = [1] * len(varnames)
    nsample = np.atleast_2d(dataset).shape[0]
    dataset = np.reshape(dataset, (nsample, sum(sizes)))
    np.savetxt(fname, dataset, delimiter=',', comments='#',
               header=','.join(varnames),
               footer=','.join(map(str, sizes)))


# NUMPY BINARY (uncompressed)

def npy_read(fname, varnames=None):
    """Reader method for numpy npy file.
    The uncompressed file contains exactly one dataset.

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, sum(sizes)).
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npy'
    data = np.load(fname)
    dataset = data.reshape(data.shape[0], -1)
    return dataset


def npy_write(fname, dataset, varnames, sizes=None):
    """Write method for numpy npy file.

    The uncompressed file contains exactly one dataset.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, sum(sizes)).
    :param list(str) varnames: column names in dataset.
    :param list(int) sizes: size of each variable
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npy'
    if sizes is None:
        sizes = [1] * len(varnames)
    nsample = np.atleast_2d(dataset).shape[0]
    dataset = np.reshape(dataset, (nsample, sum(sizes)))
    np.save(fname, dataset)


# NUMPY BINARY (compressed)

def npz_read(fname, varnames=None):
    """Reader method for numpy npz file.

    The file may be compressed or not.
    file members are:
    - 'data': values to read as a 2D array
    - 'labels': names of data columns

    :param str fname: file to read.
    :param list(str) varnames: names of variables to read.
    :return: a 2D array with shape (n_entry, sum(sizes)).
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npz'
    with np.load(fname) as fd:
        labels = fd['labels'].tolist()
        sizes = fd['sizes'].tolist()
        data = fd['data']
        nsample = len(data)
        data = data.reshape(nsample, sum(sizes))
    offsets = np.append(0, np.cumsum(sizes)[:-1])
    index = [labels.index(v) for v in varnames] if varnames else range(len(labels))

    sizes = [sizes[i] for i in index]
    offsets = [offsets[i] for i in index]
    dataset = [data[:, start:start+size] for start, size in zip(offsets, sizes)]
    dataset = np.concatenate(dataset, axis=1)
    return dataset


def npz_write(fname, dataset, varnames, sizes=None):
    """Write method for numpy npz file.

    The file is compressed.

    :param str fname: file to write.
    :param array-like dataset: a 2D array of shape (n_entry, sum(sizes)).
    :param list(str) varnames: column names in dataset.
    :param list(int) sizes: size of each variable
    """
    # enforce .npy extension
    fname, _ = os.path.splitext(fname)
    fname += '.npz'
    if sizes is None:
        sizes = [1] * len(varnames)
    nsample = np.atleast_2d(dataset).shape[0]
    dataset = np.reshape(dataset, (nsample, sum(sizes)))
    np.savez_compressed(fname, data=dataset, labels=varnames, sizes=sizes)


# Available formater instances

FORMATER = {
    'json': Formater(read=json_read, write=json_write),
    'csv': Formater(read=csv_read, write=csv_write),
    'npy': Formater(read=npy_read, write=npy_write),
    'npz': Formater(read=npz_read, write=npz_write),
}
