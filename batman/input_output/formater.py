"""
Simple Inputs / Outputs.

This module provides several formater objects 
for reading and writing dataset with named fields.
The formaters are available from the :ref:`FORMATER` 
dictionary whose keys are format names.
"""
from collections import namedtuple
import os
import json
import numpy as np


# Formater interface:
# - np.ndarray read(filepath, varnames)
# - write(filepath, np.ndarray, varnames)
Formater = namedtuple('Formater', ['read', 'write'])


# JSON

def json_read(filepath, varnames):
    """Reader method for json file.
    
    :param str filepath: file to read.
    :param list varnames: names of variables to read.
    :return: a 2D numpy array.
    :rtype: numpy.ndarray
    """
    with open(filepath, 'r') as fd:
        data = json.load(fd)
    data = list(zip(*[np.ravel(data[var]) for var in varnames]))
    return np.array(data)
    

def json_write(filepath, dataset, varnames):
    """Write method for json file.
    
    :param str filepath: file to write.
    :param dataset: a 2D numpy array (n_sample, n_feature).
    :param list varnames: column names in dataset.
    """
    data = dict(zip(varnames, np.reshape(dataset, (-1, len(varnames))).T.tolist()))
    with open(filepath, 'w') as fd:
        json.dump(data, fd)


# CSV

def csv_read(filepath, varnames):
    """Reader method for csv file.
    
    :param str filepath: file to read.
    :param list varnames: names of variables to read.
    :return: a 2D numpy array.
    :rtype: numpy.ndarray
    """
    # 1st line of file is column names, can be a comment line
    data = np.genfromtxt(filepath, delimiter=',', names=True)[varnames]
    return np.array([list(d) for d in np.atleast_1d(data)])


def csv_write(filepath, dataset, varnames):
    """Write method for csv file.
    
    :param str filepath: file to write.
    :param dataset: a 2D numpy array (n_sample, n_feature).
    :param list varnames: column names in dataset.
    """
    data = np.reshape(dataset, (-1, len(varnames)))
    np.savetxt(filepath, data, delimiter=',', header=','.join(varnames))


# NUMPY BINARY (uncompressed)

def npy_read(filepath, varnames):
    """Reader method for numpy npy file.
    The uncompressed file contains exactly one dataset.
    
    :param str filepath: file to read.
    :param list varnames: names of variables to read.
    :return: a 2D numpy array.
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    filepath, _ = os.path.splitext(filepath)
    filepath += '.npy'
    data = np.load(filepath)
    return data.reshape(-1, len(varnames))


def npy_write(filepath, dataset, varnames):
    """Write method for numpy npy file.
    The uncompressed file contains exactly one dataset.
    
    :param str filepath: file to write.
    :param dataset: a 2D numpy array (n_sample, n_feature).
    :param list varnames: ignored.
    """
    # enforce .npy extension
    filepath, _ = os.path.splitext(filepath)
    filepath += '.npy'
    data = np.reshape(dataset, (-1, len(varnames)))
    np.save(filepath, data)


# NUMPY BINARY (compressed)

def npz_read(filepath, varnames):
    """Reader method for numpy npz file.
    The file may be compressed or not.
    file members are:
    - 'data': values to read as a 2D array
    - 'labels': names of data columns
    
    :param str filepath: file to read.
    :param list varnames: names of variables to read.
    :return: a 2D numpy array.
    :rtype: numpy.ndarray
    """
    # enforce .npy extension
    filepath, _ = os.path.splitext(filepath)
    filepath += '.npz'
    with np.load(filepath) as fd:
        label = fd['labels'].tolist()
        order = [label.index(v) for v in varnames]
        return fd['data'].reshape(-1, len(varnames))[:, order]

def npz_write(filepath, dataset, varnames):
    """Write method for numpy npz file.
    The file is compressed.
    
    :param str filepath: file to write.
    :param dataset: a 2D numpy array (n_sample, n_feature).
    :param list varnames: column names in dataset.
    """
    # enforce .npy extension
    filepath, _ = os.path.splitext(filepath)
    filepath += '.npz'
    data = np.reshape(dataset, (-1, len(varnames)))
    np.savez_compressed(filepath, data=data, labels=varnames)


# Available formater instances

FORMATER = {
    'json': Formater(read=json_read, write=json_write),  # file_extension='.json'),
    'csv': Formater(read=csv_read, write=csv_write),  # file_extension='.csv'),
    'npy': Formater(read=npy_read, write=npy_write),  # file_extension='.npy'),
    'npz': Formater(read=npz_read, write=npz_write),  # file_extension='.npz'),
}
