import os
import numpy as N
from dataset import DatasetInfo, Dataset, ShapeError, DataSizeError

__docformat__ = "reStructuredText"


class FormatError(Exception):
    """This exception can be raised in sub-classes."""
    pass


class IOBase(object):
    """Base class for IO operations.

    This class is an abstract class that implements the basic algorithms and common attributes used in actual io operations implemented in the sub-classes.
    The instance attribute `info` is used internally to share dataset informations with the subclasses.
    The class attributes `format` and `extension` must be overriden in the sub-classes.
    In practice, a sub-class passes an iterable object to :func:`_read`. For each quantity (see dataset) contained in a data file, this iterable must provide the name and the data array bound to a quantity.
    The class :class:`IOBase` will select the required quantities and create the proper dataset.
    The functions `read` and `write` are only placeholder, also used for providing docstring to subclasses.
    """


    format = None
    '''file format.'''

    extension = None
    '''file name extension.'''


    def __init__(self):
        self.info = DatasetInfo()
        '''Dataset meta-data.'''


    def _read(self, iterable, names):
        """Same as `read()` with an iterable instead of a file path."""
        # process names to be read
        if names is None:
            names = self.info.names
        else:
            for v in names:
                if v not in self.info.names:
                    msg = 'no "%s" variable'%v
                    raise NameError(msg)

        # first build up names to data map
        data_map = {}
        for v,d in iterable:
            if v in names:
                data_map[v] = d
                if len(data_map) == len(names):
                    break

        # next build up data array in expected order
        # TODO: check memory usage and layout of data
        data = None
        for v in names:
            d = data_map[v][N.newaxis,...]
            if data is None:
                data = d
            else:
                data = N.vstack((data, d))

        return Dataset(names=names, data=data)


    def read(self, path, names=None):
        """Read the quantities defined by `names` from `path` and return them in a :class:`Dataset`.

        If names is not given or is `None`, then all the quantities will be returned.
        """
        raise NotImplementedError()


    def write(self, path, dataset):
        """Write `dataset` to `path`."""
        raise NotImplementedError()




def use_base_class_docstring(func):
    """Set the docstring of `func` as the one of IOBase.func."""
    func.__doc__ = IOBase.__dict__[func.__name__].__doc__
    return func
