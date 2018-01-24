"""
Wrappers to Antares I/O classes.

This module exposes Antares Readers and Writers as additionnal Batman formaters.
"""
from collections import defaultdict
import numpy as np


class AntaresFormater(object):
    """Wrapper class for using antares reader and writers."""

    def __init__(self, format_name):
        """Initialize a wrapper.
        
        :param str format_name: name of an antares format.
        """
        self._format = format_name

    def read(self, filepath, varnames):
        """Read a dataset from a file using an antares Reader.

        :param str filepath: file to read.
        :param list varnames: names of variables to read.
        :returns: a numpy structured array with variables names as field names.
        :rtype: numpy.ndarray
        """
        # read file as an antares base
        import antares
        reader = antares.Reader(self._format)
        reader['filename'] = filepath
        base = reader.read()

        # extract content:
        # - 1 named field per variable
        # - 1 sample per point in time/space
        data = defaultdict(lambda : np.empty(0, dtype=float))
        for zkey in base:
            for ikey in base[zkey]:
                for var, loc in base[zkey][ikey]:
                    data[var] = np.append(data[var], base[zkey][ikey][(var, loc)].flat)

        # return numpy 2D array
        data = zip(*[data[var] for var in varnames])
        return np.array(data)

    def write(self, filepath, dataset, varnames):
        """Write a dataset to a file using an antares Writer.

        :param str filepath: file to write.
        :param dataset: a numpy structured array with named fields.
        :param list varnames: column names in dataset.
        """
        # build a mono-zone/mono-instant base
        import antares
        base = antares.Base()
        base.init()
        for i, var in  enumerate(varnames):
            base[0][0][var] = dataset[:, i]

        # use antares writer
        writer = antares.Writer(self._format)
        writer['base'] = base
        writer['filename'] = filepath
        writer.dump()


# Discover and build antares formaters

try:
    import antares
except ImportError:
    ANTARES_FORMATER = {}
else:
    all_fmt = set(antares.reader_pool.format2reader) | set(antares.writer_pool.format2reader)
    ANTARES_FORMATER = dict([('antares_' + fmt, AntaresFormater(fmt)) for fmt in all_fmt])
