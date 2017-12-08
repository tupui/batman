"""
Define numpy IO
***************
"""
import numpy as np
from .base import *


class Npz(IOBase):
    """Manages IO for numpy npz files.

    See numpy documentation for details.
    """

    format = 'numpy'
    extension = '.npz'

    @use_base_class_docstring
    def read(self, path, names=None):
        lazy_data = np.load(path)
        self.info.set_names(lazy_data.files)

        def iteritems(lazy_data):
            for f in lazy_data.files:
                yield (f, lazy_data[f])
        return super(Npz, self)._read(iteritems(lazy_data), names)

    @use_base_class_docstring
    def write(self, path, dataset):
        data_map = dict(zip(dataset.names, dataset.data))
        np.savez(path, **data_map)

    def meta_data(self, path):
        # make this more subtle
        d = self.read(path)
        self.info.set_shape(d.shape)
