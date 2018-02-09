# coding: utf-8
"""
[TODO]
"""
import os
import glob
import itertools
import numpy as np
from ..space import Sample


class SampleCache(Sample):
    """[TODO]"""


    def __init__(self, plabels, flabels,
                 psizes=None, fsizes=None, savedir=None,
                 space_file='sample-space.json',
                 space_format='json',
                 data_file='sample-data.json',
                 data_format='json'):
        """[TODO]
        """
        self.savedir = savedir
        self.space_file = space_file
        self.data_file = data_file
        super().__init__(plabels=plabels, flabels=flabels,
                         psizes=psizes, fsizes=fsizes,
                         pformat=space_format, fformat=data_format)
        try:
            os.makedirs(savedir)
        except (OSError, TypeError):
            pass

    def discover(self, directory_pattern):
        """[TODO]"""
        for dirpath in glob.iglob(directory_pattern):
            space_file = os.path.join(dirpath, self.space_file)
            data_file = os.path.join(dirpath, self.data_file)
            try:
                self.read(space_fname=space_file, data_fname=data_file)
            except OSError:
                pass  # missing at least 1 of the 2 sample files
        
    def save(self):
        """[TODO]"""
        if len(self) > 0 and self.savedir is not None:
            pname = os.path.join(self.savedir, self.space_file)
            fname = os.path.join(self.savedir, self.data_file)
            self.write(space_fname=pname, data_fname=fname)

    def locate(self, points):
        """[TODO]"""
        idx = []
        new_idx = itertools.count(len(self))
        points = np.atleast_2d(points)
        nsample = len(points)
        points = points.reshape(nsample, -1)
        for point in points:
            found = np.flatnonzero(np.all(point == self.space, axis=1))
            idx.append(found[0] if found else next(new_idx))
        return np.asarray(idx)

