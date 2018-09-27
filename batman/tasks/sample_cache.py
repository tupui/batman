# coding: utf-8
"""
SampleCache class
=================

Manages computed samples for data providers.
Inherits from :class:`Sample`.
"""
import os
import glob
import itertools
import numpy as np
from ..space import Sample


class SampleCache(Sample):
    """Container with helper methods for handling computed snapshots."""

    def __init__(self, plabels, flabels,
                 psizes=None, fsizes=None, savedir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """Initialize an empty cache.

        :param list(str) plabels: parameter names (for space).
        :param list(str) flabels: feature names (for data).
        :param list(int) psizes: lengths of parameters (for space).
        :param list(int) fsizes: lengths of features (for data).
        :param str savedir: directory to save cache content.
        :param str space_fname: file name for space.
        :param str space_format: file format for space.
        :param str data_fname: file name for data.
        :param str data_format: file format for data.
        """
        self.savedir = savedir
        self.space_file = space_fname
        self.data_file = data_fname
        super(SampleCache, self).__init__(plabels=plabels, flabels=flabels,
                                          psizes=psizes, fsizes=fsizes,
                                          pformat=space_format, fformat=data_format)
        try:
            os.makedirs(savedir)
        except (OSError, TypeError, AttributeError):
            pass

    def discover(self, directory_pattern):
        """Search for sample file pairs and load their contents.

        :param str directory_pattern: directory name pattern.
            Supports UNIX shell wildcard expansion.
            See https://en.wikipedia.org/wiki/Glob_(programming)
        """
        dirpaths = (dirpath for dirpath in sorted(glob.iglob(directory_pattern))
                    if os.path.isdir(dirpath))
        for dirpath in dirpaths:
            space_file = os.path.join(dirpath, self.space_file)
            data_file = os.path.join(dirpath, self.data_file)
            try:
                self.read(space_fname=space_file, data_fname=data_file)
            except OSError:
                pass  # missing at least 1 of the 2 sample files

    def save(self, savedir=None):
        """Write samples to `self.savedir` directory.

        :param str savedir: directory to save sample and data.
        """
        if len(self) > 0 and self.savedir is not None:
            if savedir is None:
                savedir = self.savedir

            pname = os.path.join(savedir, self.space_file)
            fname = os.path.join(savedir, self.data_file)
            self.write(space_fname=pname, data_fname=fname)

    def locate(self, points):
        """Find points in known samples' space.

        Return position of points in cache.
        If a point is not known, return the position where it would be inserted.
        New points will always be appended after existing ones.

        :param array_like points: points to locate (n_points, n_features).
        :return: list of points positions in cache.
        :rtype: list(n_points)
        """
        idx = []
        new_idx = itertools.count(len(self))
        points = np.atleast_2d(points)
        for point in points:
            found = np.flatnonzero(np.all(point == self.space, axis=1))
            idx.append(found[0] if len(found) > 0 else next(new_idx))
        return np.asarray(idx)
