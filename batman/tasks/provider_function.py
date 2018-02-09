# coding: utf-8
"""
[TODO]
"""
from copy import copy
import os
import sys
import logging
import importlib
import numpy as np
from .sample_cache import SampleCache
from ..space import Sample


class ProviderFunction(object):
    """[TODO]
    """

    logger = logging.getLogger(__name__)

    def __init__(self, plabels, flabels, module, function,
                 psizes=None, fsizes=None,
                 discover_pattern=None, save_dir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """Initialize the provider.

        :param list(str) plabels: input parameter names.
        :param list(str) flabels: output feature names.
        :param str module: python module to load.
        :param str function: function in `module` to execute for generating data.
        :param list(int) psizes: number of components of parameters.
        :param list(int) fsizes: number of components of output features.
        :param str discover_pattern: UNIX-style patterns for directories with pairs
            of sample files to import.
        :param str save_dir: path to a directory for saving computed results.
        :param str space_fname: name of space file to write.
        :param str data_fname: name of data file to write.
        :param str space_format: space file format.
        :param str data_format: data file format.
        """
        # import python function
        sys.path.append(os.path.abspath('.'))
        plugin = importlib.import_module(module)
        self._function = getattr(plugin, function)

        # discover existing snapshots
        self._cache = SampleCache(plabels, flabels, psizes, fsizes, save_dir,
                                  space_fname, space_format, 
                                  data_fname, data_format)
        if discover_pattern:
            self._cache.discover(discover_pattern)
            self._cache.save()

    @property
    def plabels(self):
        """[TODO]"""
        return self._cache.plabels

    @property
    def flabels(self):
        """[TODO]"""
        return self._cache.flabels

    @property
    def psizes(self):
        """[TODO]"""
        return self._cache.psizes

    @property
    def fsizes(self):
        """[TODO]"""
        return self._cache.fsizes

    @property
    def known_points(self):
        """[TODO]"""
        return self._cache.space

    def get_data(self, points):
        """[TODO]"""

        # locate results in cache
        idx = self._cache.locate(points)

        # build missing results
        new_points = points[idx >= len(self._cache)]
        if len(new_points) > 0:
            self._cache += self.build_data(new_points)
            self._cache.save()

        return self._cache[idx]

    def build_data(self, points):
        """[TODO]"""
        self.logger.debug('Build Snapshot for points {}'.format(points))
        points = np.atleast_2d(points)
        result = [self._function(point) for point in points]
        sample = Sample(space=points, data=result, plabels=self.plabels, flabels=self.flabels,
                        psizes=self.psizes, fsizes=self.fsizes) 
        return sample
