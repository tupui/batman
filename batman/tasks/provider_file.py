# coding: utf-8
"""
[TODO]
"""
from copy import copy
import logging
import numpy as np
from .sample_cache import SampleCache
from ..input_output import formater


class ProviderFile(object):
    """[TODO]
    """

    logger = logging.getLogger(__name__)

    def __init__(self, plabels, flabels, file_pairs,
                 psizes=None, fsizes=None,
                 discover_pattern=None, save_dir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """[TODO]
        """
        self._cache = SampleCache(plabels, flabels, psizes, fsizes, save_dir, 
                                  space_fname, space_format, 
                                  data_fname, data_format)

        # load provided files
        for space_fname, data_fname in file_pairs:
            self._cache.read(space_fname, data_fname, plabels, flabels)
        # discover additionnal files
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
        if np.any(idx >= len(self._cache)):
            logger.error("Data cannot be provided for requested points: {}"
                         .format(points[idx >= len(self._cache)]))
            raise ValueError()

        return self._cache[idx]

    def build_data(self, points):
        """[TODO]"""
        return NotImplemented
