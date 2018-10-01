# coding: utf-8
"""
Data Provider: Read snapshots from files
========================================

This provider gets its data from a list of files.

It is unable to generate data and it will raise an error if an unknown
point is requested.
"""
import logging
import numpy as np
from .sample_cache import SampleCache


class ProviderFile:
    """Provides Snapshots loaded from a list of files."""

    logger = logging.getLogger(__name__)

    def __init__(self, plabels, flabels, file_pairs,
                 psizes=None, fsizes=None,
                 discover_pattern=None, save_dir=None,
                 space_fname='sample-space.json',
                 space_format='json',
                 data_fname='sample-data.json',
                 data_format='json'):
        """Initialize the provider.

        Load known samples from a list of files. If :attr:`discover_pattern` is
        specified, it will also try to locate and import samples from there.

        :param list(str) plabels: input parameter names (for space).
        :param list(str) flabels: output feature names (for data).
        :param list(tuple(str)) file_pairs: list of paires `(space_file, data_file)`.
        :param list(int) psizes: number of components of parameters.
        :param list(int) fsizes: number of components of output features.
        :param str discover_pattern: UNIX-style patterns for directories with pairs
            of sample files to import.
        :param str save_dir: path to a directory for saving known snapshots.
        :param str space_fname: name of space file to write.
        :param str data_fname: name of data file to write.
        :param str space_format: space file format.
        :param str data_format: data file format.
        """
        self._cache = SampleCache(plabels, flabels, psizes, fsizes, save_dir,
                                  space_fname, space_format,
                                  data_fname, data_format)

        # load provided files
        for space_file, data_file in file_pairs:
            self._cache.read(space_file, data_file, plabels, flabels)
        # discover additionnal files
        if discover_pattern:
            self._cache.discover(discover_pattern)

        self._cache.save()

    @property
    def plabels(self):
        """Names of space parameters."""
        return self._cache.plabels

    @property
    def flabels(self):
        """Names of data features."""
        return self._cache.flabels

    @property
    def psizes(self):
        """Shape of space parameters."""
        return self._cache.psizes

    @property
    def fsizes(self):
        """Shape of data features."""
        return self._cache.fsizes

    @property
    def known_points(self):
        """List of points whose associated data is already known."""
        return self._cache.space

    def require_data(self, points):
        """Return samples for requested points.

        This provider is not able to generate missing data.
        Will raise if a point is not known.

        :return: samples for requested points (carry both space and data)
        :rtype: :class:`Sample`
        """
        # locate results in cache
        idx = self._cache.locate(points)
        if np.any(idx >= len(self._cache)):
            self.logger.error("Data cannot be provided for requested points: {}"
                              .format(points[idx >= len(self._cache)]))
            raise ValueError

        return self._cache[idx]

    def build_data(self, points):
        """Compute data for requested points.

        This provider cannot compute any data and will raise if called.

        :return: `NotImplemented`
        """
        return NotImplemented
