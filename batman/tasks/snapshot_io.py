# coding: utf-8
"""
SnapshotManager Class
=====================

Defines methods to interact with snapshots:
    - build an appropriate data provider
    - perform read/write operations on snaphsots
"""
import os
import json
import numpy as np

from ..space import Point
from ..input_output import FORMATER


class SnapshotIO(object):
    """Manage data I/Os and data generation for Snapshot."""

    def __init__(self, parameter_names, feature_names,
                 coord_filename='sample-coord.json',
                 data_filename='sample-data.json',
                 coord_format='json',
                 data_format='json'):
        """Initialize the IO manager for snapshots.

        :param list parameter_names: List of parameter labels (coordinates: input space).
        :param list feature_names: List of feature labels (data: output space).
        :param str coord_filename: Name of the snapshot coordinates file.
        :param str data_filename: Name of the snapshot data file.
        :param str coord_format: Name of the coordinates file format.
        :param str data_format: Name of the data file format.
        """
        # sample parameters
        self.plabels = parameter_names
        self.coord_filename = coord_filename
        self.coord_formater = FORMATER[coord_format]
        # sample features
        self.flabels = feature_names
        self.data_filename = data_filename
        self.data_formater = FORMATER[data_format]

    def read_parameters(self, dirpath):
        """Read sample parameters from the coordinates file.

        :param str dirpath: Path to snapshot directory.
        :rtype: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.coord_filename)
        return np.ravel(self.coord_formater.read(filepath, self.plabels))

    def write_parameters(self, dirpath, coordinates):
        """Write sample parameters to the coordinates file.

        :param str dirpath: Path to snapshot directory.
        :param coordinates: Sample parameters to write.
        :type coordinate: :class:`numpy.array`
        """
        filepath = os.path.join(dirpath, self.coord_filename)
        self.coord_formater.write(filepath, coordinates, self.plabels)

    def read_features(self, dirpath):
        """Read sample features from the data file.

        :param str path: Path to snapshot directory.
        :rtype: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.data_filename)
        return np.ravel(self.data_formater.read(filepath, self.flabels))

    def write_features(self, dirpath, data):
        """Write sample features to the data file.

        :param str path: Path to snapshot directory.
        :param data: Sample features to write.
        :type data: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.data_filename)
        self.coord_formater.write(filepath, data, self.flabels)
