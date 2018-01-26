# coding: utf-8
"""
SnapshotManager Class
=====================

Defines methods to interact with snapshots:
    - build an appropriate data provider
    - perform read/write operations on snaphsots
"""
import os
import numpy as np

from ..space import Point
from ..input_output import formater


class SnapshotIO(object):
    """Manage data I/Os and data generation for Snapshot."""

    def __init__(self, parameter_names, feature_names,
                 point_filename='sample-space.json',
                 data_filename='sample-data.json',
                 point_format='json',
                 data_format='json'):
        """Initialize the IO manager for snapshots.

        :param list parameter_names: List of parameter labels.
        :param list feature_names: List of feature labels.
        :param str point_filename: Name of the snapshot point file.
        :param str data_filename: Name of the snapshot data file.
        :param str point_format: Name of the point file format.
        :param str data_format: Name of the data file format.
        """
        # sample parameters
        self.plabels = parameter_names
        self.point_filename = point_filename
        self.point_formater = formater(point_format)
        # sample features
        self.flabels = feature_names
        self.data_filename = data_filename
        self.data_formater = formater(data_format)

    def read_point(self, dirpath):
        """Read sample parameters from the point file.

        :param str dirpath: Path to snapshot directory.
        :rtype: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.point_filename)
        return Point(np.ravel(self.point_formater.read(filepath, self.plabels)))

    def write_point(self, dirpath, point):
        """Write sample parameters to the point file.

        :param str dirpath: Path to snapshot directory.
        :param array-like point: Sample parameters to write.
        """
        filepath = os.path.join(dirpath, self.point_filename)
        self.point_formater.write(filepath, point, self.plabels)

    def read_data(self, dirpath):
        """Read sample features from the data file.

        :param str path: Path to snapshot directory.
        :rtype: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.data_filename)
        return np.ravel(self.data_formater.read(filepath, self.flabels))

    def write_data(self, dirpath, data):
        """Write sample features to the data file.

        :param str path: Path to snapshot directory.
        :param data: Sample features to write.
        :type data: :class:`numpy.ndarray`
        """
        filepath = os.path.join(dirpath, self.data_filename)
        self.point_formater.write(filepath, data, self.flabels)
