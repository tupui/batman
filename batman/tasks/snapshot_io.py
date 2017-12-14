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
import logging
import numpy as np

from ..space import Point
from ..input_output import IOFormatSelector, Dataset


class SnapshotIO(object):
    """
    Manage data I/Os and data generation for Snapshot
    """

    logger = logging.getLogger(__name__)

    def __init__(self, parameter_names, variable_names, 
                 point_filename='point.json', 
                 data_filename='point.dat',
                 data_format='fmt_tp_fortran'):
        
        self.parameters = parameter_names
        self.variables = variable_names
        self.point_filename = point_filename
        self.data_filename = data_filename
        self.data_formater = IOFormatSelector(data_format)

    def read_point(self, path):
        "Read point parameters from the header file"
        with open(os.path.join(path, self.point_filename), 'r') as fd:
            point_dict = json.load(fd)
        return Point([point_dict[k] for k in self.parameters])

    def write_point(self, path, point):
        "Write point parameters as the header file"
        point_dict = dict(zip(self.parameters, point))
        with open(os.path.join(path, self.point_filename), 'w') as fd:
            json.dump(point_dict, fd)

    def read_data(self, path):
        "Read point data from the data file"
        data_filepath = os.path.join(path, self.data_filename)
        data = self.data_formater.read(data_filepath, self.variables)
        return data

    def write_data(self, path, data):
        "Writte point data as the data file"
        data_filepath = os.path.join(path, self.data_filename)
        data = np.array(data)
        dataset = Dataset(names=self.variables, shape=data.shape, data=data)
        self.data_formater.write(data_filepath, dataset)
