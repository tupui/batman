# coding: utf-8
import os
import json
import logging

from ..space import Point
from ..input_output import IOFormatSelector
from .provider_plugin import ProviderPlugin
from .provider_file import ProviderFile


class SnapshotManager(object):
    """
    Manage data I/Os and data generation for Snapshot
    """

    logger = logging.getLogger(__name__)

    provider_class = {
        'plugin': ProviderPlugin,
        'file': ProviderFile,
    }

    def __init__(self, executor, settings):

        self.parameters = settings['parameters']
        self.variables = settings['variables']

        try:
            self.point_filename = settings['io']['point_filename']
            self.data_filename = settings['io']['data_filename']
            self.data_formater = IOFormatSelector(settings['io']['data_format'])
        except KeyError:
            self.point_filename = None
            self.data_filename = None
            self.data_formater = None

        provider_type = settings['provider']['type'].lower()
        self.logger.info('Select data provider type "{}"'.format(provider_type))
        self.provider = self.provider_class[provider_type](executor, self, settings['provider'])
        self.executor = executor

    def read_point(self, path):
        try:
            with open(os.path.join(path, self.point_filename), 'r') as fd:
                point_dict = json.load(fd)
        except IOError:
            self.logger.error('Attempt to read snapshot point while "io" block '
                              'in "snapshot" settings was missing')
            raise SystemExit
        return Point([point_dict[k] for k in self.parameters])

    def write_point(self, path, point):
        try:
            point_dict = dict(zip(self.parameters, point))
            with open(os.path.join(path, self.point_filename), 'w') as fd:
                json.dump(point_dict, fd)
        except IOError:
            self.logger.error('Attempt to write snapshot point while "io" block '
                              'in "snapshot" settings was missing')
            raise SystemExit

    def read_data(self, path):
        data_filepath = os.path.join(path, self.data_filename)
        try:
            data = self.data_formater.read(data_filepath, self.variables)
        except AttributeError:
            self.logger.error('Attempt to read snapshot data while "io" block '
                              'in "snapshot" settings was missing')
            raise SystemExit
        return data

    def write_data(self, path, data):
        data_filepath = os.path.join(path, self.data_filename)
        try:
            self.data_formater.write(data_filepath, data)
        except AttributeError:
            self.logger.error('Attempt to write snapshot data while "io" block '
                              'in "snapshot" settings was missing')
            raise SystemExit
