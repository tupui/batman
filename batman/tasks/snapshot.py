# coding: utf8
import os
import sys
import logging
import shutil
import re
import numpy as np
from ..input_output import IOFormatSelector, Dataset
from ..space import Point


class SnapshotProvider(object):

    """Utility class to make the code more readable.

    This is how the provider type is figured out.
    """

    def __init__(self, provider):
        self.provider = provider

    @property
    def is_file(self):
        return isinstance(self.provider, list)

    @property
    def is_job(self):
        return isinstance(self.provider, dict)

    @property
    def is_function(self):
        try:
            check_type = isinstance(self.provider, unicode)
        except NameError:
            check_type = isinstance(self.provider, str)
        if check_type:
            sys.path.append('.')
            fun_provider = __import__(self.provider)
            self.provider = fun_provider.f
        return callable(self.provider)

    def __getitem__(self, key):
        return self.provider[key]

    def __call__(self, *args, **kwargs):
        return self.provider(*args, **kwargs)


class Snapshot(object):

    """A snapshot container.

    A snapshot is a vector bound to a point in the space of parameter.
    This class manages several aspects:
    * settings common to all snapshots,
    * IO, to read and write snapshots to disk,
    * data manipulations to be provided to the pod processing.
    """

    logger = logging.getLogger(__name__)

    point_filename = None
    '''File name for storing the coordinates of a point.'''

    point_format = '%s = %s\n'
    '''Line format for writing a point coordinate to file, name and coordinate value, the value must be repr().'''

    template_directory = None
    '''Location of io templates.'''

    shapes = None
    '''Shapes of the data partitions.'''

    variables = None
    '''Names of the variables contained in a snapshot.'''

    filenames = None
    '''File names used for io.'''

    io = None
    '''Object for doing io operations with a given file format.'''

    initialized = False
    '''Switch to check that initialization has been done.'''

    parameter_names = None
    '''List of coordinate names, used for points io.'''

    @classmethod
    def initialize(cls, settings):
        """Initialize the `settings` common to all snapshots."""
        cls.point_filename = settings['point_filename']
        cls.variables = settings['variables']
        cls.io = IOFormatSelector(settings['format'])

        # parameter names
        parameter_names = settings['parameter_names']
        cls.parameter_names = tuple(parameter_names)

        # gather all filenames
        cls.filenames = []
        for v in settings['filenames'].values():
            cls.filenames += v

        # one and only one of shapes or template_directory must be set
        cls.initialized = False

        if settings['shapes']:
            # gather all shapes
            shapes = []
            for v in settings['shapes'].values():
                shapes += v

            if len(shapes) != len(cls.filenames):
                msg = 'shapes number and filenames number mismatch : %i != %i'
                msg = msg % (len(shapes), len(cls.filenames))
                raise Exception(msg)

            cls.shapes = shapes
            cls.initialized = True

        if settings['template_directory']:
            cls._get_shapes_from_template(settings['template_directory'])
            cls.template_directory = settings['template_directory']
            cls.initialized = True

        if not cls.initialized:
            cls.logger.exception('one of "shapes" or "template_directory" must be set.')
            raise SystemExit

        # logging
        msg = ("\n----- Snapshot settings -----\n"
               "variables: {}\n"
               "format: {}\n"
               "parameter_names: {}\n"
               "template_directory: {}\n"
               "shapes: {}\n"
               "filenames: {}\n"
               "point_filename: {}\n"
               .format(cls.variables, cls.io.format, cls.parameter_names,
                       cls.template_directory, cls.shapes, cls.filenames,
                       cls.point_filename))

        cls.logger.info(msg)

    @classmethod
    def _create_templates(cls, directory):
        # create template directory
        t_d = cls.template_directory
        if not os.path.exists(t_d):
            os.makedirs(t_d)
        elif not os.path.isdir(t_d):
            cls.logger.exception('Template path must be a directory')
            raise IOError

        # gather potential template files for checking
        files_to_copy = []
        for f in cls.filenames:
            path = os.path.join(t_d, f)
            if not os.path.isfile(path):
                files_to_copy += [f]

        # check if templates are all already existing or
        # all not existing
        if files_to_copy:
            if len(files_to_copy) != len(cls.filenames):
                raise Exception('Check template directory content')
            else:
                for f in cls.filenames:
                    path = os.path.join(directory, f)
                    shutil.copy(path, t_d)

        cls._get_shapes_from_template(directory)
        cls.logger.info('Snapshot file templates created')

    @classmethod
    def _get_shapes_from_template(cls, directory):
        shapes = []
        for f in cls.filenames:
            path = os.path.join(directory, f)
            cls.io.meta_data(path)
            shapes += [cls.io.info.shape]
        cls.shapes = shapes
        cls.logger.info('Shape read from template')

    @classmethod
    def convert(cls, snapshot, path=None):
        """Convert a `snapshot` between disk and object.

        :param snapshot: either an object or a directory path
        :param path: optional, when `snapshot` is an object, path to directory where to write it.
        """
        if path is None:
            if isinstance(snapshot, cls):
                # nothing to do
                output = snapshot
            else:
                # convert snapshot on disk to a snapshot object
                output = cls.read(snapshot)
        else:
            # convert a snapshot object to a snapshot on disk
            snapshot.write(path)
            output = path

        return output

    @classmethod
    def _check_data(cls, data):
        # compute expected data size
        total_size = 0
        for shape in cls.shapes:
            size = len(cls.variables)
            for i in shape:
                size *= i
            total_size += size

        # check shapes or store them for later check
        if len(data) != total_size:
            cls.logger.exception('Bad dataset size: got {} instead of {}'
                                 .format(len(data), total_size))
            raise SystemExit

    @classmethod
    def read_point(cls, directory):
        """Read a snapshot point from `directory` and return it."""
        cls._must_be_initialized()
        names = []
        coordinates = []
        path = os.path.join(directory, cls.point_filename)
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf8')
                p = re.findall('^\s*(\S+)\s*=\s*(\S+)\s*', line)

                # checks
                if len(p) != 1:
                    cls.logger.exception('Parsing problem in {} on the line {}'
                                         .format(path, line))
                    raise ValueError

                names += [p[0][0]]
                coordinates += [p[0][1]]

        if tuple(names) != cls.parameter_names:
            cls.logger.exception("Bad coordinate names, should be: {}"
                                 " but got: {}"
                                 .format(str(cls.parameter_names), str(names)))
            raise ValueError

        cls.logger.debug('Point read from: {}'.format(path))
        return Point(coordinates)

    @classmethod
    def write_point(cls, point, directory):
        """Write a snapshot point to `directory`."""
        cls._must_be_initialized()
        if not os.path.isdir(directory):
            os.makedirs(directory)
        path = os.path.join(directory, cls.point_filename)
        with open(path, 'wb') as f:
            for k, v in zip(cls.parameter_names, point):
                f.write((cls.point_format % (k, repr(v))).encode('utf8'))
        cls.logger.debug('Point wrote to: {}'.format(path))

    @classmethod
    def read_data(cls, directory):
        """Read snapshot data from `directory` and return it.

        :param directory : directory path
        """
        cls._must_be_initialized()

        # read the data and gather dataset infos
        data = np.zeros(0)
        for f in cls.filenames:
            path = os.path.join(directory, f)
            d = cls.io.read(path, cls.variables)
            data = np.concatenate((data, d.data.ravel()))

        cls._check_data(data)

        cls.logger.debug('Data read from: %s', directory)
        return data

    @classmethod
    def write_data(cls, data, directory):
        """Write snapshot data to `directory`."""
        cls._must_be_initialized()
        cls._check_data(data)
        try:
            os.makedirs(directory)
        except OSError:
            pass

        start = 0
        for i, f in enumerate(cls.filenames):
            # determine data shape
            if cls.template_directory:
                # get template data too
                template = os.path.join(cls.template_directory, f)
                full_data = cls.io.read(template)
                shape = full_data.shape
            else:
                shape = cls.shapes[i]  # TODO: not necessary

            # create a dataset
            size = len(cls.variables)
            for i in shape:
                size *= i
            end = start + size
            dataset = Dataset(names=cls.variables, shape=shape,
                              data=data[start:end])
            start = end

            path = os.path.join(directory, f)

            # write date to disk
            if cls.template_directory:
                # replace data
                for name in dataset.names:
                    full_data[name] = dataset[name]

                cls.io.write(path, full_data)
            else:
                cls.io.write(path, dataset)

        cls.logger.debug('Data wrote to %s', directory)

    @classmethod
    def read(cls, directory):
        """Read snapshot data from disk.

        :param directory : directory path
        """
        point = cls.read_point(directory)
        data = cls.read_data(directory)
        cls.logger.info('Read snapshot at point %s from %s',
                        point, directory)
        return cls(point, data)

    @classmethod
    def _must_be_initialized(cls):
        if not cls.initialized:
            cls.logger.exception('Snapshot class is not initialized.')
            raise SystemExit

    def __init__(self, point, data):
        cls = self.__class__

        cls._must_be_initialized()

        self.point = Point(point)
        '''Point coordinates in the space of parameters.'''

        cls._check_data(data)
        self.data = data
        '''Data of a snapshot, ndarray(total nb of data).'''

    def __str__(self):
        s = ("point: {}\n"
             "data: {}\n"
             "shape: {}\n"
             "{}\n"
             .format(repr(self.point), self.data, str(self.data.shape),
                     [s.lstrip() for s in str(self.data.flags).split('\n')]))
        return s

    def write(self, directory):
        """Write snapshot to `directory`."""
        cls = self.__class__
        cls.write_point(self.point, directory)
        cls.write_data(self.data, directory)
        cls.logger.info('Snapshot at point %s wrote to %s',
                        self.point, directory)
