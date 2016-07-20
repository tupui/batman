"""Input output management entry point.

An input-output, or io, is used to deal with the permanent storage of a dataset.
"""

from .base import FormatError
from .dataset import DatasetInfo, Dataset, ShapeError, DataSizeError
from .tecplot import TecplotAscii
from .npz import Npz

# list of all supported io classes
io_types = [
TecplotAscii,
Npz
]


class FileFormatError(Exception):
    pass


def IOFormatSelector(format):
    """Return an instance of io manager corresponding to a file `format`."""
    for io in io_types:
        if format == io.format:
            return io()
    raise FileFormatError('file format %s not supported'%format)
