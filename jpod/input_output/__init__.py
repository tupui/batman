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

try:
    from .antares_wrapper import AntaresWrapper
    io_types.append(AntaresWrapper)
    import os
    os.environ["ANTARES_VERBOSE"] = "0"
except ImportError:
    logger = logging.getLogger(__name__)
    logger.info("Antares not installed")


class FileFormatError(Exception):
    pass


def IOFormatSelector(format):
    """Return an instance of io manager corresponding to a file `format`."""
    for io in io_types:
        if format == io.format:
            return io()
    raise FileFormatError('file format %s not supported'%format)
