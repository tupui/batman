"""
IO module
*********

Input output management entry point.
An input-output (io) is used to deal with the permanent storage of a dataset.
"""

from .base import FormatError
from .dataset import Dataset
from .tecplot import TecplotAscii
from .npz import Npz
import logging


class IOFormatSelector(object):

    """Return an instance of io manager corresponding to a file `format`."""

    logger = logging.getLogger(__name__)

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
        check_antares = True
    except ImportError:
        check_antares = False
        logger.info("Antares not installed")

    def __init__(self, format):
        """Select the io class to use."""
        self.format = format

        if self.check_antares is True:
            self.io_types[-1].format = self.format

        check_format_init = False

        for io in self.io_types:
            if self.format == io.format:
                try:
                    io = io()
                    check_format_init = True
                except KeyError as bt:
                    self.logger.info("Not available in Antares: {}".format(bt))
                    pass

                self.read = io.read
                self.write = io.write
                self.meta_data = io.meta_data
                self.info = io.info
                return

        if check_format_init is False:
            raise FormatError("File format {} doesn't exist: {}"
                              .format(self.format))
