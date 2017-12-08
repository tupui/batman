import os
import re
from ._tecplot import ascii
from .base import *

__docformat__ = "reStructuredText"


class TecplotAscii(IOBase):
    """Manages IO for ASCII tecplot files.

    See data format documentation on tecplot web site.
    In addition to the attributes of the base class :class:`IOBase`, the
    `data_format` attribute is used to defined the format used for writing data
    in fortran.

    """

    format = 'fmt_tp_fortran'
    extension = '.dat'

    data_format = '(6e15.7)'
    '''Fortran format used for writing data.'''

    @use_base_class_docstring
    def read(self, path, names=None):
        """Process header."""
        self.meta_data(path)

        # provides an iterator with lazy evaluation over the quantities stored
        # in the file.
        def generator(path):
            # verify what's done with the fortran unit when the loop is
            # broken
            unit = ascii.open_file(path, 'formatted', 'read', 'rewind')
            ascii.skip_header(unit)
            for v in self.info.names:
                yield v, ascii.read_array(unit, *self.info.shape)
            ascii.close_file(unit)

        return super(TecplotAscii, self)._read(generator(path), names)

    @use_base_class_docstring
    def write(self, path, dataset):
        """Write header and data."""
        # header
        with open(path, 'wb') as f:
            for line in self.header(dataset):
                f.write(line.encode('utf8'))

        # data
        unit = ascii.open_file(path, 'formatted', 'write', 'append')
        for v in dataset.data:
            ascii.write_array(unit, self.data_format, v)
        ascii.close_file(unit)

    def header(self, dataset):
        """Return a header from dataset info, as a list of strings."""
        header = []

        # if self.file:
        #     # get header from source file
        #     for line in open(self.file):
        #         header += [line]
        #         if re.match('^\s*ZONE', line, flags=re.IGNORECASE):
        #             break
        # else:
        # otherwise create a dummy one

        header += ['TITLE = ""']  # elsA_IO will barf otherwise
        header += ['VARIABLES = "' + '" "'.join(dataset.names) + '"']
        indices = ('I', 'J', 'K')
        s = ''
        for i, idx in enumerate(dataset.shape):
            s += '%s=%d, ' % (indices[i], idx)
        header += ['ZONE ' + s + ' F=BLOCK']
        header = [i + '\n' for i in header]

        return header

    def meta_data(self, path):
        """Parse meta-data from `path`."""
        # check file here as io_fortran will not barf
        if not os.path.isfile(path):
            raise IOError('No such file: %s' % path)

        # process names
        names = None
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf8')
                if re.match(r"^\s*VARIABLES", line, flags=re.IGNORECASE):
                    names = re.findall('"(.*?)"', line)
                    break

        if names is None:
            raise NameError('cannot find "names" field in the file %s' % path)
        else:
            self.info.set_names(names)

        # process shape
        shape_map = None
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('utf8')
                # assume shape are on the ZONE line
                if re.match(r"^\s*ZONE", line, flags=re.IGNORECASE):
                    # must be block format, not point
                    if re.match(r".*=\s*POINT.*", line, flags=re.IGNORECASE):
                        raise FormatError('data not in block format.')

                    ids = re.findall(r"(I|J|K)\s*=\s*(\d+)",
                                     line, flags=re.IGNORECASE)
                    for i in ids:
                        try:
                            if shape_map is None:
                                shape_map = {}
                            shape_map[i[0].lower()] = int(i[1])
                        except:
                            msg = 'cannot read shape from the line "%s" in the file %s' \
                                  % (line, path)
                            raise ShapeError(msg)
                    break

        if shape_map is None:
            raise ShapeError('cannot find "zone" field in the file %s' % path)
        else:
            # get the values in order
            shape = [1, 1, 1]
            for i, key in enumerate(sorted(shape_map.keys())):
                shape[i] = shape_map[key]

            self.info.set_shape(shape)
