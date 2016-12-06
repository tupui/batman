
import antares
from .base import IOBase


class AntaresWrapper(IOBase):

    """Manages IO with :module:`Antares`.

    See Antares documentation for details.
    """

    format    = []
    extension = '.dat'

    def __init__(self):
        super(AntaresWrapper, self).__init__()
        self.reader = antares.Reader(self.format)
        self.writer = antares.Writer(self.format)

    def read(self, path, names=None):
        self.reader['filename'] = path
        base = self.reader.read()

        def iteritems(base):
            for zone in base:
                for instant in base[zone]:
                    for i in base[zone][instant]:
                        yield (i[0], base[zone][instant][i])

        names = []
        for i in base[0][0]:
            names.append(i[0])

        self.info.set_names(names)
        self.info.set_shape(base[0][0].shape)

        return super(AntaresWrapper, self)._read(iteritems(base), names)

    def write(self, path, dataset):
        self.writer['filename'] = path
        
        base = antares.Base()
        base["0"] = antares.Zone()
        base["0"]["0"] = antares.Instant()

        for i, n in enumerate(dataset.names):
            base[0][0][n] = dataset.data[i]

        self.writer['base'] = base
        self.writer.dump()

    def meta_data(self, path):
        d = self.read(path)
        self.info.set_shape(d.shape)
