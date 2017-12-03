"""
Dataset module
**************
"""
__docformat__ = "reStructuredText"

"""This module contains the classes :class:`DatasetInfo` and :class:`Dataset`."""


class ShapeError(Exception):
    pass


class DataSizeError(Exception):
    pass


class DatasetInfo(object):
    """A meta-data container.

    This class provides a container for the meta informations needed to
    interpret the content of a :class:`Dataset`. At its core, a dataset is a
    numpy array that is composed of one or several quantities. A quantity is a
    part of the dataset that is bound to a variable name, such as a coordinate
    or a velocity component. The data array or sub-array related to a quantity
    also has a structure or shape which is defined by its number of elements in
    each dimensions.

    An instance of :class:'DatasetInfo' has the following attributes:
    * 'names', the names of the quantities of a dataset, stored as a tuple,
    it defines the order in which the quantities are stored in the dataset array.
    * 'shape', the array shape of one quantity, all quantities have the same shape.

    """

    def __init__(self, names=None, shape=None):
        self.names = None
        self.shape = None
        if names is not None:
            self.set_names(names)
        if shape is not None:
            self.set_shape(shape)

    def __eq__(self, other):
        return (self.names == other.names) and \
               (self.shape == other.shape)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return 'names: {}\nshape: {}'.format(self.names, self.shape)

    @property
    def size_of_one_quantity(self):
        size = 1
        for i in self.shape:
            size *= i
        return size

    @property
    def size(self):
        return self.size_of_one_quantity * len(self.names)

    def set_names(self, names):
        self.names = tuple(names)

    def set_shape(self, shape):
        self.shape = tuple(shape)

    def _check_type(self, type_, sequence):
        ok = True
        if not isinstance(sequence, (list, tuple)) \
           or len(sequence) == 0:
            ok = False
        else:
            for v in sequence:
                if not isinstance(v, type_):
                    ok = False
                    break
        return ok


class Dataset(DatasetInfo):
    """A data container.

    This class provides a container for datasets. Besides being a subclass of
    :class:'DatasetInfo', it holds the actual data as a numpy array.
    Quantities data can be accessed or set in a python dictionary like way,
    i.e. by using a quantity name as a key.

    Whatever the actual shape of the data array passed to this class,
    internally the data will be stored as a view on the actual array (see numpy
    documentation for explanations) shaped according the attributes of the
    corresponding :class:`DatasetInfo`.

    For instance, if a 1D array with 8 elements is passed to :class:`Dataset`,
    which meta-data define 2 quantities with the shapes (2,2), then internally
    the shape of the array view will be (2,2,2).

    """

    def __init__(self, data=None, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.data = None

        if data is not None:
            self.set_data(data)

    def set_data(self, data):
        """Set dataset array from `data`.

        If the dataset shape is not set, the shape of `data` but the first
        dimension will be used.
        """
        if self.shape is None:
            self.shape = data.shape[1:]
        elif data.size != self.size:
            msg = 'data size mismatch : %d != %d' % (data.size, self.size)
            raise DataSizeError(msg)

        self.data = data.view()
        self.data.shape = [len(self.names)] + list(self.shape)

    def __getitem__(self, name):
        if name not in self.names:
            raise KeyError(name)
        else:
            return self.data[self.names.index(name)]

    def __setitem__(self, name, data):
        if name not in self.names:
            raise KeyError(name)
        else:
            index = list(self.names).index(name)
            self.data[index] = data.view().reshape(self.shape)

    @property
    def info(self):
        """Return the :class:`DatasetInfo` corresponding to a :class:`Dataset`."""
        return DatasetInfo(names=self.names, shape=self.shape)
