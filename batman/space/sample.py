# coding: utf8
"""
Sample class
============

Wrapper around a :class:`pandas.DataFrame` for storing point samples.
A sample is given by the data associated to a point,
and the point coordinates in the space of parameters.

The main benefit of this class is to carry feature labels
and to handle I/Os.

The internal dataframe is publicly available.
Class attributes are configured to return array-like objects
(:class:`numpy.ndarray` or :py:class:`list`)
"""
from copy import copy
from numbers import Number
import os
import logging
import numpy as np
import pandas as pd
from ..input_output import formater


class Sample(object):
    """Container class for samples."""

    logger = logging.getLogger(__name__)

    def __init__(self, space=None, data=None, plabels=None, flabels=None,
                 psizes=None, fsizes=None, pformat='json', fformat='json'):
        """Initialize the container and build the column index.

        This index carries feature names. Features can be scalars or vectors.
        Vector features do not need to be of the same size.
        Samples are stored as a 2D row-major array: 1 sample per row.

        :param array-like space: parameter space (1 point per sample)
        :param array-like data: data associated to points
        :param list(str) plabels: parameter names (for space)
        :param list(str) flabels: feature names (for data)
        :param list(int) psizes: lengths of parameters (for space)
        :param list(int) fsizes: lengths of features (for data)
        :param str pformat: file format name for space
        :param str fformat: file format name for data
        """
        # space dataframe
        df_space = None
        if space is not None:
            df_space = create_dataframe(space, clabel='space', flabels=plabels,
                                        fsizes=psizes)
        elif ((plabels is not None and list(plabels))
              or (psizes is not None and list(psizes))):
            index = create_index(clabel='space', flabels=plabels, fsizes=psizes)
            df_space = pd.DataFrame(columns=index)

        # data dataframe
        df_data = None
        if data is not None:
            df_data = create_dataframe(data, clabel='data', flabels=flabels,
                                       fsizes=fsizes)
        elif ((flabels is not None and list(flabels))
              or (fsizes is not None and list(fsizes))):
            index = create_index(clabel='data', flabels=flabels, fsizes=fsizes)
            df_data = pd.DataFrame(columns=index)

        # concatenate
        try:
            self._dataframe = pd.concat([df_space, df_data], axis=1)
        except ValueError:
            self._dataframe = pd.DataFrame()

        # I/O formaters
        self._pformater = formater(pformat)
        self._fformater = formater(fformat)

        self.desc = ''

    # ----------------
    # Field Accessors
    # ----------------

    @property
    def shape(self):
        """Shape of the internal array."""
        return self._dataframe.shape

    @property
    def plabels(self):
        """List of space feature labels.

        :returns: a list of column labels, ordered the same as the underlying array.
        :rtype: list(str)
        """
        try:
            index = self._dataframe['space'].columns
        except KeyError:
            return []
        else:
            uniq, pos = np.unique(index.labels[0], return_index=True)
            uniq = uniq[np.argsort(pos)]
            return list(index.levels[0][uniq])

    @property
    def flabels(self):
        """List of data feature labels.

        :returns: a list of column labels, ordered the same as the underlying array.
        :rtype: list(str)
        """
        try:
            index = self._dataframe['data'].columns
        except KeyError:
            return []
        else:
            uniq, pos = np.unique(index.labels[0], return_index=True)
            uniq = uniq[np.argsort(pos)]
            return list(index.levels[0][uniq])

    @property
    def psizes(self):
        """Sizes of space features.

        :returns: the number of components of each feature.
        :rtype: list(int)
        """
        try:
            index = self._dataframe['space'].columns
        except KeyError:
            return []
        else:
            _, sizes = np.unique(index.labels[0], return_counts=True)
            return list(sizes)

    @property
    def fsizes(self):
        """Sizes of data features.

        :returns: the number of components of each feature.
        :rtype: list(int)
        """
        try:
            index = self._dataframe['data'].columns
        except KeyError:
            return []
        else:
            _, sizes = np.unique(index.labels[0], return_counts=True)
            return list(sizes)

    @property
    def dataframe(self):
        """Underlying dataframe."""
        return self._dataframe

    @property
    def values(self):
        """Underlying :class:`numpy.ndarray`.

        Shape is `(n_sample, n_columns)`.
        There may be multiple columns per feature.
        See `Sample.psizes` and `Sample.fsizes`.
        """
        if not self:
            return np.empty(self.shape)
        return self._dataframe.values

    @property
    def space(self):
        """Space :class:`numpy.ndarray` (point coordinates)."""
        try:
            return self._dataframe['space'].values
        except KeyError:
            return np.empty((len(self), 0))

    @property
    def data(self):
        """Core of the data :class:`numpy.ndarray`."""
        try:
            return self._dataframe['data'].values
        except KeyError:
            return np.empty((len(self), 0))

    # ------------------
    # Container methods
    # ------------------

    def append(self, other, axis=0):
        """Append samples to the container.

        :param other: samples to append (1 sample per row)
        :param axis: how to append (add new samples or new features).
        :type other: array-like or :class:`pandas.DataFrame` or :class:`Sample`
        :type axis: 0 or 1
        """
        # get dataframe
        if other is None:
            return
        elif isinstance(other, Sample):
            df_other = other.dataframe
        elif isinstance(other, (pd.DataFrame, pd.Series)):
            idx = other.columns if isinstance(other, pd.DataFrame) else other.index
            assert idx.nlevels == 3 or idx.size == 0
            if axis == 0:
                assert ('space' in other) == ('space' in self._dataframe)
                assert ('data' in other) == ('data' in self._dataframe)
                for label in self.plabels:
                    assert label in other['space']
                for label in self.flabels:
                    assert label in other['data']
            df_other = other
        else:
            if axis == 1:
                msg = 'Cannot append unnamed dataset as columns.'
                self.logger.error(msg)
                raise ValueError(msg)
            if isinstance(other, Number):
                other = np.broadcast_to(other, (1, self._dataframe.shape[-1]))
            other = np.asarray(other)
            if len(other.shape) < 2:
                other = other.reshape(1, other.size)
            if len(other.shape) > 2:
                other = other.reshape(other.shape[0], np.prod(other.shape[1:]))
            df_other = pd.DataFrame(other, columns=self._dataframe.columns)

        # append
        ignore_index = (axis == 0)
        self._dataframe = pd.concat([self._dataframe, df_other],
                                    axis=axis,
                                    ignore_index=ignore_index)

    def pop(self, sid=-1):
        """Return and remove a sample (default: last one)."""
        item = self[sid]
        del self[sid]
        return item

    def empty(self):
        """Remove every stored samples."""
        del self[:]

    # -----------------
    # Inputs / Outputs
    # -----------------

    def read(self, space_fname='sample-space.json', data_fname='sample-data.json',
             plabels=None, flabels=None):
        """Read and append samples from files.

        Samples are stored in 2 files: space and data.

        :param str space_fname: path to space file.
        :param str data_fname: path to data file.
        :param list(str) plabels: labels in space file
          (if different from `self.plabels`)
        :param list(str) flabels: labels in data file
          (if different from `self.flabels`)
        """
        pd_sample = []
        if self.plabels:
            if plabels is None:
                plabels = self.plabels
            try:
                np_space = self._pformater.read(space_fname, plabels)
            except (OSError, IOError):
                self.logger.error('Cannot read {} in {}'
                                  .format(plabels, space_fname))
            else:
                pd_sample.append(pd.DataFrame(np_space))

        if self.flabels:
            if flabels is None:
                flabels = self.flabels
            try:
                np_data = self._fformater.read(data_fname, flabels)
            except (OSError, IOError):
                self.logger.error('Cannot read {} in {}'
                                  .format(plabels, data_fname))
            else:
                pd_sample.append(pd.DataFrame(np_data))

        if pd_sample:
            concat = pd.concat(pd_sample, axis=1)
            n_not_found = concat.isnull().values.sum()

            if n_not_found:
                self.logger.warning('Inconsistent number of sample/data:'
                                    ' {} data not loaded'.format(n_not_found))

            np_sample = pd.DataFrame.dropna(concat).values
            self.append(np_sample)

    def write(self, space_fname='sample-space.json', data_fname='sample-data.json'):
        """Write samples to files.

        Samples are stored in 2 files: space and data.
        Override if files exist.

        :param str space_fname: path to space file.
        :param str data_fname: path to data file.
        """
        if self.space.size:
            self._pformater.write(space_fname, self.space, self.plabels, self.psizes)
        if self.data.size:
            self._fformater.write(data_fname, self.data, self.flabels, self.fsizes)

    # -----------
    # Data Model
    # -----------

    def __len__(self):
        """Python Data Model.

        `len` function. Return the number of samples."""
        return len(self._dataframe)

    def __repr__(self):
        """Python Data Model.

        `str` function. Underlying dataframe representation."""
        msg = str(self._dataframe)
        if self.desc:
            msg = self.desc + os.linesep + msg
        return msg

    def __iadd__(self, other):
        """Python Data Model. `+=` operator. Append samples."""
        self.append(other)
        return self

    def __add__(self, other):
        """Python Data Model.

        `+` operator.
        :returns: :class:`Sample` with samples from both operands.
        """
        new = copy(self)
        new.append(other)
        return new

    def __getitem__(self, sid):
        """Python Data Model. `[]` operator. Return requested samples.

        :returns: 1D :class:`numpy.ndarray` if requested 1 item,
          :class:`Sample` otherwise.
        """
        item = self._dataframe.iloc[sid]
        if item.ndim > 1:
            new = copy(self)
            new._dataframe = item
            return new
        return item.values

    def __setitem__(self, sid, value):
        """Python Data Model. `[]` operator. Replace specified samples.

        :param array-like value: 1D array if setting 1 sample, 2D array otherwise.
        """
        self._dataframe.iloc[sid] = value

    def __delitem__(self, sid):
        """Python Data Model. `del []` statement. Remove specified samples."""
        sid = self._dataframe.index[sid]
        self._dataframe = self._dataframe.drop(sid)

    def __contains__(self, item):
        """Python Data Model.

        `is in` statement. Test if item is one of the stored samples.
        """
        try:
            return item.values in self._dataframe.values
        except AttributeError:
            return item in self._dataframe.values

    def __iter__(self):
        """Python Data Model. `for in` statement. Iterate other samples as 1D arrays."""
        generator = (row.values for i, row in self._dataframe.iterrows())
        return generator


# -----------------
# Helper functions
# -----------------

def create_dataframe(dataset, clabel='space', flabels=None, fsizes=None):
    """Create a DataFrame with a 3-level column index.

    Columns are feature components, rows are dataset entries (samples).

    :param dataset: array-like or :class:`pandas.DataFrame`.
    :param str clabel: class of features (1st level in index).
      Typically 'space' or 'data.'
    :param list(str) flabels: labels of features.
    :param list(int) fsizes: number of components of features.
    :rtype: :class:`pandas.DataFrame`
    """
    # enforce a 3-level index for columns
    if isinstance(dataset, pd.DataFrame):
        # get multilevel index
        idx = pd.MultiIndex.from_tuples([np.atleast_1d(c) for c in dataset.columns.values])
        idx_levels = idx.levels
        idx_labels = idx.labels
        # prepend level 'clabel' if missing
        if not np.array_equal([clabel], idx.levels[0]):
            idx_levels = [[clabel]] + idx_levels
            idx_labels = [[0] * idx.size] + idx_labels
        # merge levels trailing levels
        if len(idx_levels) > 3:
            _, pos, counts = np.unique(idx_labels[1], return_index=True, return_counts=True)
            last_level = list(range(max(counts)))
            last_label = sum([range(c) for i, c in sorted(zip(pos, counts))], [])
            idx_levels = idx_levels[:2] + [last_level]
            idx_labels = idx_labels[:2] + [last_label]
        # rebuild dataframe
        idx = pd.MultiIndex(idx_levels, idx_labels)
        dataframe = pd.DataFrame(dataset.values, columns=idx)

    else:
        # built index from scratch
        dataset = np.atleast_3d(dataset)
        nsample, n_features, n_components = dataset.shape
        length = n_features * n_components

        if (flabels is not None) and list(flabels):
            n_features = len(flabels)
            n_components = length // n_features
        if (fsizes is None) or not list(fsizes):
            fsizes = [n_components] * n_features

        idx = create_index(clabel, flabels, fsizes)
        dataframe = pd.DataFrame(dataset.reshape(nsample, length), columns=idx)

    return dataframe


def create_index(clabel, flabels=None, fsizes=None):
    """Build a 3-level index.

    - 1st level: feature class ('space', 'data', ...)
    - 2nd level: feature names
    - 3rd level: component indices: `[0, N]` where `N` is the number of
      components of 1 feature.

    :param str clabel: class of features.
    :param list(str) flabels: labels of features.
    :param list(int) fsizes: number of components of features.
    :rtype: :class:`pandas.MultiIndex`
    """
    if ((flabels is None or not list(flabels))
            and (fsizes is None or not list(fsizes))):
        msg = 'Unable to build an index: number of labels is unknown'
        raise ValueError(msg)
    if (flabels is None) or not list(flabels):
        char = 'p' if clabel == 'space' else 'f'
        flabels = ['{}{}'.format(char, i) for i in range(len(fsizes))]
    if (fsizes is None) or not list(fsizes):
        fsizes = [1] * len(flabels)
    tuples = [(clabel, label, i) for label, size in zip(flabels, fsizes) for i in range(size)]
    return pd.MultiIndex.from_tuples(tuples)
