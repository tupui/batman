# coding: utf8
"""
[TODO]
"""
from copy import copy
from numbers import Number
import os
import logging
import numpy as np
import pandas as pd
from ..input_output import formater


class Sample(object):
    """[TODO]
    """

    logger = logging.getLogger(__name__)

    def __init__(self, space=None, data=None, plabels=None, flabels=None,
                 psizes=None, fsizes=None, pformat='json', fformat='json'):
        """[TODO]

        dataset shapes:  ruled by numpy.atleast_3d
        - (n_features,)                         -->  (1, n_features, 1)
        - (n_sample, n_features)                -->  (n_sample, n_features, 1)
        - (n_sample, n_features, n_components)  -->  (n_sample, n_features, n_components)

        if features have different number of components, please pass a pandas dataframe

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
            df_space = create_dataframe(space, clabel='space', flabels=plabels, fsizes=psizes)
        elif (((plabels is not None) and (len(plabels) > 0))
                or ((psizes is not None) and (len(psizes) > 0))):
            index = create_index(clabel='space', flabels=plabels, fsizes=psizes)
            df_space = pd.DataFrame(columns=index)
            
        # data dataframe
        df_data = None
        if data is not None:
            df_data = create_dataframe(data, clabel='data', flabels=flabels, fsizes=fsizes)
        elif (((flabels is not None) and (len(flabels) > 0))
                or ((fsizes is not None) and (len(fsizes) > 0))):
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

    # -----------------------------------------------------------
    # Field Accessors
    # -----------------------------------------------------------

    @property
    def shape(self):
        """[TODO]"""
        return self._dataframe.shape

    @property
    def plabels(self):
        """[TODO]"""
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
        """[TODO]"""
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
        """[TODO]"""
        try:
            index = self._dataframe['space'].columns
        except KeyError:
            return []
        else:
            _, sizes = np.unique(index.labels[0], return_counts=True)
            return list(sizes)

    @property
    def fsizes(self):
        """[TODO]"""
        try:
            index = self._dataframe['data'].columns
        except KeyError:
            return []
        else:
            _, sizes = np.unique(index.labels[0], return_counts=True)
            return list(sizes)

    @property
    def dataframe(self):
        """[TODO]"""
        return self._dataframe

    @property
    def values(self):
        """[TODO]"""
        if len(self) == 0:
            return np.empty(self.shape)
        return self._dataframe.values

    @property
    def space(self):
        """[TODO]"""
        try:
            df_space = self._dataframe['space']
            if len(df_space) == 0:
                return np.empty(df_space.shape)
            return df_space.values
        except KeyError:
            return np.empty((0, 0))

    @property
    def data(self):
        """[TODO]"""
        try:
            df_data = self._dataframe['data']
            if len(df_data) == 0:
                return np.empty(df_data.shape)
            return df_data.values
        except KeyError:
            return np.empty((0, 0))

    # -----------------------------------------------------------
    # Container methods
    # -----------------------------------------------------------

    def append(self, other, axis=0):
        """[TODO]

        :type other: array-like or :class:`pandas.DataFrame`
        """
        # get dataframe
        if other is None:
            return
        elif isinstance(other, Sample):
            df_other = other.dataframe
        elif isinstance(other, pd.DataFrame) or isinstance(other, pd.Series):
            idx = other.columns if isinstance(other, pd.DataFrame) else other.index
            assert idx.nlevels == 3
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
            assert other.shape[-1] == self._dataframe.shape[-1]
            df_other = pd.DataFrame(other, columns=self._dataframe.columns)

        # append
        ignore_index = (axis == 0)
        self._dataframe = pd.concat([self._dataframe, df_other], 
                                    axis=axis, 
                                    ignore_index=ignore_index)

    def pop(self, sid=-1):
        """[TODO]"""
        item = self[sid]
        del self[sid]
        return item

    def empty(self):
        """[TODO]"""
        del self[:]

    # -----------------------------------------------------------
    # Inputs / Outputs
    # -----------------------------------------------------------

    def read(self, space_fname=None, data_fname=None, plabels=None, flabels=None):
        """[TODO]
        
        :param str space_fname: path to space file.
        :param str data_fname: path to data file.
        """
        np_sample = []
        if (space_fname is not None) and (len(self.plabels) > 0):
            if plabels is None:
                plabels = self.plabels
            np_space = self._pformater.read(space_fname, plabels)
            np_sample.append(np_space)

        if (data_fname is not None) and (len(self.plabels) > 0):
            if flabels is None:
                flabels = self.flabels
            np_data = self._fformater.read(data_fname, flabels)
            np_sample.append(np_data)

        if (np_space is not None) or (np_data is not None):
            np_sample = np.concatenate(np_sample, axis=1)
            self.append(np_sample)

    def write(self, space_fname='sample-space.json', data_fname='sample-data.json'):
        """[TODO]
        
        :param str space_fname: path to space file.
        :param str data_fname: path to data file.
        """
        if self.space.size:
            self._pformater.write(space_fname, self.space, self.plabels, self.psizes)
        if self.data.size:
            self._fformater.write(data_fname, self.data, self.flabels, self.fsizes)

    # -----------------------------------------------------------
    # Data Model
    # -----------------------------------------------------------

    def __len__(self):
        """[TODO]"""
        return len(self._dataframe)

    def __str__(self):
        """[TODO]"""
        return self._dataframe.__str__()

    def __iadd__(self, other):
        """[TODO]"""
        self.append(other)
        return self

    def __add__(self, other):
        """[TODO]"""
        new = copy(self)
        new.append(other)
        return new

    def __getitem__(self, sid):
        """[TODO]"""
        item = self._dataframe.iloc[sid]
        if item.ndim > 1:
            new = copy(self)
            new._dataframe = item
            return new
        return item.values

    def __setitem__(self, sid, value):
        """[TODO]"""
        self._dataframe.iloc[sid] = value

    def __delitem__(self, sid):
        """[TODO]"""
        sid = self._dataframe.index[sid]
        self._dataframe = self._dataframe.drop(sid)

    def __contains__(self, item):
        """[TODO]"""
        try:
            return item.values in self._dataframe.values
        except AttributeError:
            return item in self._dataframe.values

    def __iter__(self):
        """[TODO]"""
        generator = (row.values for i, row in self._dataframe.iterrows())
        return generator


def create_dataframe(dataset, clabel='space', flabels=None, fsizes=None):
    """Create a DataFrame with a 3-level column index.

    [TODO]
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
        
        if (((flabels is None) or (len(flabels) == 0))
                and ((fsizes is None) or (len(fsizes) == 0))):
            fsizes = [n_components] * n_features
        index = create_index(clabel, flabels, fsizes)

#        if flabels is None:
#            char = 'p' if clabel == 'space' else 'f'
#            flabels = ['{}{}'.format(char, i) for i in range(n_features)]
#        if len(flabels) != n_features:
#            msg = ("Detected {} features in '{}', but found {} labels"
#                   .format(n_features, clabel, len(flabels)))
#            logging.error(msg)
#            raise ValueError(msg)
#        index = pd.MultiIndex.from_product([[clabel], flabels, range(n_components)])
        dataframe = pd.DataFrame(dataset.reshape(nsample, length), columns=index)

    return dataframe


def create_index(clabel, flabels=None, fsizes=None):
    """[TODO]
    """
    if (((flabels is None) or (len(flabels) == 0))
            and ((fsizes is None) or (len(fsizes) == 0))):
        msg = 'Unable to build an index: number of labels is unknown'
        raise ValueError(msg)
    if (flabels is None) or (len(flabels) == 0):
        char = 'p' if clabel == 'space' else 'f'
        flabels = ['{}{}'.format(char, i) for i in range(len(fsizes))]
    if (fsizes is None) or (len(fsizes) == 0):
        fsizes = [1] * len(flabels)
    tuples = [(clabel, label, i) for label, size in zip(flabels, fsizes) for i in range(size)]
    return pd.MultiIndex.from_tuples(tuples)

