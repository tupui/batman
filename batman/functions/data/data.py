"""
Data
----

Define function related to datasets.

* :class:`Data`,
* :func:`el_nino`.
"""
import collections
import os
import logging
import numpy as np


class Data(collections.Mapping):

    """Wrap datasets into a Mapping container."""

    logger = logging.getLogger(__name__)

    def __init__(self, data, desc, sample=None, plabels=None, flabels=None):
        """Dataset container.

        :param array_like data: (n_features, n_samples).
        :param str desc: dataset description.
        :param array_like sample: sampling used to create the data
          (n_features, n_samples).
        :param list(str) plabels: parameters' labels (n_features,).
        :param list(str) flabel: name of the quantities of interest (n_features,).
        """
        self.desc = desc
        self.data = np.asarray(data)
        self.sample = np.asarray(sample) if sample is not None else None

        if (self.sample is not None) and (len(self.sample) != len(self.data)):
            self.logger.error("Sample shape not consistent with data shape: "
                              "{} != {}".format(len(self.sample), len(self.data)))
            raise SystemError

        self.plabels = plabels
        self.flabels = flabels

    @property
    def shape(self):
        """Numpy like shape."""
        return self.data.shape if self.sample is None else\
            (self.sample.shape, self.data.shape)

    def __getitem__(self, key):
        return self.data[key] if self.sample is None else\
            (self.sample[key], self.data[key])

    def __iter__(self):
        return iter(self.data) if self.sample is None else\
            (iter(self.sample), iter(self.data))

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        input_dim = len(self.sample) if self.sample is not None else '-'
        msg = ("Dataset summary:\n"
               "-> Input dimension: {}\n"
               "-> Output dimension: {}\n"
               "-> Number of samples: {}\n"
               "-> Input labels:\n{}\n"
               "-> Output labels:\n{}\n"
               ).format(input_dim, self.shape[1], self.shape[0],
                        self.plabels, self.flabels)

        return msg


def el_nino():
    """El Nino dataset."""
    # Water surface temperature data from:
    # http://www.cpc.ncep.noaa.gov/data/indices/
    path = os.path.dirname(os.path.realpath(__file__))
    labels, data = np.loadtxt(os.path.join(path, 'functional_dataset/elnino.dat'),
                              skiprows=1, usecols=(0, 2), unpack=True)
    labels = labels.reshape(-1, 12)[:, 0]
    data = data.reshape(-1, 12)

    # labels_tahiti, *data_tahiti = np.loadtxt(os.path.join(path, 'functional_dataset/tahiti.dat'),
    #                                          skiprows=4, usecols=range(0, 13),
    #                                          unpack=True)
    # data_tahiti = np.array(data_tahiti).T

    return Data()
