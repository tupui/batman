"""
Data
----

Define some datasets as functions. In each case, an instance of :class:`Data`
is returned to have a consistant representation.

* :class:`Data`,
* :func:`el_nino`,
* :func:`tahiti`,
* :func:`mascaret`,
* :func:`marthe`.
"""
import collections
import os
import logging
import numpy as np


class Data(collections.Mapping):

    """Wrap datasets into a Mapping container.

    Store a dataset allong with some informations about it.
    :attr:`data` corresponds to model's output and :attr:`sample` to the
    corresponding inputs.

    Structured array are created for both :attr:`data` and :attr:`sample`.
    This allows to access values using either normal indexing or attribute
    indexing by use of labels' features.

    If required, :meth:`toarray` convert both :attr:`data` and :attr:`sample`
    into regular arrays.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, data, desc, sample=None, plabels=None, flabels=None):
        """Dataset container.

        Both :attr:`data` and :attr:`sample` are required to be 2D arrays.
        Thus with one feature, shape must be (n_samples, 1).

        :param array_like data: (n_features, n_samples).
        :param str desc: dataset description.
        :param array_like sample: sampling used to create the data
          (n_features, n_samples).
        :param list(str) plabels: parameters' labels (n_features,).
        :param list(str) flabel: name of the quantities of interest
          (n_features,).
        """
        self.desc = desc
        self.plabels = plabels
        self.flabels = flabels

        # Dataset conversion to structured arrays
        if self.plabels is not None:
            dt_sample = {'names': self.plabels,
                         'formats': ['f8'] * len(self.plabels)}
        else:
            dt_sample = None
        if self.flabels is not None:
            dt_data = {'names': self.flabels,
                       'formats': ['f8'] * len(self.flabels)}
        else:
            dt_data = None

        self.shape = data.shape if sample is None else (sample.shape, data.shape)
        self.in_shape = sample.shape[1] if sample is not None else None

        self.data = np.asarray([tuple(datum) for datum in data], dtype=dt_data)
        self.sample = np.asarray([tuple(snap) for snap in sample],
                                 dtype=dt_sample) if sample is not None else None

        if (self.sample is not None) and (len(self.sample) != len(self.data)):
            self.logger.error("Sample shape not consistent with data shape: "
                              "{} != {}".format(len(self.sample), len(self.data)))
            raise SystemError

    def toarray(self):
        """Convert the structured array to regular arrays.

        This will prevent the hability to access :attr:`sample` and
        :attr:`data` using attributes from respective labels.
        """
        self.data = self.data.view((self.data.dtype[0],
                                    len(self.data.dtype.names)))
        self.sample = self.sample.view((self.sample.dtype[0],
                                        len(self.sample.dtype.names)))

    def __getitem__(self, key):
        """Return the corresponding data or a tuple of (sample, data)."""
        return self.data[key] if self.sample is None else\
            (self.sample[key], self.data[key])

    def __iter__(self):
        """Iterate over data or a zip of sample and data."""
        return iter(self.data) if self.sample is None else\
            iter(zip(self.sample, self.data))

    def __len__(self):
        """Based on the number of sample."""
        return len(self.data)

    def __str__(self):
        """Describe and summarize."""
        return self.desc + '\n\n' + self.__repr__()

    def __repr__(self):
        """Summarize the container."""
        msg = ("Dataset summary:\n"
               "-> Input dimension: {}\n"
               "-> Output dimension: {}\n"
               "-> Number of samples: {}\n"
               "-> Input labels:\n{}\n"
               "-> Output labels:\n{}\n"
              ).format(self.in_shape, self.shape[1][1], self.shape[0][0],
                       self.plabels, self.flabels)

        return msg


# Common path
PATH = os.path.dirname(os.path.realpath(__file__))


def el_nino():
    """El Nino dataset."""
    desc = ("Averaged monthly sea surface temperature (SST) in degrees Celcius"
            " of the Pacific Ocean at 0-10 deg South and 90-80 deg West"
            " between 1950 and 2007.\nSource: NOAA - ERSSTv5 - Nino 1+2 at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")

    labels, data = np.loadtxt(os.path.join(PATH, 'elnino.dat'),
                              skiprows=1, usecols=(0, 2), unpack=True)
    labels = labels.reshape(-1, 12)[:, 0]
    data = data.reshape(-1, 12)

    flabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return Data(data=data, desc=desc, sample=labels.reshape(-1, 1),
                plabels=['Year'], flabels=flabels)


def tahiti():
    """Tahiti dataset."""
    desc = ("Averaged monthly sea level pressure (SLP) in millibars"
            "at Tahiti between 1951 and 2016.\nSource: NOAA - Tahiti SLP at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")
    dataset = np.loadtxt(os.path.join(PATH, 'tahiti.dat'),
                         skiprows=4, usecols=range(0, 13))

    labels = dataset[:, 0].reshape(-1, 1)
    data = dataset[:, 1:]

    flabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return Data(data=data, desc=desc, sample=labels,
                plabels=['Year'], flabels=flabels)


def mascaret():
    """Mascaret dataset."""
    desc = ("Monte-Carlo sampling simulated using MASCARET flow solver."
            " The Garonne river was used and the output consists in 14 water"
            " height observations. Two random variables are used:"
            " the friction coefficient Ks~U(15, 60) and the mass flow"
            " rate Q~N(4035, 400).")
    sample = np.load(os.path.join(PATH, 'input_mascaret.npy'))
    data = np.load(os.path.join(PATH, 'output_mascaret.npy'))
    flabels = ['13150', '19450', '21825', '21925', '25775', '32000',
               '36131.67', '36240', '36290', '38230.45', '44557.5', '51053.33',
               '57550', '62175']

    return Data(data=data, desc=desc, sample=sample,
                plabels=['Ks', 'Q'], flabels=flabels)


def marthe():
    """MARTHE dataset."""
    desc = ("In 2005, CEA (France) and Kurchatov Institute (Russia) developed"
            " a model of strontium 90 migration in a porous water-saturated"
            " medium. The scenario concerned the temporary storage of"
            " radioactive waste (STDR) in a site close to Moscow. The main"
            " purpose was to predict the transport of 90Sr between 2002 and"
            " 2010, in order to determine the aquifer contamination. The"
            " numerical simulation of the 90Sr transport in the upper aquifer"
            " of the site was realized via the MARTHE code"
            " (developed by BRGM, France).")
    dataset = np.loadtxt(os.path.join(PATH, 'marthe.dat'), skiprows=1)

    plabels = ['per1', 'per2', 'per3', 'perz1', 'perz2', 'perz3', 'perz4',
               'd1', 'd2', 'd3', 'dt1', 'dt2', 'dt3', 'kd1', 'kd2', 'kd3',
               'poros', 'i1', 'i2', 'i3']

    flabels = ['p102K', 'p104', 'p106', 'p2.76', 'p29K',
               'p31K', 'p35K', 'p37K', 'p38', 'p4b']

    return Data(data=dataset[:, 20:], desc=desc, sample=dataset[:, :20],
                plabels=plabels, flabels=flabels)
