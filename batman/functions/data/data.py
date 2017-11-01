"""
Data
----

Define some datasets as functions. In each case, an instance of :class:`Data`
is returned to have a consistant representation.

* :class:`Data`,
* :func:`el_nino`,
* :func:`tahiti`,
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
    """

    logger = logging.getLogger(__name__)

    def __init__(self, data, desc, sample=None, plabels=None, flabels=None):
        """Dataset container.

        :param array_like data: (n_features, n_samples).
        :param str desc: dataset description.
        :param array_like sample: sampling used to create the data
          (n_features, n_samples).
        :param list(str) plabels: parameters' labels (n_features,).
        :param list(str) flabel: name of the quantities of interest
          (n_features,). It can also be a dictionary if multiple identification
          is required. Ex: ``{'Temperature': ['Jan', 'Feb', 'Mar']}``
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
        input_dim = len(self.sample) if self.sample is not None else None
        msg = ("Dataset summary:\n"
               "-> Input dimension: {}\n"
               "-> Output dimension: {}\n"
               "-> Number of samples: {}\n"
               "-> Input labels:\n{}\n"
               "-> Output labels:\n{}\n"
               ).format(input_dim, self.shape[1], self.shape[0],
                        self.plabels, self.flabels)

        return msg


# Common path
PATH = os.path.dirname(os.path.realpath(__file__))


def el_nino():
    """El Nino dataset."""
    desc = ("Averaged monthly sea surface temperature (SST) in degrees Celcius"
            " of the Pacific Ocean at 0-10°South and 90°West-80°West between"
            " 1950 and 2007.\nSource: NOAA - ERSSTv5 - Nino 1+2 at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")

    labels, data = np.loadtxt(os.path.join(PATH, 'elnino.dat'),
                              skiprows=1, usecols=(0, 2), unpack=True)
    labels = labels.reshape(-1, 12)[:, 0]
    data = data.reshape(-1, 12)

    flabels = {'Temperature': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}

    return Data(data=data, desc=desc, sample=labels,
                plabels='Year', flabels=flabels)


def tahiti():
    """Tahiti dataset."""
    desc = ("Averaged monthly sea level pressure (SLP) in millibars"
            "at Tahiti between 1951 and 2016.\nSource: NOAA - Tahiti SLP at"
            " http://www.cpc.ncep.noaa.gov/data/indices/")
    labels, *data = np.loadtxt(os.path.join(PATH, 'tahiti.dat'),
                               skiprows=4, usecols=range(0, 13), unpack=True)
    data = np.array(data).T

    flabels = {'Pressure (-1000 mbar)': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}

    return Data(data=data, desc=desc, sample=labels,
                plabels='Year', flabels=flabels)


def mascaret():
    """Mascaret dataset."""
    desc = ("Monte-Carlo sampling simulated using MASCARET flow solver."
            " The Garonne river was used and the output consists in 14 water"
            " height observations. Two random variables are used:"
            " the friction coefficient Ks~U(15, 60) and the mass flow"
            " rate Q~N(4035, 400).")
    sample = np.load(os.path.join(PATH, 'input_mascaret.npy'))
    data = np.load(os.path.join(PATH, 'output_mascaret.npy'))
    flabels = np.array([13150.0, 19450, 21825, 21925, 25775, 32000,
                        36131.67, 36240, 36290, 38230.45, 44557.50,
                        51053.33, 57550, 62175])

    return Data(data=data, desc=desc, sample=sample,
                plabels=['Ks', 'Q'], flabels={'Curvilinear abscissa': flabels})


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

    plabels = {'Hydraulic conductivity (layer)': ['per1', 'per2', 'per3'],
               'Hydraulic conductivity (zone)': ['perz1', 'perz2', 'perz3'],
               'Longitudinal dispersivity (layer)': ['d1', 'd2', 'd3'],
               'Transversal dispersivity (layer)': ['dt1', 'dt2', 'dt3'],
               'Volumetric distribution coefficient': ['kd1', 'kd2', 'kd3'],
               'Porozity': 'poros',
               'Infiltration type': ['i1', 'i2', 'i3']}

    flabels = {'Contaminant concentrations':
               ['p102K', 'p104', 'p106', 'p2.76', 'p29K',
                'p31K', 'p35K', 'p37K', 'p38', 'p4b']}

    return Data(data=dataset[:, 20:], desc=desc, sample=dataset[:, :20],
                plabels=plabels, flabels=flabels)
