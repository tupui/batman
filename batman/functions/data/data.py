"""
Data
----

Define some datasets as functions. In each case, an instance of :class:`Sample`
is returned to have a consistant representation.

* :func:`el_nino`,
* :func:`tahiti`,
* :func:`mascaret`,
* :func:`marthe`.
"""
import collections
import os
import logging
import numpy as np
from batman.space import Sample


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
    labels = labels.reshape(-1, 12)[:, 0].reshape(-1, 1)
    data = data.reshape(-1, 12)

    flabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    sample = Sample(space=labels, data=data, plabels=['Year'], flabels=flabels)
    sample.desc = desc
    return sample


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

    sample = Sample(space=labels, data=data, plabels=['Year'], flabels=flabels)
    sample.desc = desc
    return sample


def mascaret():
    """Mascaret dataset."""
    desc = ("Monte-Carlo sampling simulated using MASCARET flow solver."
            " The Garonne river was used and the output consists in 14 water"
            " height observations. Two random variables are used:"
            " the friction coefficient Ks~U(15, 60) and the mass flow"
            " rate Q~N(4035, 400).")

    flabels = ['13150', '19450', '21825', '21925', '25775', '32000',
               '36131.67', '36240', '36290', '38230.45', '44557.5', '51053.33',
               '57550', '62175']

    sample = Sample(plabels=['Ks', 'Q'], flabels=flabels, pformat='npy', fformat='npy')
    sample.read(space_fname=os.path.join(PATH, 'input_mascaret.npy'),
                data_fname=os.path.join(PATH, 'output_mascaret.npy'))
    sample.desc = desc
    return sample


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

    sample = Sample(plabels=plabels, flabels=flabels)
    sample += dataset
    sample.desc = desc
    return sample
