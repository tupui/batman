# coding: utf8
"""
Mascaret module
===============

This module use a database consituted of 100000 snapshots sampled using a
Monte-Carlo and simulated using Mascaret flow solver. The Garonne river was
used and the output consists in 14 water height observations.

"""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import numpy as np
from scipy.spatial import distance
import os
import logging


class Mascaret(object):

    """Mascaret class."""

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Read the database and define the channel."""
        path = os.path.dirname(os.path.realpath(__file__))
        self.data_input = np.load(path + '/input_mascaret.npy')
        self.data_output = np.load(path + '/output_mascaret.npy')
        self.x = np.array([13150.0, 19450, 21825, 21925, 25775, 32000,
                           36131.67, 36240, 36290, 38230.45, 44557.50,
                           51053.33, 57550, 62175])
        self.logger.info("Using function Mascaret")

    def __call__(self, x):
        """Call function.

        :param list x: inputs [Ks, Q]
        :return: f(x)
        :rtype: np.array 1D (1, 14)
        """
        dists = distance.cdist([x, x], self.data_input, 'seuclidean')
        idx = np.argmin(dists, axis=1)
        idx = idx[0]

        a, b = self.data_input[idx]
        self.logger.debug("Input: {} -> Database:".format(x, [a, b]))

        return self.data_output[idx]
