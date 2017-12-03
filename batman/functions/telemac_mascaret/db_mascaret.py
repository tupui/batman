"""
Mascaret module
===============

This module use a database consituted of 100000 snapshots sampled using a
Monte-Carlo and simulated using Mascaret flow solver. The Garonne river was
used and the output consists in 14 water height observations.

"""
import logging
import numpy as np
from scipy.spatial import distance
from ..utils import multi_eval
from .. import mascaret


class Mascaret(object):
    """Mascaret class."""

    logger = logging.getLogger(__name__)

    def __init__(self):
        """Read the database and define the channel."""
        dataset = mascaret()
        self.data_input, self.data_output = dataset.sample, dataset.data
        self.data_input = np.array(self.data_input.tolist())
        self.data_output = np.array(self.data_output.tolist())
        self.x = [float(label) for label in dataset.flabels]
        self.d_out = 14
        self.d_in = 2

        self.s_second_full = np.array([[[0., -0.03020037], [-0.03020037, 0.]],
                                       [[0., -0.03881756], [-0.03881756, 0.]],
                                       [[0., -0.04251338], [-0.04251338, 0.]],
                                       [[0., -0.0426679], [-0.0426679, 0.]],
                                       [[0., -0.04966869], [-0.04966869, 0.]],
                                       [[0., -0.03019764], [-0.03019764, 0.]],
                                       [[0., -0.02242943], [-0.02242943, 0.]],
                                       [[0., -0.02222612], [-0.02222612, 0.]],
                                       [[0., -0.02279468], [-0.02279468, 0.]],
                                       [[0., -0.02418406], [-0.02418406, 0.]],
                                       [[0., -0.0261341], [-0.0261341, 0.]],
                                       [[0., -0.03064743], [-0.03064743, 0.]],
                                       [[0., -0.03868296], [-0.03868296, 0.]],
                                       [[0., 0.00282709], [0.00282709, 0.]]])
        self.s_first_full = np.array([[0.10107270978302792, 0.8959930919247889],
                                      [0.18120319110283745, 0.8273998127843324],
                                      [0.23451964408156595, 0.7800462867106654],
                                      [0.23685958750154648, 0.7779717432101445],
                                      [0.4098437677793702, 0.6191189440935079],
                                      [0.7751331218908732, 0.2495823405408702],
                                      [0.8742876967854374, 0.1451778693930793],
                                      [0.8742603876671973, 0.14530386765866726],
                                      [0.8722028358385836, 0.14773687242711417],
                                      [0.8714371617522463, 0.14967046813425272],
                                      [0.8579152536671296, 0.1656617547600983],
                                      [0.8146262099773994, 0.21113331675809266],
                                      [0.7333161075183993, 0.2961806754581718],
                                      [-0.0009836372837096455, 0.9692830624285731]])
        self.s_total_full = np.array([[0.1116597072943773, 0.875221090352921],
                                      [0.1965660368992014, 0.7969560350458335],
                                      [0.2532846779268521, 0.7456644739879672],
                                      [0.25573942637517416, 0.7433651143730953],
                                      [0.4359824833681346, 0.5741773108039986],
                                      [0.8071753364503617, 0.21183223499031062],
                                      [0.9023827296735317, 0.11319757917424246],
                                      [0.902784341465201, 0.11326732390157283],
                                      [0.9005785778706416, 0.11547577068029803],
                                      [0.8993969774433382, 0.11685853224140431],
                                      [0.8849832956790847, 0.1308629902137176],
                                      [0.8436987264754154, 0.1736232878097748],
                                      [0.7644789560062502, 0.25458734925061216],
                                      [-0.0017544241163383715, 0.9730845491814776]])

        self.s_second = np.array([[0., 0.01882222], [0.01882222, 0.]])
        self.s_first = np.array([0.75228948, 0.21880863])
        self.s_total = np.array([0.76464851, 0.24115337])

        self.logger.info("Using function Mascaret")

    @multi_eval
    def __call__(self, x):
        """Call function.

        :param array_like x: inputs [Ks, Q].
        :return: f(x).
        :rtype: array_like 1D (1, 14).
        """
        dists = distance.cdist([x, x], self.data_input, 'seuclidean')
        idx = np.argmin(dists, axis=1)
        idx = idx[0]

        _ks, discharge = self.data_input[idx]
        self.logger.debug("Input: {} -> Database: {}".format(x, [_ks, discharge]))

        return self.data_output[idx]
