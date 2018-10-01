# coding: utf8
"""
Evofusion Class
===============

Interpolation using Evofusion method.


Reference
---------

Forrester Alexander I.J, Bressloff Neil W, Keane Andy J.Optimization using
surrogate models and partially converged computational fluid dynamics
simulations. Proceedings of the Royal Society A: Mathematical, Physical and
Engineering Sciences. 2006. DOI: 10.1098/rspa.2006.1679
"""
import logging
import numpy as np
from .kriging import Kriging
from ..functions.utils import multi_eval


class Evofusion:
    """Multifidelity algorithm using Evofusion."""

    logger = logging.getLogger(__name__)

    def __init__(self, sample, data):
        """Create the predictor.

        Data are arranged as decreasing fidelity. Hence, ``sample[0]``
        corresponds to the highest fidelity.

        :param array_like sample: The sample used to generate the data.
          (fidelity, n_samples, n_features)
        :param array_like data: The observed data. (fidelity, n_samples, [n_features])

        """
        sample = np.array(sample)
        data = np.array(data)

        # Split into cheap and expensive arrays
        sample = [sample[sample[:, 0] == 0][:, 1:],
                  sample[sample[:, 0] == 1][:, 1:]]

        n_e = sample[0].shape[0]
        n_c = sample[1].shape[0]

        data = [data[:n_e].reshape((n_e, -1)),
                data[n_e:].reshape((n_c, -1))]

        # Low fidelity model
        self.model_c = Kriging(sample[1], data[1])

        idx_cross_doe = np.where(sample[0][:, None] == sample[1][None, :])[0]
        idx_cross_doe = np.unique(idx_cross_doe)

        data_err = data[0] - data[1][idx_cross_doe]

        # Error model between high and low fidelity
        self.model_err = Kriging(sample[0], data_err)

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param array_like point: The point to evaluate (n_features,).
        :return: The predictions.
        :rtype: array_like (n_features,).
        :return: The standard deviations.
        :rtype: array_like (n_features,).
        """
        f_c, sigma_c = self.model_c.evaluate(point)
        f_err, sigma_err = self.model_err.evaluate(point)
        prediction = f_c + f_err
        sigma = sigma_c + sigma_err

        return prediction, sigma
