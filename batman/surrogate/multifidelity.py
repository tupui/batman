# coding: utf8
"""
Evofusion Class
===============

Interpolation using Evofusion method.Evofusion


Reference
---------

Optimization using surrogate models and partially converged computational fluid dynamics simulations

"""
import numpy as np
import logging
from .kriging import Kriging
from ..functions import multi_eval


class Evofusion(object):

    """Multifidelity algorithm using Evofusion."""

    logger = logging.getLogger(__name__)

    def __init__(self, inputs, output):
        """Create the predictor.

        Data are arranged as decreasing fidelity. Hence, ``inputs[0]``
        corresponds to the highest fidelity.

        :param ndarray inputs: The inputs used to generate the output. (fidelity, nb snapshots, nb parameters)
        :param ndarray output: The observed data. (fidelity, nb snapshots, [nb output dim])

        """
        inputs = np.array(inputs)
        output = np.array(output)

        # Split into cheap and expensive arrays
        inputs = [inputs[inputs[:, 0] == 0][:, 1:],
                  inputs[inputs[:, 0] == 1][:, 1:]]

        n_e = inputs[0].shape[0]
        n_c = inputs[1].shape[0]

        output = [output[:n_e].reshape((n_e, -1)),
                  output[n_e:].reshape((n_c, -1))]

        # Low fidelity model
        self.model_c = Kriging(inputs[1], output[1])

        idx_cross_doe = np.where(inputs[0][:, None] == inputs[1][None, :])[0]
        idx_cross_doe = np.unique(idx_cross_doe)

        output_err = output[0] - output[1][idx_cross_doe]

        # Error model between high and low fidelity
        self.model_err = Kriging(inputs[0], output_err)

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param tuple(float) point: The point to evaluate.
        :return: The predictions.
        :rtype: lst
        :return: The standard deviations.
        :rtype: lst

        """
        f_c, sigma_c = self.model_c.evaluate(point)
        f_err, sigma_err = self.model_err.evaluate(point)
        prediction = f_c + f_err
        sigma = sigma_c + sigma_err

        return prediction, sigma
