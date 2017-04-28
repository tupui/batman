# coding: utf8
"""
Evofusion Class
===============

Interpolation using Evofusion method.Evofusion


Reference
---------

Forrester, Sobester et al.: Multi-Fidelity Optimization via Surrogate Modelling. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences. 2007. DOI 10.1098/rspa.2007.1900

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

        print(inputs[~inputs[:,0]==0])

        self.model_c = Kriging(inputs[1], output[1])
        try:
            inputs[0][0][0]
        except (TypeError, IndexError):
            pass
        else:
            inputs_array = np.array([np.array(inputs[0][:]).reshape(len(inputs[0]), -1),
                                    np.array(inputs[1][:]).reshape(len(inputs[1]), -1)])

        idx_cross_doe = np.where(inputs_array[0].reshape(-1,1) == inputs_array[1].reshape(1,-1))[1]
        output_err = output[0] - output[1][idx_cross_doe]

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
