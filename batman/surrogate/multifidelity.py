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

    def __init__(self, input, output):
        """Create the predictor.

        Data are arranged as decreasing fidelity. Hence, ``input[0]``
        corresponds to the highest fidelity.

        :param ndarray input: The input used to generate the output. (fidelity, nb snapshots, nb parameters)
        :param ndarray output: The observed data. (fidelity, nb snapshots, [nb output dim])

        """
        self.model_c = Kriging(input[1], output[1])
        try:
            input[0][0][0]
        except (TypeError, IndexError):
            pass
        else:
            input_array = np.array([np.array(input[0][:]).reshape(len(input[0]), -1),
                                    np.array(input[1][:]).reshape(len(input[1]), -1)])

        idx_cross_doe = np.where(input_array[0].reshape(-1,1) == input_array[1].reshape(1,-1))[1]
        output_err = output[0] - output[1][idx_cross_doe]

        self.model_err = Kriging(input[0], output_err)

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
        f = f_c + f_err
        sigma = sigma_c + sigma_err

        return f, sigma
