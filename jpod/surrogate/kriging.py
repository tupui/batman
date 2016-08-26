# -*- coding: utf-8 -*-
"""
Kriging Class
=============

Interpolation using Gaussian Process method.

:Example:
::

    >> from kriging import Kriging
    >> import numpy as np
    >> input = np.array([[2, 4], [3, 5], [6, 9]])
    >> output = np.array([[12, 1], [10, 2], [9, 4]])
    >> predictor = Kriging(input, output)
    >> point = (5.0, 8.0)
    >> predictor.evaluate(point)
    (array([ 8.4363201 ,  3.77281636]), array([ 0.76631883,  0.15439491]))

Reference
---------

F. Pedregosa et al.: Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011. ArXiv ID: 1201.0490

"""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
except ImportError:
    raise NotImplementedError('No Kriging available, without scikits.learn module.')
import numpy as np
import logging


class Kriging():

    """Kriging based on Gaussian Process."""

    logger = logging.getLogger(__name__)

    def __init__(self, input, output):
        """Create the predictor.

        Uses input and output to construct a predictor using Gaussian Process.

        `self.data` contains the predictors as a list(array) of the size of the `ouput`.

        :param ndarray input: The input used to generate the output.
        :param ndarray output: The observed data.

        """
        self.kernel = 1.0 * RBF(length_scale=10., length_scale_bounds=(0.01, 100.))
        self.data = []
        self.hyperparameter = []

        # Create a predictor per output
        for column in output.T:
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
            self.data += [gp.fit(input, column)]
            self.hyperparameter += [np.exp(gp.kernel_.theta)]

        self.logger.debug("Hyperparameters: {}".format(self.hyperparameter))

    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param tuple(float) point: The point to evaluate.
        :return: The predictions.
        :rtype: lst
        :return: The standard deviations.
        :rtype: lst

        """
        point_array = np.asarray(point).reshape(1, len(point))

        prediction = np.ndarray((len(self.data)))
        sigma = np.ndarray((len(self.data)))

        # Compute a prediction per predictor
        for i, gp in enumerate(self.data):
            prediction[i], sigma[i] = gp.predict(point_array, return_std=True, return_cov=False)

        return prediction, sigma