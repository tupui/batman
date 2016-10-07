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
    (array([ 8.4526528 ,  3.57976035]), array([ 0.40982369,  0.05522197]))

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
from scipy.optimize import differential_evolution


class Kriging():

    """Kriging based on Gaussian Process."""

    logger = logging.getLogger(__name__)

    def __init__(self, input, output):
        """Create the predictor.

        Uses input and output to construct a predictor using Gaussian Process.
        Input is to be normalized before and depending on the number of parameters,
        the kernel is adapted to be anisotropic.

        `self.data` contains the predictors as a list(array) of the size of the `ouput`.

        :param ndarray input: The input used to generate the output.
        :param ndarray output: The observed data.

        """
        input_len = input.shape[1]
        l_scale = ((1.0),) * input_len
        scale_bounds = [(1e-03, 1000.0)] * input_len
        self.kernel = 1.0 * RBF(length_scale=l_scale,
                                length_scale_bounds=scale_bounds)
        self.data = []
        self.hyperparameter = []

        # Create a predictor per output
        for column in output.T:
            gp = GaussianProcessRegressor(kernel=self.kernel,
                                          n_restarts_optimizer=1,
                                          optimizer=self.optim_evolution)
            self.data += [gp.fit(input, column)]
            self.hyperparameter += [np.exp(gp.kernel_.theta)]

        self.logger.debug("Hyperparameters: {}".format(self.hyperparameter))

    def optim_evolution(self, obj_func, initial_theta, bounds):

        def func(args):
            return obj_func(args)[0]

        results = differential_evolution(func, bounds)
        theta_opt = results.x
        func_min = results.fun

        return theta_opt, func_min

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
            prediction[i], sigma[i] = gp.predict(point_array,
                                                 return_std=True,
                                                 return_cov=False)

        return prediction, sigma
