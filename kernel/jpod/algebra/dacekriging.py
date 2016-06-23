try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
except ImportError:
    raise NotImplementedError('No Kriging available, without scikits.learn module.')
import numpy as np 
import logging

class Kriging():
    """Kriging interpolation using Gaussian Process method."""

    logger = logging.getLogger(__name__)

    def __init__(self, input, output):
        """Create the predictor.

            Uses input and output to find a predictor using Gaussian Process.

            self.data contains the predictors as a list.

            :param array input: The input used to generate the output.
            :param array output: The observed data.

        """
        self.kernel = 1.0 * RBF(length_scale=10., length_scale_bounds=(0.01, 100.))
        self.data = []
        self.hyperparameter = []
        for column in output.T:
            gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=10)
            self.data += [ gp.fit(input, column) ]
            self.hyperparameter += [np.exp(gp.kernel_.theta)]
        self.logger.debug("Hyperparameters: {}".format(self.hyperparameter))

    def evaluate(self, point):
        """Make a prediction. 

           From a point, make a new prediction.

           :param list point: The point to evaluate.
           :return: The predictions.
           :rtype: list

        """
        point_array = np.asarray(point).reshape(1, len(point))
        v = np.ndarray((len(self.data)))
        sigma = np.ndarray((len(self.data)))
        for i, gp in enumerate(self.data):
            v[i], sigma[i] = gp.predict(point_array, return_std=True, return_cov=False)
        return v, sigma

