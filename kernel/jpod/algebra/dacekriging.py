try:
    from sklearn.gaussian_process import GaussianProcess
    import numpy as np 
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    class Kriging():
        """Kriging interpolation using Gaussian Process method."""

        def __init__(self, input, output):
	    """Create the predictor.

            Uses input and output to find a predictor using Gaussian Process.

            self.data contains the predictors as a list.

            :param array input: The input used to generate the output.
            :param array output: The observed data.
	    
	    """
	    self.kernel = 1.0 * RBF(length_scale=10., length_scale_bounds=(0.01, 100.))
	    gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
            self.data = []
            for column in output.T:
                self.data += [ gp.fit(input, column) ]
    
        def evaluate(self, point):
	    """Make a prediction. 

            From a point, make a new prediction.

            :param list point: The point to evaluate.
            :return: The predictions.
            :rtype: list

            """
            point_array = np.asarray(point).reshape(1, len(point))
            v = np.ndarray((len(self.data)))
            for i, gp in enumerate(self.data):
                v[i] = gp.predict(point_array, return_std=False, return_cov=False)
            return v

except ImportError:
    class Kriging():

        def __init__(self, parameters, VS):
            raise NotImplementedError(
                'No Kriging available, without scikits.learn module.')
