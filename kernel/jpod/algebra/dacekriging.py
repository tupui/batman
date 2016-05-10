try:
    from sklearn.gaussian_process import GaussianProcess
    import numpy as np 
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel 


    class Kriging(object):
        """Wrapper to class-less dacekriging module."""

        def __init__(self, input, output):
            #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(0.1, 10.))
	    kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(0.01, 1000.)) + WhiteKernel(noise_level=1.0e-5, noise_level_bounds=(1.0e-10, 10.0))
	    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
            self.data = []
            for column in output.T:
                self.data += [ gp.fit(input, column) ]
    
        def evaluate(self, point):
            point_array = np.asarray(point).reshape(1, len(point))
            v = np.ndarray((len(self.data)))
            for i, gp in enumerate(self.data):
                v[i] = gp.predict(point_array, return_std=False, return_cov=False)
            return v


    class Kriging_test(object):
        def __init__(self, parameters, VS):
            """
            parameters : numpy array of points coordinates
            VS         : numpy array, product of pod V and S matrices
            """
            self.data = []
            for column in VS.T:
                self.data += [
                    GaussianProcess(
                        theta0=10.,
                        thetaL=1e-15,
                        thetaU=10.).fit(
                        parameters,
                        column)]

        def evaluate(self, point):
            """Compute a prediction.

            point : point coordinates
            """
            import numpy as np

            point_array = np.asarray(point).reshape(1, len(point))
            v = np.ndarray((len(self.data)))
            for i, d in enumerate(self.data):
                v[i] = d.predict(point_array, eval_MSE=False, batch_size=None)
            return v

except ImportError:
    class Kriging(object):

        def __init__(self, parameters, VS):
            raise NotImplementedError(
                'no Kriging available, without scikits.learn module.')
