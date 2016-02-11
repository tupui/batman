try:
    from sklearn.gaussian_process import GaussianProcess

    class Kriging(object):
        """Wrapper to class-less dacekriging module."""

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

        def error_estimation(self, sampled_space, discretization):
            """Compute the Gaussian confidence interval of the Kriging
            space : Sampling space ::class Space
            discretization : discretization of the space to search the maximum error
            """
            import numpy as np
<<<<<<< HEAD
            
=======
            import space
            #Some comment
>>>>>>> c56c56a1b07e8466d77c6cde6333fd4995c16776
            
            bounds = np.asarray(sampled_space.corners)
            limit_number = bounds.shape[1] ** discretization
            uniform_space = space.Space(sampled_space.corners,limit_number,plot=False)
            x = uniform_space.sampling('uniform', discretization)
            mesh = np.asarray(x)
            y_pred = np.ndarray((len(x)))
            MSE = np.ndarra((len(x)))
            sigma_pred = np.ndarra((len(x)))
            for i, d in enumerate(self.data):
                y_pred[i], MSE[i] = d.predict(mesh, eval_MSE = True)
		sigma_pred[i] = np.sqrt(MSE[i])
            return sigma_pred

except ImportError:
    class Kriging(object):

        def __init__(self, parameters, VS):
            raise NotImplementedError(
                'no Kriging available, without scikits.learn module.')
