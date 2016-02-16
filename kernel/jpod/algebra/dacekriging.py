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

except ImportError:
    class Kriging(object):

        def __init__(self, parameters, VS):
            raise NotImplementedError(
                'no Kriging available, without scikits.learn module.')
