# coding: utf8
"""
Kriging Class
=============

Interpolation using Gaussian Process method.

:Example:

::

    >> from kriging import Kriging
    >> import numpy as np
    >> sample = np.array([[2, 4], [3, 5], [6, 9]])
    >> data = np.array([[12, 1], [10, 2], [9, 4]])
    >> predictor = Kriging(sample, data)
    >> point = (5.0, 8.0)
    >> predictor.evaluate(point)
    (array([ 8.4526528 ,  3.57976035]), array([ 0.40982369,  0.05522197]))

Reference
---------
F. Pedregosa et al.: Scikit-learn: Machine Learning in Python. Journal of
Machine Learning Research. 2011. ArXiv ID: 1201.0490
"""
import logging
import warnings
import numpy as np
from ..misc import (NestedPool, cpu_system)
from ..functions.utils import multi_eval


class SklearnRegressor(object):
    """Interface to Scikit-learn regressors."""

    logger = logging.getLogger(__name__)

    def __init__(self, sample, data, regressor):
        r"""Create the predictor.

        Uses sample and data to construct a predictor using sklearn.
        Input is to be normalized before and depending on the number of
        parameters, the kernel is adapted to be anisotropic.

        :param array_like sample: Sample used to generate the data
          (n_samples, n_features).
        :param array_like data: Observed data (n_samples, n_features).
        :param regressor: Scikit-Learn regressor.
        :type regressor: Either regressor object or
          str(:module:`sklearn.ensemble`.Regressor)
        """
        try:
            sample[0][0]
        except (TypeError, IndexError):
            pass
        else:
            sample = np.array(sample).reshape(len(sample), -1)

        dim = sample.shape[1]
        self.model_len = data.shape[1]
        if self.model_len == 1:
            data = data.ravel()

        # Define the CPU multi-threading/processing strategy
        n_cpu_system = cpu_system()
        self.n_cpu = n_cpu_system if n_cpu_system // (self.model_len) < 1 else\
            self.model_len
        self.n_cpu = 1 if self.n_cpu == 0 else self.n_cpu

        try:
            # Regressor is already a sklearn object
            self.logger.debug('Regressor info: {}'.format(regressor.get_params))
        except AttributeError:
            # Instanciate regressor from str
            try:
                regressor = eval('ske.' + regressor, {'__builtins__': None},
                                 {'ske': __import__('sklearn.ensemble')})
            except (TypeError, AttributeError):
                raise AttributeError('Regressor unknown from sklearn.')

            self.logger.debug('Regressor info: {}'.format(regressor.get_params))

        def model_fitting(column):
            """Fit an instance of :class:`sklearn.ensemble`.Regressor."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = regressor.fit(sample, column)

            return data

        # Create a predictor per data, parallelize if several data
        if self.model_len > 1:
            pool = NestedPool(self.n_cpu)
            results = pool.imap(model_fitting, data.T)
            self.regressor = list(results)
            pool.terminate()
        else:
            self.regressor = [model_fitting(data)]

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param array_like point: The point to evaluate (n_features,).
        :return: The predictions.
        :rtype: array_like (n_features,).
        """
        point_array = np.asarray(point).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute a prediction per predictor
            prediction = [reg.predict(point_array) for reg in self.regressor]

        return np.array(prediction)
