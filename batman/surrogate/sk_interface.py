# coding: utf8
"""
SklearnRegressor Class
======================

Interpolation using regressors from Scikit-Learn.

:Example:

::

    >> from batman.surrogate import SklearnRegressor
    >> import numpy as np
    >> sample = np.array([[2, 4], [3, 5], [6, 9]])
    >> data = np.array([[12, 1], [10, 2], [9, 4]])
    >> regressor = 'RandomForestRegressor()'
    >> predictor = SklearnRegressor(sample, data, regressor)
    >> point = (5.0, 8.0)
    >> predictor.evaluate(point)
    array([9.7, 2.9])

"""
import logging
import warnings
import numpy as np
from ..misc import (NestedPool, cpu_system)
from ..functions.utils import multi_eval


class SklearnRegressor:
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
          str(:mod:`sklearn.ensemble`.Regressor)
        """
        sample = np.atleast_2d(sample)

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
            self.logger.debug('Regressor info:\n{}'.format(regressor.get_params))
        except AttributeError:
            # Instanciate regressor from str
            try:
                regressor = eval('ske.' + regressor, {'__builtins__': None},
                                 {'ske': __import__('sklearn').ensemble})
            except (TypeError, AttributeError):
                raise AttributeError('Regressor unknown from sklearn.')

            self.logger.debug('Regressor info:\n{}'.format(regressor.get_params))

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
        point_array = np.atleast_2d(point)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute a prediction per predictor
            prediction = [reg.predict(point_array) for reg in self.regressor]

        return np.array(prediction)
