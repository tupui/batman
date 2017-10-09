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
import os
import numpy as np
from scipy.optimize import differential_evolution
from pathos.multiprocessing import cpu_count
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from ..misc import NestedPool
from ..functions import multi_eval


class Kriging(object):

    """Kriging based on Gaussian Process."""

    logger = logging.getLogger(__name__)

    def __init__(self, sample, data):
        r"""Create the predictor.

        Uses sample and data to construct a predictor using Gaussian Process.
        Input is to be normalized before and depending on the number of
        parameters, the kernel is adapted to be anisotropic.

        :attr:`self.data` contains the predictors as a list(array) of the size
        of the `ouput`. A predictor per line of `data` is created. This leads
        to a line of predictors that predicts a new column of `data`.

        A multiprocessing strategy is used:

        1. Create a process per mode, do not create if only one,
        2. Create `n_restart` (3 by default) processes by process.

        In the end, there is :math:`N=n_{restart} \times n_{modes})` processes.
        If there is not enought CPU, :math:`N=\frac{n_{cpu}}{n_restart}`.

        :param array_like sample: The sample used to generate the data. (n_samples, n_features)
        :param array_like data: The observed data. (n_samples, [n_features])

        """
        try:
            sample[0][0]
        except (TypeError, IndexError):
            pass
        else:
            sample = np.array(sample).reshape(len(sample), -1)

        sample_len = sample.shape[1]
        self.model_len = data.shape[1]
        if self.model_len == 1:
            data = data.ravel()

        # Define the model settings
        l_scale = ((1.0),) * sample_len
        scale_bounds = [(1e-03, 1000.0)] * sample_len
        self.kernel = 1.0 * RBF(length_scale=l_scale,
                                length_scale_bounds=scale_bounds)

        self.n_restart = 3
        # Define the CPU multi-threading/processing strategy
        try:
            n_cpu_system = cpu_count()
        except NotImplementedError:
            n_cpu_system = os.sysconf('SC_NPROCESSORS_ONLN')
        self.n_cpu = self.model_len
        if n_cpu_system // (self.n_restart * self.model_len) < 1:
            self.n_cpu = n_cpu_system // self.n_restart

        def model_fitting(column):
            """Fit an instance of :claa:`sklearn.GaussianProcessRegressor`."""
            gp = GaussianProcessRegressor(kernel=self.kernel,
                                          n_restarts_optimizer=0,
                                          optimizer=self.optim_evolution)
            data = gp.fit(sample, column)
            hyperparameter = np.exp(gp.kernel_.theta)

            return data, hyperparameter

        # Create a predictor per data, parallelize if several data
        if self.model_len > 1:
            pool = NestedPool(self.n_cpu)
            results = pool.imap(model_fitting, data.T)
            results = list(results)
            pool.terminate()
        else:
            results = [model_fitting(data)]

        # Gather results
        self.data, self.hyperparameter = zip(*results)

        self.logger.debug("Hyperparameters: {}".format(self.hyperparameter))

    def optim_evolution(self, obj_func, initial_theta, bounds):
        """Genetic optimization of the hyperparameters.

        Use DE strategy to optimize theta. The process
        is done several times using multiprocessing.
        The best results are returned.

        :param callable obj_func: function to optimize
        :param lst(float) initial_theta: initial guess
        :param lst(lst(float)) bounds: bounds
        :return: theta_opt and func_min
        :rtype: lst(float), float
        """
        def func(args):
            """Get the output from sklearn."""
            return obj_func(args)[0]

        def fork_optimizer(i):
            """Optimize hyperparameters."""
            results = differential_evolution(func, bounds,
                                             tol=0.001, popsize=15+i)
            theta_opt = results.x
            func_min = results.fun
            return theta_opt, func_min

        pool = NestedPool(self.n_restart)
        results = pool.imap(fork_optimizer, range(self.n_restart))

        # Gather results
        results = list(results)
        pool.terminate()

        theta_opt, func_min = zip(*results)

        # Find best results
        min_idx = np.argmin(func_min)
        func_min = func_min[min_idx]
        theta_opt = theta_opt[min_idx]

        return theta_opt, func_min

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
        point_array = np.asarray(point).reshape(1, -1)
        prediction = np.empty((self.model_len))
        sigma = np.empty((self.model_len))

        # Compute a prediction per predictor
        for i, gp in enumerate(self.data):
            prediction[i], sigma[i] = gp.predict(point_array,
                                                 return_std=True,
                                                 return_cov=False)

        return prediction, sigma
