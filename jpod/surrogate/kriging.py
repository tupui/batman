# coding: utf8
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

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import logging
from scipy.optimize import differential_evolution
from pathos.multiprocessing import ThreadingPool, ProcessingPool, cpu_count
import os


class Kriging():

    """Kriging based on Gaussian Process."""

    logger = logging.getLogger(__name__)

    def __init__(self, input, output):
        r"""Create the predictor.

        Uses input and output to construct a predictor using Gaussian Process.
        Input is to be normalized before and depending on the number of
        parameters, the kernel is adapted to be anisotropic.

        `self.data` contains the predictors as a list(array) of the size
        of the `ouput`.

        A multiprocessing strategy is used:

        1. Create a thread per mode, do not create if only one,
        2. Create `n_restart` (3 by default) processes by thread.

        In the end, there is :math:`N=n_{restart} \times n_{modes})` processes.
        If there is not enought CPU, :math:`N=\frac{n_{cpu}}{n_restart}`.

        :param ndarray input: The input used to generate the output.
        :param ndarray output: The observed data.

        """
        input_len = input.shape[1]
        l_scale = ((1.0),) * input_len
        scale_bounds = [(1e-03, 1000.0)] * input_len
        self.kernel = 1.0 * RBF(length_scale=l_scale,
                                length_scale_bounds=scale_bounds)
        self.model_len = len(output.T)
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
            gp = GaussianProcessRegressor(kernel=self.kernel,
                                          n_restarts_optimizer=0,
                                          optimizer=self.optim_evolution)
            data = gp.fit(input, column)
            hyperparameter = np.exp(gp.kernel_.theta)

            return data, hyperparameter

        # Create a predictor per output, parallelize if several output
        if self.model_len > 1:
            pool = ThreadingPool(self.n_cpu)
            results = pool.imap(model_fitting, output.T)
        else:
            results = [model_fitting(output.T[0])]

        # Gather results
        results = list(results)
        self.data = [None] * self.model_len
        self.hyperparameter = [None] * self.model_len
        for i in range(self.model_len):
            self.data[i], self.hyperparameter[i] = results[i]

        self.logger.debug("Hyperparameters: {}".format(self.hyperparameter))

    def optim_evolution_sequential(self, obj_func, initial_theta, bounds):
        """Same as :func:`optim_evolution` without restart."""
        def func(args):
            return obj_func(args)[0]

        results = differential_evolution(func, bounds, tol=0.001, popsize=20)
        theta_opt = results.x
        func_min = results.fun

        return theta_opt, func_min

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
            return obj_func(args)[0]

        def fork_optimizer(i):
            results = differential_evolution(func, bounds,
                                             tol=0.001, popsize=15+i)
            theta_opt = results.x
            func_min = results.fun
            return theta_opt, func_min

        pool = ProcessingPool(self.n_restart)
        results = pool.imap(fork_optimizer, range(self.n_restart))

        # Gather results
        results = list(results)
        theta_opt = [None] * self.n_restart
        func_min = [None] * self.n_restart

        for i in range(self.n_restart):
            theta_opt[i], func_min[i] = results[i]

        # Find best results
        min_idx = np.argmin(func_min)
        func_min = func_min[min_idx]
        theta_opt = theta_opt[min_idx]

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
