# coding: utf8
"""
Kriging Class
=============

Interpolation using Gaussian Process method.

:Example:

::

    >> from batman.surrogate import Kriging
    >> import numpy as np
    >> sample = np.array([[2, 4], [3, 5], [6, 9]])
    >> data = np.array([[12, 1], [10, 2], [9, 4]])
    >> predictor = Kriging(sample, data)
    >> point = (5.0, 8.0)
    >> predictor.evaluate(point)
    (array([10.333,  3.591]), array([1.247, 0.694]))

Reference
---------
F. Pedregosa et al.: Scikit-learn: Machine Learning in Python. Journal of
Machine Learning Research. 2011. ArXiv ID: 1201.0490
"""
import logging
import warnings
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (WhiteKernel, Matern, ConstantKernel)
from ..misc import (NestedPool, cpu_system)
from ..functions.utils import multi_eval


class Kriging:
    """Kriging based on Gaussian Process."""

    logger = logging.getLogger(__name__)

    def __init__(self, sample, data, kernel=None, noise=False,
                 global_optimizer=True):
        r"""Create the predictor.

        Uses sample and data to construct a predictor using Gaussian Process.
        Input is to be normalized before and depending on the number of
        parameters, the kernel is adapted to be anisotropic.

        :attr:`self.data` contains the predictors as a list(array) of the size
        of the `ouput`. A predictor per line of `data` is created. This leads
        to a line of predictors that predicts a new column of `data`.

        If :attr:`noise` is a float, it will be used as :attr:`noise_level` by
        :class:`sklearn.gaussian_process.kernels.WhiteKernel`. Otherwise, if
        :attr:`noise` is ``True``, default values are use for the WhiteKernel.
        If :attr:`noise` is ``False``, no noise is added.

        A multiprocessing strategy is used:

        1. Create a process per mode, do not create if only one,
        2. Create `n_restart` (3 by default) processes by process.

        In the end, there is :math:`N=n_{restart} \times n_{modes})` processes.
        If there is not enought CPU, :math:`N=\frac{n_{cpu}}{n_{restart}}`.

        :param array_like sample: Sample used to generate the data
          (n_samples, n_features).
        :param array_like data: Observed data (n_samples, n_features).
        :param kernel: Kernel from scikit-learn.
        :type kernel: :class:`sklearn.gaussian_process.kernels`.*.
        :param float/bool noise: Noise used into kriging.
        :param bool global_optimizer: Whether to do global optimization or
          gradient based optimization to estimate hyperparameters.
        """
        sample = np.atleast_2d(sample)

        dim = sample.shape[1]
        self.model_len = data.shape[1]
        if self.model_len == 1:
            data = data.ravel()
        if kernel is not None:
            self.kernel = kernel
        else:
            # Define the model settings
            l_scale = (1.0,) * dim
            self.scale_bounds = [(0.01, 100)] * dim
            self.kernel = ConstantKernel() * Matern(length_scale=l_scale,
                                                    length_scale_bounds=self.scale_bounds)

        # Add a noise on the kernel using WhiteKernel
        if noise:
            if isinstance(noise, bool):
                noise = WhiteKernel()
            else:
                noise = WhiteKernel(noise_level=noise)
            self.kernel += noise

        # Global optimization
        args_optim = {'kernel': self.kernel, 'normalize_y': True}
        if global_optimizer:
            args_optim.update({'optimizer': self._optim_evolution,
                               'n_restarts_optimizer': 0})
            self.n_restart = 3
        else:
            args_optim.update({'n_restarts_optimizer': 10 * dim})
            self.n_restart = 1

        # Define the CPU multi-threading/processing strategy
        n_cpu_system = cpu_system()
        self.n_cpu = self.n_restart * self.model_len
        if (n_cpu_system // (self.n_restart * self.model_len) < 1)\
                or (self.n_cpu > n_cpu_system // self.n_restart):
            self.n_cpu = n_cpu_system // self.n_restart

        self.n_cpu = 1 if self.n_cpu == 0 else self.n_cpu

        def model_fitting(column):
            """Fit an instance of :class:`sklearn.GaussianProcessRegressor`."""
            gp = GaussianProcessRegressor(**args_optim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = gp.fit(sample, column)
            hyperparameter = np.exp(gp.kernel_.theta)

            # Convergence check with bounds only when kernel not user defined
            if kernel is None:
                hyper_bounds = all([i[0] < j < i[1]
                                    for i, j in zip(self.scale_bounds,
                                                    hyperparameter[1:dim+1])])

                if not hyper_bounds:
                    self.logger.warning("Hyperparameters optimization not "
                                        "converged: {}"
                                        .format(gp.kernel_))

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
        self.gp_models, self.hyperparameter = zip(*results)
        self.logger.debug("Kernels:\n{}".format([gp.kernel_ for gp in self.gp_models]))

    def _optim_evolution(self, obj_func, initial_theta, bounds):
        """Genetic optimization of the hyperparameters.

        Use DE strategy to optimize theta. The process
        is done several times using multiprocessing.
        The best results are returned.

        :param callable obj_func: function to optimize.
        :param lst(float) initial_theta: initial guess.
        :param lst(lst(float)) bounds: bounds.
        :return: theta_opt and func_min.
        :rtype: lst(float), float.
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

        :param array_like point: The point to evaluate (n_features,).
        :return: The predictions.
        :rtype: array_like (n_features,).
        """
        point_array = np.atleast_2d(point)
        prediction = np.empty((self.model_len))
        sigma = np.empty((self.model_len))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute a prediction per predictor
            for i, gp in enumerate(self.gp_models):
                prediction[i], sigma[i] = gp.predict(point_array,
                                                     return_std=True,
                                                     return_cov=False)

        return prediction, sigma
