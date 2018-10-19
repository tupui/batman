"""Utility for functions."""
import inspect
import numpy as np


def multi_eval(fun):
    """Detect space or unique point.

    Return the evaluation with shape (n_samples, n_features).
    """
    def wrapper_fun(self, x_n, *args, **kwargs):
        """Get evaluation from space or point.

        If the function is a Kriging instance, get and returns the variance.

        :return: function evaluation(s) [sigma(s)]
        :rtype: np.array([n_eval], n_feature)
        """
        x_n = np.atleast_2d(x_n)
        shape_eval = (len(x_n), -1)

        full = kwargs['full'] if 'full' in kwargs else False

        feval = [fun(self, x_i, *args, **kwargs) for x_i in x_n]

        if any(method in inspect.getmodule(fun).__name__
               for method in ['kriging', 'multifidelity']) or full:
            feval, sigma = zip(*feval)
            feval = np.array(feval).reshape(shape_eval)
            sigma = np.array(sigma).reshape(shape_eval)
            return feval, sigma
        feval = np.array(feval).reshape(shape_eval)
        return feval
    return wrapper_fun


def output_to_sequence(fun):
    """Convert float output to list."""
    def wrapper_fun(x_n):
        """Wrap function with a list."""
        return [fun(x_n)]
    return wrapper_fun
