# coding: utf8
"""
Utility for functions.
"""
import numpy as np
import inspect


def multi_eval(fun):
    """Decorator to detect space or unique point."""
    def wrapper_fun(self, x, *args, **kwargs):
        """Get evaluation from space or point.

        If the function is a Kriging instance, get and returns the variance.

        :return: function evaluation(s) [sigma(s)]
        :rtype: np.array([n_eval], n_feature)
        """
        try:
            x[0][0]
            n_eval = len(x)
            shape_eval = (n_eval, -1)
        except (TypeError, IndexError):
            x = [x]
            n_eval = 1
            shape_eval = (-1)

        f = [None] * n_eval

        for i, x_i in enumerate(x):
            f[i] = fun(self, x_i, *args, **kwargs)

        if any(method in inspect.getmodule(fun).__name__
               for method in ['kriging', 'multifidelity']):
            sigma = [None] * n_eval
            for i, _ in enumerate(x):
                f[i], sigma[i] = f[i]
            f = np.array(f).reshape(shape_eval)
            sigma = np.array(sigma).reshape(shape_eval)
            return f, sigma
        f = np.array(f).reshape(shape_eval)
        return f
    return wrapper_fun


def output_to_sequence(fun):
    """Convert float output to list."""
    def wrapper_fun(x):
        return [fun(x)]
    return wrapper_fun
