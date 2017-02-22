# coding: utf8
"""
Utility for functions.
"""
import numpy as np
from .. import space
import inspect


def multi_eval(fun):
    """Decorator to detect space or unique point."""
    def wrapper_fun(self, x):
        """Get evaluation from space or point.

        If the function is a Kriging instance, get and returns the variance.

        :return: function evaluation(s) [sigma(s)]
        :rtype: np.array([n_eval], n_feature)
        """
        try:
            x[0][0]
        except (TypeError, IndexError):
            x = [x]

        n_eval = len(x)
        f = [None] * n_eval

        if n_eval != 1:
            shape_eval = (n_eval, -1)
        else:
            shape_eval = (-1)

        if 'kriging' in inspect.getmodule(fun).__name__:
            sigma = [None] * n_eval
            for i, x_i in enumerate(x):
                f[i], sigma[i] = fun(self, x_i)
            f = np.array(f).reshape(shape_eval)
            sigma = np.array(sigma).reshape(shape_eval)
            return f, sigma
        else:
            for i, x_i in enumerate(x):
                f[i] = fun(self, x_i)

        f = np.array(f).reshape(shape_eval)
        return f
    return wrapper_fun


def output_to_sequence(fun):
    """Convert flot output to list."""
    def wrapper_fun(x):
        return [fun(x)]
    return wrapper_fun
