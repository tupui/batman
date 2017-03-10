# coding: utf8
"""
Wrapper Classes
===============

Defines an interface between the models and OpenTURNS.
This allows to surcharge the function with ``_exec_sample``. This enable
multiprocessing capabilities.

:Example:

::

    >> import otwrapy as otw
    >> surrogate = driver.surrogate
    >> p_len = 3
    >> output_len = 1
    >> model = otw.Parallelizer(Wrapper(surrogate, p_len, output_len), backend='pathos', n_cpus=3)
"""
import openturns as ot
import numpy as np


class Wrapper(ot.OpenTURNSPythonFunction):

    """Wrap predictor with OpenTURNS."""

    def __init__(self, surrogate, p_len, output_len, block=False):
        """Initialize the wrapper.

        :param :class:`surrogate.surrogate_model.SurrogateModel` surrogate: a surrogate
        :param int p_len: input dimension
        :param int output_len: output dimension
        :param bool block: If true, return the integral
        """
        super(Wrapper, self).__init__(p_len, output_len)
        self.surrogate = surrogate

        if block:
            self._exec = self.int_func
        else:
            self._exec = self.func

    def func(self, coords):
        """Evaluate the surrogate at a given point.

        This function calls the surrogate to compute a prediction.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        f_eval, _ = self.surrogate(coords)
        try:
            _, f_eval = np.split(f_eval[0].data, 2)
        except:
            f_eval = f_eval[0].data
        return f_eval

    def int_func(self, coords):
        """Evaluate the POD at a given point and return the integral.

        Same as :func:`func` but compute the integral using the trapezoidale rule.
        It simply returns the prediction in case of a scalar output.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The integral of the function.
        :rtype: float

        """
        f_eval, _ = self.surrogate(coords)
        try:
            f_input, f_eval = np.split(f_eval[0].data, 2)
            int_f_eval = np.trapz(f_eval, f_input)
        except:
            f_eval = f_eval[0].data
            int_f_eval = f_eval
        return [int_f_eval.item()]
