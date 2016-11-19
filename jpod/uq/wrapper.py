# coding: utf8
"""
Wrapper Classes
===============

Defines an interface between the models and OpenTURNS.

:Example:

::

    >> import otwrapy as otw
    >> model = otw.Parallelizer(Wrapper(pod, p_len, output_len), backend='pathos', n_cpus=3)
"""
import openturns as ot
import numpy as np

class Wrapper(ot.OpenTURNSPythonFunction):
    def __init__(self, pod, p_len, output_len, block=False):
        """Initialize the wrapper."""
        super(Wrapper, self).__init__(p_len, output_len)
        self.f_input = None
        self.pod = pod

        if block:
            self._exec = self.int_func
        else:
            self._exec = self.func

    def func(self, coords):
        """Evaluate the POD at a given point.

        This function calls the :func:`predict` function to compute a prediction.
        If the prediction returns a vector, it create `self.f_input` which contains the discretisation information.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        f_eval, _ = self.pod.predictor([coords])
        try:
            f_input, f_eval = np.split(f_eval[0].data, 2)
            if self.f_input is None:
                self.f_input = f_input
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
        f_eval, _ = self.pod.predictor([coords])
        try:
            f_input, f_eval = np.split(f_eval[0].data, 2)
            int_f_eval = np.trapz(f_eval, f_input)
        except:
            f_eval = f_eval[0].data
            int_f_eval = f_eval
        return [int_f_eval.item()]
