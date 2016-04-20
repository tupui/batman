"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS framework.

"""

import numpy as np
import openturns as ot

class UQ(object):
    def __init__(self, jpod, settings):
        self.method_sample = settings.uq['method']
        self.points_sample = settings.uq['points']
        self.method_pod = settings.prediction['method']
        self.pod = jpod
        p_lst = settings.snapshot['io']['parameter_names']
        self.p = len(p_lst)

    def func(self, coords):
        """Evaluate the pod on a given point. 

        The function uses the pod and interpolate it using Kriging's method to reconstruct the solution.        

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        point = [(float(coords[0]),float(coords[1]), float(coords[2]))]
        f_eval = self.pod.predict(self.method_pod, point)
        return f_eval[0].data

    def sobol(self):
        """Compute the sobol indices.

        """
        model = ot.PythonFunction(3, 1, self.func)
        formula = ['sin(X1)+7*sin(X2)*sin(X2)+0.05*((X3)*(X3)*(X3)*(X3))*sin(X1)']
        model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)

        distribution = ot.ComposedDistribution([ot.Uniform(-1, 1)] * 3, ot.IndependentCopula(3))
        sample1 = distribution.getSample(self.points_sample)
        sample2 = distribution.getSample(self.points_sample)
        
	sobol = ot.SensitivityAnalysis(sample1, sample2, model)
        
        err_max = 0.
        err_l2 = 0.
        err_l2_pod = 0.
        for _, j in enumerate(sample1):
	    eval_ref = model_ref(j)[0]
            eval_pod = model(j)[0]
            err_max = max(err_max, 100 * abs(eval_ref - eval_pod) / abs(eval_ref))
            err_l2 = err_l2 + (eval_pod - eval_ref) ** 2
            err_l2_pod = err_l2 + eval_pod ** 2

        print("L_max(error): {}\nL2(error): {}".format(err_max, np.sqrt(err_l2 / err_l2_pod)))

        print "\nSecond order: ", sobol.getSecondOrderIndices()
        print "\nFirst order: ", sobol.getFirstOrderIndices()
        print "\nTotal: ", sobol.getTotalOrderIndices()


        #return self.func(float(1.235), float(np.pi), float(2.3627727))
