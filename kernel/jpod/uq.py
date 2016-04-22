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
        f_eval = self.pod.predict(self.method_pod, [coords])
        return f_eval[0].data

    def sobol(self):
        """Compute the sobol indices.

        """
        model = ot.PythonFunction(3, 1, self.func)
        formula = ['sin(X1)+7*sin(X2)*sin(X2)+0.1*((X3)*(X3)*(X3)*(X3))*sin(X1)']
        model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)

        # TODO use corners
        distribution = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * 3, ot.IndependentCopula(3))
        sample1 = distribution.getSample(self.points_sample)
        sample2 = distribution.getSample(self.points_sample)
        
	sobol = ot.SensitivityAnalysis(sample1, sample2, model)
       
        # TODO create err function 
        err_max = 0.
        err_l2 = 0.
        eval_mean = 0.
        for _, j in enumerate(sample1):
	    eval_ref = model_ref(j)[0]
            eval_pod = model(j)[0]
            eval_mean = eval_mean + eval_ref
            err_max = max(err_max, 100 * abs(eval_pod - eval_ref) / abs(eval_ref))
            err_l2 = err_l2 + (eval_pod - eval_ref) ** 2

        eval_mean = eval_mean / self.points_sample
        eval_var = 0.
        for _, j in enumerate(sample1):
            eval_ref = model_ref(j)[0]
            eval_var = eval_var + (eval_mean - eval_ref) ** 2
        err_q2 = 1 - err_l2 / eval_var
        print "\n----- POD error -----"
        print("L_max(error %): {}\nQ2(error): {}".format(err_max, err_q2))

        print "\n----- Sobol indices -----\n"
        s_second = sobol.getSecondOrderIndices()
        s_first = np.array(sobol.getFirstOrderIndices())
        s_total = sobol.getTotalOrderIndices()
        s_first_th = np.array([0.3139, 0.4424, 0.])
        s_err_l2 = np.sqrt(np.sum((s_first_th - s_first) ** 2))
        print "Second order: ", s_second
        print("First order: {} -> L2(error): {}".format(s_first, s_err_l2))
        print "Total: ", s_total

        # TODO create option for sobol or FAST
        print "\n----- FAST indices -----\n"
        fast = ot.FAST(model, ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * 3), 2000)
        s_first = fast.getFirstOrderIndices()
        s_err_l2 = np.sqrt(np.sum((s_first_th - s_first) ** 2))
        print("First order: {} -> L2(error): {}".format(s_first, s_err_l2))
        print "Total: ", fast.getTotalOrderIndices()

        # TODO create function for moment evaluation
        distribution = ot.ComposedDistribution([ot.Normal(0.3, 0.01), ot.Normal(0.1, 0.01), ot.Normal(-0.8, 0.01)])
        sample = distribution.getSample(self.points_sample)
        output = model(sample)
        print "\n----- Moment evaluation -----"
        print "Ref mean value: ", model_ref([0.3, 0.1, -0.8])
        print "Mean value: ", output.computeMean()
        print "Standard deviation: ", output.computeStandardDeviation()

