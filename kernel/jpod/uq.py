"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS framework.

"""

import logging
import numpy as np
import openturns as ot
from openturns.viewer import View


class UQ(object):
    logger = logging.getLogger(__name__)
    def __init__(self, jpod, settings):
        self.logger.info("UQ module")
        try:
            self.test = settings.uq['test']
        except:
            pass
        self.method_sobol = settings.uq['method']
        self.points_sample = settings.uq['points']
        self.method_pod = settings.prediction['method']
        self.pod = jpod
        p_lst = settings.snapshot['io']['parameter_names']
        self.p = len(p_lst)
        self.input = None
        self.output = settings.snapshot['io']['shapes'][0][0][0]
        self.model = ot.PythonFunction(self.p, self.output, self.func)
        self.int_model = ot.PythonFunction(self.p, 1, self.int_func)

    def func(self, coords):
        """Evaluate the pod on a given point.

        The function uses the pod and interpolate it using Kriging's method to reconstruct the solution.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        f_eval = self.pod.predict(self.method_pod, [coords])
        try:
            input, f_eval = np.split(f_eval[0].data, 2)
            if self.input is None:
            	self.input = input
        except:
            f_eval = f_eval[0].data
        return f_eval

    def int_func(self, coords):
        """Evaluate the pod on a given point and return the integral.

        The function uses the pod and interpolate it using Kriging's method to reconstruct the solution.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The integral of the function.
        :rtype: float

        """
        f_eval = self.pod.predict(self.method_pod, [coords])
        try:
            input, f_eval = np.split(f_eval[0].data, 2)
            int_f_eval = np.trapz(f_eval, input)
        except:
            f_eval = f_eval[0].data
            int_f_eval = f_eval
        return [int_f_eval.item()]

    def error_pod(self, distribution, s_first, function):
        """Compute the error between the POD and the analytic function.

        For test purpose. From the POD of the function, evaluate the error
        using the analytical evaluation of the function on the sample points.

        Also, it computes the error on the Sobol first order indices.

        :param ot.NumericalSample sample: input sample.
        :param ot. s_first: Sobol first order indices computed using the POD.
        :param str function: name of the analytic function.

        """
        if function == 'Ishigami':
            formula = ['sin(X1)+7*sin(X2)*sin(X2)+0.1*((X3)*(X3)*(X3)*(X3))*sin(X1)']
            model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)
            s_first_th = np.array([0.3139, 0.4424, 0.])
            s_err_l2 = np.sqrt(np.sum((s_first_th - s_first) ** 2))
        else:
            print "No or wrong analytical function, options are: Ishigami"
            return

        err_max = 0.
        err_l2 = 0.
        eval_mean = 0.
        sample = distribution.getSample(self.points_sample)
        for _, j in enumerate(sample):
            eval_ref = model_ref(j)[0]
            eval_pod = self.int_model(j)[0]
            eval_mean = eval_mean + eval_ref
            err_max = max(err_max, 100 * abs(eval_pod - eval_ref) / abs(eval_ref))
            err_l2 = err_l2 + (eval_pod - eval_ref) ** 2
        eval_mean = eval_mean / self.points_sample
        eval_var = 0.
        for _, j in enumerate(sample):
            eval_ref = model_ref(j)[0]
            eval_var = eval_var + (eval_mean - eval_ref) ** 2
        err_q2 = 1 - err_l2 / eval_var
        print "\n----- POD error -----"
        print("L_max(error %): {}\nQ2(error): {}\nL2(sobol first order indices error): {}".format(err_max, err_q2, s_err_l2))

        output_ref = model_ref(sample)
        output = self.int_model(sample)
        qq_plot = ot.VisualTest_DrawQQplot(output_ref, output)
        View(qq_plot).show()

        return model_ref

    def sobol(self):
        """Compute the sobol indices.

        It returns the seond, first and total order indices of Sobol.
        The second order indices are only available with the sobol method.

        :return: The Sobol indices
        :rtype: lst

        """
        indices = [[], [], []]
        if self.method_sobol == 'sobol':
            # TODO use corners
            distribution = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * self.p, ot.IndependentCopula(self.p))
            sample1 = distribution.getSample(self.points_sample)
            sample2 = distribution.getSample(self.points_sample)

            sobol = ot.SensitivityAnalysis(sample1, sample2, self.int_model)
            
            print "\n----- Sobol indices -----\n"
            s_second = sobol.getSecondOrderIndices()
            s_first = np.array(sobol.getFirstOrderIndices())
            s_total = sobol.getTotalOrderIndices()
            print "Second order: ", s_second
            indices[0] = np.array(s_second)

        elif self.method_sobol == 'FAST':
            print "\n----- FAST indices -----\n"
            # TODO use corners
            distribution = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * self.p)
            fast = ot.FAST(self.int_model, distribution, self.points_sample)
            s_first = fast.getFirstOrderIndices()
            s_total = fast.getTotalOrderIndices()

        else:
            print("The method {} doesn't exist".format(self.method_sobol))

        print "First order: ", s_first
        print "Total: ", s_total

        try:
            self.error_pod(distribution, s_first, self.test)
        except AttributeError:
            print "No analytical function to compare the POD from"

        indices[1] = np.array(s_first)
        indices[2] = np.array(s_total)
        return indices

    def error_propagation(self):
        """Compute the mean and standard deviation.

        """
        # TODO be able to change the distributions and corners
        distribution = ot.ComposedDistribution([ot.Normal(0.3, 0.35), ot.Normal(0.1, 0.01), ot.Normal(-0.8, 0.01)])
        sample = distribution.getSample(self.points_sample)
        output = self.model(sample)
        mean = output.computeMean()
        sd = output.computeStandardDeviationPerComponent()
        sd_min = mean - sd
        sd_max = mean + sd
        var = output.computeVariance()
        min = output.getMin()
        max = output.getMax()
        print "\n----- Moment evaluation -----"
#       print "Mean value: ", mean
#       print "Standard deviation: ", sd
#       print "Variance: ", var
#       print "Max: ", max
#       print "Min: ", min
        
        nb_value = np.size(self.input)

        with open('./moment.dat', 'w') as f:
            f.writelines('TITLE = \" Moment evaluation \" \n')
            f.writelines('VARIABLES = \"x\" \"Min\" \"SD_min\"  \"Mean\" \"SD_max\"  \"Max\" \n')
            f.writelines('ZONE T = \"zone1 \" , I='+str(self.output)+', F=BLOCK  \n') 

            for w in [self.input, min, sd_min, mean, sd_max, max]:
               for i in range(nb_value):
                   f.writelines("{:.7E}".format(float(w[i])) + "\t ")
                   if i % 1000:
                       f.writelines('\n')
            f.writelines('\n')














