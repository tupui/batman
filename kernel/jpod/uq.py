# -*- coding: utf-8 -*-
"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS framework.

Example:
>> analyse = UQ(pod, settings)
>> sobol = analyse.sobol()
>> analyse.error_propagation()

"""

import logging
import numpy as np
import openturns as ot
from openturns.viewer import View
from os import times

# TODO several output files for function.py
# TODO create output folder for UQ

class UQ:
    """UQ class.

    It implements the following methods:
    - func(self, coords)
    - int_func(self, coords)
    - error_pod(self, distribution, s_first, function)
    - sobol(self)
    - error_propagation(self).

    """

    logger = logging.getLogger(__name__)

    def __init__(self, jpod, settings):
        """Init the UQ class.

        From the settings file, get:
        - Method to use for the Sensitivity Analysis (SA),
        - Number of prediction to use for SA,
        - Method to use to predict a new snapshot,
        - The list of input variables,
        - The lengh of the output function.

        Also, it creates the `model` and `int_model` ot.PythonFunction().

        :param pod jpod: The POD,
        :param dict settings: The settings_template file.

        """
        self.logger.info("UQ module")
        try:
            self.test = settings.uq['test']
        except:
            pass
        self.pod = jpod
        self.method_sobol = settings.uq['method']
        self.points_sample = settings.uq['sample']
        self.method_pod = settings.prediction['method']
        self.p_lst = settings.snapshot['io']['parameter_names']
        self.output_len = settings.snapshot['io']['shapes'][0][0][0]
        self.p_len = len(self.p_lst)
        self.f_input = None
        self.model = ot.PythonFunction(self.p_len, self.output_len, self.func)
        self.int_model = ot.PythonFunction(self.p_len, 1, self.int_func)
	self.snapshot = settings.space['size_max']

    def func(self, coords):
        """Evaluate the POD on a given point.

        The function uses the POD and interpolate it using Kriging's method to reconstruct the solution.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        f_eval = self.pod.predict(self.method_pod, [coords])
        try:
            f_input, f_eval = np.split(f_eval[0].data, 2)
            if self.f_input is None:
                self.f_input = f_input
        except:
            f_eval = f_eval[0].data
        return f_eval

    def int_func(self, coords):
        """Evaluate the POD on a given point and return the integral.

        Same as `func` but compute the integral using the trapezoidale rule.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The integral of the function.
        :rtype: float

        """
        f_eval = self.pod.predict(self.method_pod, [coords])
        try:
            f_input, f_eval = np.split(f_eval[0].data, 2)
            int_f_eval = np.trapz(f_eval, f_input)
        except:
            f_eval = f_eval[0].data
            int_f_eval = f_eval
        return [int_f_eval.item()]

    def error_pod(self, distribution, indices, function):
        """Compute the error between the POD and the analytic function.

        For test purpose. From the POD of the function, evaluate the error
        using the analytical evaluation of the function on the sample points.

        r2 = 1 - err_l2/var_model

        Also, it computes the error on the Sobol first and total order indices.

        err_l2 = sum()

        :param ot.NumericalSample sample: input sample.
        :param lst(array) indices: Sobol first order indices computed using the POD.
        :param str function: name of the analytic function.

        """
        # TODO add Functions, be able to do n-D Rosenbrock etc use PythonFunction. 
        if function == 'Ishigami':
            formula = ['sin(X1)+7*sin(X2)*sin(X2)+0.1*((X3)*(X3)*(X3)*(X3))*sin(X1)']
            model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)
            s_first_th = np.array([0.3139, 0.4424, 0.])
	    s_second_th = np.array([[0., 0., 0.2], [0., 0., 0.], [0.2, 0., 0.]]) 
	    s_total_th = np.array([0.558, 0.442, 0.244])
            s_err_l2_second = np.sqrt(np.sum((s_second_th - indices[0]) ** 2))
            s_err_l2_first = np.sqrt(np.sum((s_first_th - indices[1]) ** 2))
            s_err_l2_total = np.sqrt(np.sum((s_total_th - indices[2]) ** 2))
        elif function == 'Rosenbrock':
            formula = ['100*(X2-X1*X1)*(X2-X1*X1) + (X1-1)*(X1-1) + 100*(X3-X2*X2)*(X3-X2*X2) + (X2-1)*(X2-1)']
            model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)
        elif function == 'Channel_Flow':
	    def channel_flow(x):
	        Q = x[0]
		Ks = x[1]
	        L=500.
		I=5e-4
		g=9.8
		dx=100
		longueur=40000
		Long=longueur/dx
		hc=np.power((Q**2)/(g*L*L),1./3.);
		hn=np.power((Q**2)/(I*L*L*Ks*Ks),3./10.);
		hinit=10.
		hh=hinit*np.ones(Long);
		for i in xrange(2,Long):
		    hh[Long-i]=hh[Long-i+1]-dx*I*((1-np.power(hh[Long-i+1]/hn,-10./3.))/(1-np.power(hh[Long-i+1]/hc,-3.)))
		h=hh

		X=np.arange(dx, longueur+1, dx)

		Zref=-X*I
		return Zref+h
	    model_ref = ot.PythonFunction(2, 400, channel_flow)
            s_first_th = np.array([0.1, 0.8])
            s_second_th = np.array([[0., 0.1], [0.1, 0.]])
            s_total_th = np.array([0.1, 0.9])
            s_err_l2_second = np.sqrt(np.sum((s_second_th - indices[0]) ** 2))
            s_err_l2_first = np.sqrt(np.sum((s_first_th - indices[1]) ** 2))
            s_err_l2_total = np.sqrt(np.sum((s_total_th - indices[2]) ** 2))

        else:
            print "No or wrong analytical function, options are: Ishigami (3D), Rosenbrock (3D)"
            return

        err_max = 0.
        err_l2 = 0.
        eval_mean = 0.
        sample = distribution.getSample(self.points_sample)
        for _, j in enumerate(sample):
            eval_ref = model_ref(j)[100]
            eval_pod = self.model(j)[100]
            eval_mean = eval_mean + eval_ref
            err_max = max(err_max, abs(eval_pod - eval_ref))
            err_l2 = err_l2 + (eval_pod - eval_ref) ** 2
        eval_mean = eval_mean / self.points_sample
        eval_var = 0.
        for _, j in enumerate(sample):
            eval_ref = model_ref(j)[100]
            eval_var = eval_var + (eval_mean - eval_ref) ** 2
        err_q2 = 1 - err_l2 / eval_var
	err_q2 = err_l2 / self.points_sample
        print "\n----- POD error -----"
        print("L_inf(error): {}\nQ2(error): {}\nL2(sobol first, second and total order indices error): {}, {}, {}".format(err_max, err_q2, s_err_l2_first, s_err_l2_second, s_err_l2_total))
        # Write error to file pod_err.dat
        with open('./pod_err.dat', 'w') as f:
            f.writelines(str(self.snapshot)+' '+str(err_q2)+' '+str(self.points_sample)+' '+str(s_err_l2_first)+' '+str(s_err_l2_second)+' '+str(s_err_l2_total))

        try:
	    output_ref = model_ref(sample)
            output = self.int_model(sample)
            qq_plot = ot.VisualTest_DrawQQplot(output_ref, output)
            View(qq_plot).show()
        except:
	    print "Cannot draw QQplot with output dimension > 1"

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
            #distribution = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * self.p_len, ot.IndependentCopula(self.p_len))
            distribution = ot.ComposedDistribution([ot.Normal(4035., 400.), ot.Uniform(15., 60.)], ot.IndependentCopula(self.p_len))
            sample1 = distribution.getSample(self.points_sample)
            sample2 = distribution.getSample(self.points_sample)

            sobol = ot.SensitivityAnalysis(sample1, sample2, self.int_model)
            sobol.setBlockSize(int(ot.ResourceMap.Get("parallel-threads")))

            print "\n----- Sobol indices -----"
            s_second = sobol.getSecondOrderIndices()
            s_first = sobol.getFirstOrderIndices()
            s_total = sobol.getTotalOrderIndices()
            print "Second order: ", s_second
            indices[0] = np.array(s_second)

        elif self.method_sobol == 'FAST':
            print "\n----- FAST indices -----"
            # TODO use corners
            distribution = ot.ComposedDistribution([ot.Uniform(-np.pi, np.pi)] * self.p_len)
            fast = ot.FAST(self.int_model, distribution, self.points_sample)
            s_first = fast.getFirstOrderIndices()
            s_total = fast.getTotalOrderIndices()

        else:
            print("The method {} doesn't exist".format(self.method_sobol))

        print "First order: ", s_first
        print "Total: ", s_total

        # TODO ANCOVA
        # ancova = ot.ANCOVA(results, sample)
        # indices = ancova.getIndices()
        # uncorrelated = ancova.getUncorrelatedIndices()
        # correlated = indices - uncorrelated

        # Draw importance factors
        s_plt = ot.NumericalPointWithDescription(s_total)
        s_plt.setDescription(self.p_lst)
        try:
            i_factor = ot.SensitivityAnalysis.DrawImportanceFactors(s_plt)
            i_factor.setTitle("Total order Sensitivity Indices")
            View(i_factor).show()
        except:
            print "Cannot draw importance factors: expected positive values"

        indices[1] = np.array(s_first)
        indices[2] = np.array(s_total)

	# Compute error of the POD with a known function
        try:
            self.error_pod(distribution, indices, self.test)
        except AttributeError:
            print "No analytical function to compare the POD from"

        return indices

    def error_propagation(self):
        """Compute the moments.

        All 4 order moments are computed for every output of the function.
        It also compute the PDF for these outputs as a 2D cartesian plot.

        The file moment.dat contains the moments and the file pdf.dat contains the PDFs.

        """
        print "\n----- Moment evaluation -----"
        # TODO be able to change the distributions and corners
        distribution = ot.ComposedDistribution([ot.Normal(4035., 400.), ot.Uniform(15., 60.)])
        sample = distribution.getSample(self.points_sample)
        output = self.model(sample)
        output = output.sort()
        mean = output.computeMean()
        sd = output.computeStandardDeviationPerComponent()
        sd_min = mean - sd
        sd_max = mean + sd
        # var = output.computeVariance()
        min = output.getMin()
        max = output.getMax()
        kernel = ot.KernelSmoothing()

        # Create the PDFs
        output_pts = np.array(output)
	pdf_pts = [None] * self.output_len
        for i in range(self.output_len):
            try:
	        pdf = kernel.build(output[:, i])
            except:
	        pdf = ot.Normal(output[i,i], 0.001)
            pdf_pts[i] = np.array(pdf.computePDF(output[:, i]))
        # Write moments to file
        with open('./moment.dat', 'w') as f:
            f.writelines('TITLE = \" Moment evaluation \" \n')
            if self.output_len == 1:
                f.writelines('VARIABLES = \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
                w_lst = [min, sd_min, mean, sd_max, max]
            else:
                f.writelines('VARIABLES = \"x\" \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
                w_lst = [self.f_input, min, sd_min, mean, sd_max, max]
            f.writelines('ZONE T = \"Moments \" , I='+str(self.output_len)+', F=BLOCK  \n')
            for w in w_lst:
                for i in range(self.output_len):
                    f.writelines("{:.7E}".format(float(w[i])) + "\t ")
                    if i % 1000:
                        f.writelines('\n')
                f.writelines('\n')

        # Write PDF to file
        with open('./pdf.dat', 'w') as f:
            f.writelines('TITLE = \" Probability Density Functions \" \n')
            if self.output_len == 1:
                f.writelines('VARIABLES =  \"output\" \"PDF\" \n')
                f.writelines('ZONE T = \"PDF \" , I='+str(self.output_len)+', J='+str(self.points_sample)+',  F=BLOCK  \n')
            else:
                f.writelines('VARIABLES =  \"x\" \"output\" \"PDF\" \n')
                f.writelines('ZONE T = \"PDF \" , I='+str(self.output_len)+', J='+str(self.points_sample)+',  F=BLOCK  \n')
                # X
                for j in range(self.points_sample):
                    for i in range(self.output_len):
                        f.writelines("{:.7E}".format(float(self.f_input[i])) + "\t ")
                        if (i % 1000) or (j % 1000):
                            f.writelines('\n')
                f.writelines('\n')
            # Output
            for j in range(self.points_sample):
                for i in range(self.output_len):
                    f.writelines("{:.7E}".format(float(output_pts[j][i])) + "\t ")
                    if (i % 1000) or (j % 1000):
                        f.writelines('\n')
            f.writelines('\n')
            # PDF
            for j in range(self.points_sample):
                for i in range(self.output_len):
                    f.writelines("{:.7E}".format(float(pdf_pts[i][j])) + "\t ")
                    if (i % 1000) or (j % 1000):
                        f.writelines('\n')
            f.writelines('\n')

