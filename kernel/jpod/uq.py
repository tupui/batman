# -*- coding: utf-8 -*-
"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS framework.

Example:
>> analyse = UQ(pod, settings, output)
>> analyse.sobol()
>> analyse.error_propagation()

"""

import logging
import numpy as np
import openturns as ot
# from openturns.viewer import View
from os import mkdir
import itertools


class UQ:
    """UQ class.

    It implements the following methods:
    - func(self, coords)
    - int_func(self, coords)
    - error_pod(self, indices, function)
    - sobol(self)
    - error_propagation(self).

    """

    logger = logging.getLogger(__name__)

    def __init__(self, jpod, settings, output):
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
        self.output_folder = output
        try:
            mkdir(output)
        except:
            pass
        self.pod = jpod
        self.p_lst = settings.snapshot['io']['parameter_names']
        self.p_len = len(self.p_lst)
        try:
            self.method_sobol = settings.uq['method']
            self.points_sample = settings.uq['sample']
            pdf = settings.uq['pdf']
        except KeyError:
            self.logger.exception("Need to configure method, sample and PDF")
            raise SystemExit
        input_pdf = "ot." + pdf[0]
        for i in xrange(self.p_len - 1):
            input_pdf = input_pdf + ", ot." + pdf[i + 1]
        self.distribution = eval("ot.ComposedDistribution([" + input_pdf + "], ot.IndependentCopula(self.p_len))")
        self.method_pod = settings.prediction['method']
        self.output_len = settings.snapshot['io']['shapes'][0][0][0]
        self.f_input = None
        self.model = ot.PythonFunction(self.p_len, self.output_len, self.func)
        self.int_model = ot.PythonFunction(self.p_len, 1, self.int_func)
        self.snapshot = settings.space['size_max']

    def __repr__(self):
        """Information about object."""
        return "UQ object: Method({}), Input({}), Distribution({})".format(self.method_sobol, self.p_lst, self.distribution)

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

    def error_pod(self, indices, function):
        """Compute the error between the POD and the analytic function.

        For test purpose. From the POD of the function, evaluate the error
        using the analytical evaluation of the function on the sample points.

        Q2 = 1 - err_l2/var_model

        Also, it computes the error on the Sobol first and total order indices.

        err_l2 = sum()

        :param lst(array) indices: Sobol first order indices computed using the POD.
        :param str function: name of the analytic function.

        """
        if function == 'Ishigami':
            formula = ['sin(X1)+7*sin(X2)*sin(X2)+0.1*((X3)*(X3)*(X3)*(X3))*sin(X1)']
            model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)
            s_first_th = np.array([0.3139, 0.4424, 0.])
            s_second_th = np.array([[0., 0., 0.2], [0., 0., 0.], [0.2, 0., 0.]])
            s_total_th = np.array([0.558, 0.442, 0.244])
        elif function == 'Rosenbrock':
            formula = ['100*(X2-X1*X1)*(X2-X1*X1) + (X1-1)*(X1-1) + 100*(X3-X2*X2)*(X3-X2*X2) + (X2-1)*(X2-1)']
            model_ref = ot.NumericalMathFunction(['X1', 'X2', 'X3'], ['Y'], formula)
            s_first_th = np.array([0.229983, 0.4855, 0.130659])
            s_second_th = np.array([[0., 0.0920076, 0.00228908], [0.0920076, 0., 0.0935536], [0.00228908, 0.0935536, 0.]])
            s_total_th = np.array([0.324003, 0.64479, 0.205122])
        elif function == 'Channel_Flow':
            def channel_flow(x):
                Q = x[0]
                Ks = x[1]
                L = 500.
                I = 5e-4
                g = 9.8
                dx = 100
                longueur = 40000
                Long = longueur / dx
                hc = np.power((Q**2) / (g * L * L), 1. / 3.)
                hn = np.power((Q**2) / (I * L * L * Ks * Ks), 3. / 10.)
                hinit = 10.
                hh = hinit * np.ones(Long)
                for i in xrange(2, Long + 1):
                    hh[Long - i] = hh[Long - i + 1] - dx * I * ((1 - np.power(hh[Long - i + 1] / hn, -10. / 3.)) / (1 - np.power(hh[Long - i + 1] / hc, -3.)))
                h = hh

                X = np.arange(dx, longueur + 1, dx)

                Zref = - X * I
                return Zref + h
            model_ref = ot.PythonFunction(2, 400, channel_flow)
            s_first_th = np.array([0.1, 0.8])
            s_second_th = np.array([[0., 0.1], [0.1, 0.]])
            s_total_th = np.array([0.1, 0.9])
        else:
            self.logger.error("Wrong analytical function, options are: Ishigami, Rosenbrock and Channel_Flow")
            return
        try:
            s_err_l2_second = np.sqrt(np.sum((s_second_th - indices[0]) ** 2))
        except:
            self.logger.warn("No Second order indices with FAST")

        s_err_l2_second = 0.
        s_err_l2_first = np.sqrt(np.sum((s_first_th - indices[1]) ** 2))
        s_err_l2_total = np.sqrt(np.sum((s_total_th - indices[2]) ** 2))

        err_max = 0.
        err_l2 = 0.
        eval_mean = 0.
        sample = self.distribution.getSample(self.points_sample)

        for _, j in enumerate(sample):
            eval_ref = model_ref(j)[0]
            eval_pod = self.model(j)[0]
            eval_mean = eval_mean + eval_ref
            err_max = max(err_max, abs(eval_pod - eval_ref))
            err_l2 = err_l2 + (eval_pod - eval_ref) ** 2
        eval_mean = eval_mean / self.points_sample
        eval_var = 0.
        for _, j in enumerate(sample):
            # eval_ref = model_ref(j)[0]
            eval_ref = self.model(j)[0]
            eval_var = eval_var + (eval_mean - eval_ref) ** 2
        err_q2 = 1 - err_l2 / eval_var

        self.logger.info("\n----- POD error -----")
        self.logger.info("L_inf(error): {}\nQ2(error): {}\nL2(sobol first, second and total order indices error): {}, {}, {}".format(err_max, err_q2, s_err_l2_first, s_err_l2_second, s_err_l2_total))
        # Write error to file pod_err.dat
        with open(self.output_folder + '/pod_err.dat', 'w') as f:
            f.writelines(str(self.snapshot) + ' ' + str(err_q2) + ' ' + str(self.points_sample) + ' ' + str(s_err_l2_first) + ' ' + str(s_err_l2_second) + ' ' + str(s_err_l2_total))

        if self.output_len == 1:
            output_ref = model_ref(sample)
            output = self.int_model(sample)
            qq_plot = ot.VisualTest_DrawQQplot(output_ref, output)
            # View(qq_plot).show()
            qq_plot.draw(self.output_folder + '/qq_plot.png')
        else:
            self.logger.debug("Cannot draw QQplot with output dimension > 1")

    def sobol(self):
        """Compute the Sobol' indices.

        It returns the second, first and total order indices of Sobol'.
        The second order indices are only available with the sobol method.

        :return: The Sobol' indices
        :rtype: lst

        """
        indices = [[], [], []]
        if self.method_sobol == 'sobol':
            self.logger.info("\n----- Sobol' indices -----")
            sample1 = self.distribution.getSample(self.points_sample)
            sample2 = self.distribution.getSample(self.points_sample)
            sobol = ot.SensitivityAnalysis(sample1, sample2, self.model)
            sobol.setBlockSize(int(ot.ResourceMap.Get("parallel-threads")))
            for i in range(self.output_len):
                indices[0].append(np.array(sobol.getSecondOrderIndices(i)))
            self.logger.debug("Second order: {}".format(indices[0]))
        elif self.method_sobol == 'FAST':
            self.logger.info("\n----- FAST indices -----")
            sobol = ot.FAST(self.model, self.distribution, self.points_sample)
        else:
            self.logger.error("The method {} doesn't exist".format(self.method_sobol))
            return

        for i in range(self.output_len):
            indices[1].append(np.array(sobol.getFirstOrderIndices(i)))
            indices[2].append(np.array(sobol.getTotalOrderIndices(i)))

            self.logger.debug("First order: {}".format(indices[1]))
            self.logger.debug("Total: {}".format(indices[2]))

            # TODO ANCOVA
            # ancova = ot.ANCOVA(results, sample)
            # indices = ancova.getIndices()
            # uncorrelated = ancova.getUncorrelatedIndices()
            # correlated = indices - uncorrelated

        # Write Sobol' indices to file
        # with open(self.output_folder + '/sensitivity.dat', 'w') as f:
        #    f.writelines('TITLE = \" Sobol indices \" \n')
        #    if self.output_len == 1:
        #        f.writelines('VARIABLES = \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
        #        w_lst = [min, sd_min, mean, sd_max, max]
        #    else:
        #        f.writelines('VARIABLES = \"x\" \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
        #        w_lst = [self.f_input, min, sd_min, mean, sd_max, max]
        #    f.writelines('ZONE T = \"Moments \" , I='+str(self.output_len)+', F=BLOCK  \n')
        #    for w in w_lst:
        #        for i in range(self.output_len):
        #            f.writelines("{:.7E}".format(float(w[i])) + "\t ")
        #            if i % 1000:
        #                f.writelines('\n')
        #        f.writelines('\n')

        # Compute error of the POD with a known function
        try:
            self.error_pod(indices, self.test)
        except AttributeError:
            self.logger.info("No analytical function to compare the POD from")

    def error_propagation(self):
        """Compute the moments.

        All 4 order moments are computed for every output of the function.
        It also compute the PDF for these outputs as a 2D cartesian plot.

        The file moment.dat contains the moments and the file pdf.dat contains the PDFs.

        """
        self.logger.info("\n----- Moment evaluation -----")
        sample = self.distribution.getSample(self.points_sample)
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
        pdf_pts = [None] * self.output_len
        d_PDF = 100
        sample = self.distribution.getSample(d_PDF)
        output_extract = self.model(sample)
        for i in range(self.output_len):
            try:
                pdf = kernel.build(output[:, i])
            except:
                pdf = ot.Normal(output[i, i], 0.001)
            pdf_pts[i] = np.array(pdf.computePDF(output_extract[:, i]))
        # Write moments to file
        with open(self.output_folder + '/moment.dat', 'w') as f:
            f.writelines('TITLE = \" Moment evaluation \" \n')
            if self.output_len == 1:
                f.writelines('VARIABLES = \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
                w_lst = [min, sd_min, mean, sd_max, max]
            else:
                f.writelines('VARIABLES = \"x\" \"Min\" \"SD_min\" \"Mean\" \"SD_max\" \"Max\" \n')
                w_lst = [self.f_input, min, sd_min, mean, sd_max, max]
            f.writelines('ZONE T = \"Moments \" , I=' + str(self.output_len) + ', F=BLOCK  \n')
            for w, i in itertools.product(w_lst, range(self.output_len)):
                f.writelines("{:.7E}".format(float(w[i])) + "\t ")
                if i % 1000:
                    f.writelines('\n')
            f.writelines('\n')

        # Write PDF to file
        with open(self.output_folder + '/pdf.dat', 'w') as f:
            f.writelines('TITLE = \" Probability Density Functions \" \n')
            if self.output_len == 1:
                f.writelines('VARIABLES =  \"output\" \"PDF\" \n')
                f.writelines('ZONE T = \"PDF \" , I=' + str(self.output_len) + ', J=' + str(d_PDF) + ',  F=BLOCK  \n')
            else:
                f.writelines('VARIABLES =  \"x\" \"output\" \"PDF\" \n')
                f.writelines('ZONE T = \"PDF \" , I=' + str(self.output_len) + ', J=' + str(d_PDF) + ',  F=BLOCK  \n')
                # X
                for j, i in itertools.product(range(d_PDF), range(self.output_len)):
                    f.writelines("{:.7E}".format(float(self.f_input[i])) + "\t ")
                    if (i % 1000) or (j % 1000):
                        f.writelines('\n')
                f.writelines('\n')
            # Output
            for j, i in itertools.product(range(d_PDF), range(self.output_len)):
                f.writelines("{:.7E}".format(float(output_extract[j][i])) + "\t ")
                if (i % 1000) or (j % 1000):
                    f.writelines('\n')
            f.writelines('\n')
            # PDF
            for j, i in itertools.product(range(d_PDF), range(self.output_len)):
                f.writelines("{:.7E}".format(float(pdf_pts[i][j])) + "\t ")
                if (i % 1000) or (j % 1000):
                    f.writelines('\n')
            f.writelines('\n')
