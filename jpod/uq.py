# -*- coding: utf-8 -*-
"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS framework.

.. seealso:: The documentation of the used class :class:`openturns.SensitivityAnalysis`

It is called using option `-u`. Configuration is done from *settings*:

1. Computes Sobol' indices (map and block or aggragated),
2. Compare the result with a know function if available,
3. Propagate uncertainties from input distributions.

At each step, an output file is beeing written within the UQ folder.

It implements the following methods:

- :func:`UQ.func`
- :func:`UQ.int_func`
- :func:`UQ.error_pod`
- :func:`UQ.sobol`
- :func:`UQ.error_propagation`

Usage
-----

The *settings* file must contains the following parameters::

    'UQ' : {
        'method' : sobol, (or FAST)
        'type' : aggregated, (or block)
        'sample' : 20000,
        'pdf' : [Normal(sigma, mu), Uniform(inf, sup)] (OpenTURNS factories)
    }

:Example:
::

    >> analyse = UQ(pod, settings, output)
    >> analyse.sobol()
    >> analyse.error_propagation()

References
----------

A. Marrel, N. Saint-Geours. M. De Lozzo: Sensitivity Analysis of Spatial and/or Temporal Phenomena. Handbook of Uncertainty Quantification. 2015.   DOI:10.1007/978-3-319-11259-6_39-1

B. Iooss: Revue sur l’analyse de sensibilité globale de modèles numériques. Journal de la Société Française de Statistique. 2010

M. Baudin, A. Dutfoy, B. Iooss, A. Popelin: OpenTURNS: An industrial software for uncertainty quantification in simulation. 2015. ArXiv ID: 1501.05242


"""

# Authors: Pamphile ROY <roy.pamphile@gmail.fr>
# Copyright: CERFACS

import logging
import numpy as np
import openturns as ot
# from openturns.viewer import View
from os import mkdir
import itertools


class UQ:

    """Uncertainty Quantification class."""

    logger = logging.getLogger(__name__)

    def __init__(self, jpod, settings, output=None):
        """Init the UQ class.

        From the settings file, it gets:

        - Method to use for the Sensitivity Analysis (SA),
        - Type of Sobol' indices to compute,
        - Number of prediction to use for SA,
        - Method to use to predict a new snapshot,
        - The list of input variables,
        - The lengh of the output function.

        Also, it creates the `model` and `int_model` as `ot.PythonFunction()`.

        :param pod jpod: The POD,
        :param dict settings: The settings_template file.

        """
        self.logger.info("UQ module")
        try:
            self.test = settings.uq['test']
        except:
            self.test = None
        self.output_folder = output
        try:
            mkdir(output)
        except:
            self.logger.debug("Output folder already exists.")
        self.pod = jpod
        self.p_lst = settings.snapshot['io']['parameter_names']
        self.p_len = len(self.p_lst)
        try:
            self.method_sobol = settings.uq['method']
            self.type_indices = settings.uq['type']
            self.points_sample = settings.uq['sample']
            pdf = settings.uq['pdf']
        except KeyError:
            self.logger.exception("Need to configure method, type, sample and PDF")
            raise SystemExit

        input_pdf = "ot." + pdf[0]
        for i in range(self.p_len - 1):
            input_pdf = input_pdf + ", ot." + pdf[i + 1]
        self.distribution = eval("ot.ComposedDistribution([" + input_pdf + "], ot.IndependentCopula(self.p_len))")
        # self.experiment = ot.MonteCarloExperiment(self.distribution, self.points_sample)
        self.experiment = ot.LHSExperiment(self.distribution, self.points_sample)
        self.sample = self.experiment.generate()
        self.logger.info("Created {} samples with an LHS experiment".format(self.points_sample))

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

    def error_pod(self, indices, function):
        r"""Compute the error between the POD and the analytic function.

        .. warning:: For test purpose only. Choises are Ishigami, Rosenbrock and Channel Flow test functions.

        From the surrogate of the function, evaluate the error
        using the analytical evaluation of the function on the sample points.

        .. math:: Q^2 = 1 - \frac{err_{l2}}{var_{model}}

        Knowing that :math:`err_{l2} = \sum \frac{(prediction - reference)^2}{n}`, :math:`var_{model} = \sum \frac{(prediction - mean)^2}{n}`

        Also, it computes the mean square error on the Sobol first and total order indices.

        A summary is written within `pod_err.dat`.

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
                for i in range(2, Long + 1):
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

        eval_mean = np.zeros(self.output_len)
        eval_ref = []
        err_max = 0.
        err_l2 = 0.

        for i, j in enumerate(self.sample):
            eval_ref.append(np.array(model_ref(j)))
            eval_pod = np.array(self.model(j))
            eval_mean = eval_mean + eval_ref[i]
            err_max = max(err_max, max(abs(eval_pod - eval_ref[i])))
            err_l2 = err_l2 + np.sum((eval_pod - eval_ref[i]) ** 2)
        eval_mean = eval_mean / self.points_sample
        eval_var = 0.
        for i, _ in enumerate(self.sample):
            eval_var = eval_var + np.sum((eval_mean - eval_ref[i]) ** 2)
        err_q2 = 1 - err_l2 / eval_var

        self.logger.info("\n----- POD error -----")
        self.logger.info("\nL_inf(error): {}\nQ2(error): {}\nL2(sobol first, second and total order indices error): {}, {}, {}".format(err_max, err_q2, s_err_l2_first, s_err_l2_second, s_err_l2_total))

        # Write error to file pod_err.dat
        try:
            with open(self.output_folder + '/pod_err.dat', 'w') as f:
                f.writelines(str(self.snapshot) + ' ' + str(err_q2) + ' ' + str(self.points_sample) + ' ' + str(s_err_l2_first) + ' ' + str(s_err_l2_second) + ' ' + str(s_err_l2_total))

            # Write a QQplot in case of a scalar output
            if self.output_len == 1:
                output_ref = model_ref(self.sample)
                output = self.int_model(self.sample)
                qq_plot = ot.VisualTest_DrawQQplot(output_ref, output)
                # View(qq_plot).show()
                qq_plot.draw(self.output_folder + '/qq_plot.png')
            else:
                self.logger.debug("Cannot draw QQplot with output dimension > 1")
        except:
            self.logger.debug("No output folder to write errors in")

    def sobol(self):
        """Compute Sobol' indices.

        It returns the second, first and total order indices of Sobol'.
        Two methods are possible for the indices:

        - `sobol`
        - `FAST`

        .. warning:: The second order indices are only available with the sobol method.

        And three types of computation are availlable for the global indices:

        - `block`
        - `map`
        - `aggregated`

        If *aggregated*, *map* indices are computed. In case of a scalar value, all types returns the same values.
        *map* or *block* indices are written within `sensitivity.dat` and aggregated indices within `sensitivity_aggregated.dat`.

        Finally, it calls :func:`error_pod` in order to compare the indices with their analytical values.

        :return: Sobol' indices
        :rtype: lst(np.array)

        """
        indices = [[], [], []]

        if self.type_indices == 'block':
            sobol_model = self.int_model
            sobol_len = 1
        else:
            sobol_model = self.model
            sobol_len = self.output_len

        if self.method_sobol == 'sobol':
            self.logger.info("\n----- Sobol' indices -----")
            sample1 = self.sample
            experiment = ot.LHSExperiment(self.distribution, self.points_sample)
            sample2 = experiment.generate()
            sobol = ot.SensitivityAnalysis(sample1, sample2, sobol_model)
            sobol.setBlockSize(int(ot.ResourceMap.Get("parallel-threads")))
            for i in range(sobol_len):
                indices[0].append(np.array(sobol.getSecondOrderIndices(i)))
            self.logger.debug("Second order: {}".format(indices[0]))
        elif self.method_sobol == 'FAST':
            self.logger.info("\n----- FAST indices -----")
            sobol = ot.FAST(sobol_model, self.distribution, self.points_sample)
        else:
            self.logger.error("The method {} doesn't exist".format(self.method_sobol))
            return

        for i in range(sobol_len):
            indices[1].append(np.array(sobol.getFirstOrderIndices(i)))
            indices[2].append(np.array(sobol.getTotalOrderIndices(i)))

        self.logger.debug("First order: {}".format(indices[1]))
        self.logger.debug("Total: {}".format(indices[2]))

        # TODO ANCOVA
        # ancova = ot.ANCOVA(results, sample)
        # indices = ancova.getIndices()
        # uncorrelated = ancova.getUncorrelatedIndices()
        # correlated = indices - uncorrelated

        # Write Sobol' indices to file: block or map
        try:
            with open(self.output_folder + '/sensitivity.dat', 'w') as f:
                f.writelines('TITLE = \" Sobol indices \" \n')
                var = ''
                for p in self.p_lst:
                    var += ' \"S_' + str(p) + '\" \"S_T_' + str(p) + '\"'
                var += '\n'
                if (self.output_len == 1) or (self.type_indices == 'block'):
                    variables = 'VARIABLES =' + var
                    f.writelines(variables)
                    f.writelines('ZONE T = \"Sensitivity \" , I=1, F=BLOCK  \n')
                else:
                    variables = 'VARIABLES = \"x\"' + var
                    f.writelines(variables)
                    f.writelines('ZONE T = \"Sensitivity \" , I=' + str(self.output_len) + ', F=BLOCK  \n')
                    # X
                    for i in range(self.output_len):
                        f.writelines("{:.7E}".format(float(self.f_input[i])) + "\t ")
                        if i % 1000:
                            f.writelines('\n')
                    f.writelines('\n')
                # Indices
                w_lst = [indices[1], indices[2]]
                for j, w, i in itertools.product(range(self.p_len), w_lst, range(sobol_len)):
                    f.writelines("{:.7E}".format(float(w[i][j])) + "\t ")
                    if i % 1000:
                        f.writelines('\n')
                f.writelines('\n')
        except:
            self.logger.debug("No output folder to write indices in")

        # Aggregated Indices
        if self.type_indices == 'aggregated':
            self.logger.info("\n----- Aggregated Sensitivity Indices -----")
            output = self.model(self.sample)
            output_var = output.computeVariance()
            sum_var_indices = [np.zeros((self.p_len, self.p_len)), np.zeros((self.p_len)), np.zeros((self.p_len))]
            for i, j in itertools.product(range(self.output_len), range(3)):
                indices[:][j][i] = np.nan_to_num(indices[:][j][i])
                sum_var_indices[j] += float(output_var[i]) * indices[:][j][i]
            sum_var = np.sum(output_var)
            for i in range(3):
                indices[i] = sum_var_indices[i] / sum_var
            self.logger.info("Aggregated_indices: {}".format(indices))

            try:
                with open(self.output_folder + '/sensitivity_aggregated.dat', 'w') as f:
                    f.writelines('TITLE = \" Sobol indices \" \n')
                    variables = 'VARIABLES =' + var
                    f.writelines(variables)
                    f.writelines('ZONE T = \"Sensitivity \" , I=1, F=BLOCK  \n')
                    w_lst = [indices[1], indices[2]]
                    for j, w in itertools.product(range(self.p_len), w_lst):
                        f.writelines("{:.7E}".format(float(w[j])) + "\t ")
                        if i % 1000:
                            f.writelines('\n')
                    f.writelines('\n')
            except:
                self.logger.debug("No output folder to write aggregated indices in")

        # Compute error of the POD with a known function
        if (self.type_indices in ['aggregated', 'block']) and (self.test is not None):
            self.error_pod(indices, self.test)

        return indices

    def error_propagation(self):
        """Compute the moments.

        All 4 order moments are computed for every output of the function.
        It also compute the PDF for these outputs as a 2D cartesian plot.

        The file `moment.dat` contains the moments and the file `pdf.dat` contains the PDFs.

        """
        self.logger.info("\n----- Moment evaluation -----")
        output = self.model(self.sample)
        output = output.sort()
        mean = output.computeMean()
        sd = output.computeStandardDeviationPerComponent()
        sd_min = mean - sd
        sd_max = mean + sd
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
            pdf_pts[i] = np.nan_to_num(pdf_pts[i])

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
