# coding: utf8
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
import otwrapy as otw
from sklearn.metrics import (r2_score, mean_squared_error)
from multiprocessing import cpu_count
# from openturns.viewer import View
from os import mkdir
import itertools
from .wrapper import Wrapper
from ..input_output import (IOFormatSelector, Dataset)
from .. import functions as func_ref


class UQ:

    """Uncertainty Quantification class."""

    logger = logging.getLogger(__name__)

    def __init__(self, pod, settings, output=None):
        """Init the UQ class.

        From the settings file, it gets:

        - Method to use for the Sensitivity Analysis (SA),
        - Type of Sobol' indices to compute,
        - Number of prediction to use for SA,
        - Method to use to predict a new snapshot,
        - The list of input variables,
        - The lengh of the output function.

        Also, it creates the `model` and `int_model` as `ot.PythonFunction()`.

        :param jpod.pod.pod.Pod pod: a POD,
        :param dict settings: The settings file.

        """
        self.logger.info("UQ module")
        try:
            self.test = settings['uq']['test']
        except:
            self.test = None
        self.output_folder = output
        try:
            mkdir(output)
        except OSError:
            self.logger.debug("Output folder already exists.")
        except TypeError:
            self.logger.debug("Not using output folder.")
        self.pod = pod
        self.io = IOFormatSelector(settings['snapshot']['io']['format'])
        self.surrogate = settings['surrogate']['method']
        self.p_lst = settings['snapshot']['io']['parameter_names']
        self.p_len = len(self.p_lst)
        self.output_len = settings['snapshot']['io']['shapes']["0"][0][0]
        self.method_sobol = settings['uq']['method']
        self.type_indices = settings['uq']['type']

        # Generate samples
        self.points_sample = settings['uq']['sample']
        pdf = settings['uq']['pdf']
        input_pdf = "ot." + pdf[0]
        for i in range(self.p_len - 1):
            input_pdf = input_pdf + ", ot." + pdf[i + 1]
        self.distribution = eval("ot.ComposedDistribution(["
                                 + input_pdf
                                 + "], ot.IndependentCopula(self.p_len))")
        experiment = ot.LHSExperiment(self.distribution,
                                      self.points_sample)
        self.sample = experiment.generate()
        self.logger.info("Created {} samples with an LHS experiment"
                         .format(self.points_sample))

        # Get discretization if functionnal output
        try:
            f_eval, _ = self.pod.predict(self.surrogate, [self.sample[0]])
            self.f_input, _ = np.split(f_eval[0].data, 2)
        except:
            self.f_input = None

        # Wrapper for parallelism
        self.n_cpus = 1  # cpu_count()
        self.wrapper = Wrapper(self.pod, self.surrogate,
                               self.p_len, self.output_len)
        self.model = otw.Parallelizer(self.wrapper,
                                      backend='pathos', n_cpus=self.n_cpus)

        self.snapshots = settings['space']['sampling']['init_size']
        self.resamp_size = settings['space']['resampling']['resamp_size']

    def __repr__(self):
        """Information about object."""
        return "UQ object: Method({}), Input({}), Distribution({})".format(self.method_sobol, self.p_lst, self.distribution)

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

        f = func_ref.dispatcher[function]()

        if f.d_out > 1:
            def fun(x):
                return f(x)
        else:
            def fun(x):
                return [f(x)]

        model_ref = ot.PythonFunction(f.d_in, f.d_out, fun)

        s_l2_2nd = np.sqrt(np.sum((f.s_second - indices[0]) ** 2))
        s_l2_1st = np.sqrt(np.sum((f.s_first - indices[1]) ** 2))
        s_l2_total = np.sqrt(np.sum((f.s_total - indices[2]) ** 2))

        # Q2 computation
        y_ref = np.array(model_ref(self.sample))
        y_pred = np.array(self.model(self.sample))
        err_q2 = r2_score(y_ref, y_pred, multioutput='uniform_average')

        # MSE computation
        mse = mean_squared_error(y_ref, y_pred, multioutput='uniform_average')

        self.logger.info("\n----- Surrogate Model Error -----")
        self.logger.info("\nQ2: {}"
                         "\nMSE: {}"
                         "\nL2(sobol 2nd, 1st and total order indices error): "
                         "{}, {}, {}"
                         .format(err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total))

        # Write error to file pod_err.dat
        if self.output_folder is not None:
            with open(self.output_folder + '/pod_err.dat', 'w') as f:
                f.writelines("{} {} {} {} {} {} {}".format(self.snapshots,
                                                           self.resamp_size,
                                                           err_q2,
                                                           self.points_sample,
                                                           s_l2_2nd,
                                                           s_l2_1st,
                                                           s_l2_total))

            # Visual tests
            # if self.output_len == 1:
                # cobweb = ot.VisualTest.DrawCobWeb(self.sample, y_pred, y_min, y_max, 'red', False)
                # View(cobweb).show()
                # qq_plot = ot.VisualTest_DrawQQplot(y_ref, y_pred)
                # qq_plot.draw(self.output_folder + '/qq_plot.png')
                # View(qq_plot).show()
            # else:
            #     self.logger.debug(
            #         "Cannot draw QQplot with output dimension > 1")
        else:
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
            self.wrapper = Wrapper(self.pod, self.surrogate,
                                   self.p_len, 1, block=True)
            int_model = otw.Parallelizer(self.wrapper,
                                         backend='pathos', n_cpus=self.n_cpus)
            sobol_model = int_model
            sobol_len = 1
        else:
            sobol_model = self.model
            sobol_len = self.output_len

        if self.method_sobol == 'sobol':
            self.logger.info("\n----- Sobol' indices -----")

            if float(ot.__version__[:3]) < 1.8:
                experiment = ot.LHSExperiment(self.distribution,
                                              self.points_sample)
                sample2 = experiment.generate()
                sobol = ot.SensitivityAnalysis(self.sample, sample2,
                                               sobol_model)
                sobol.setBlockSize(self.n_cpus)
            else:
                input_design = ot.SobolIndicesAlgorithmImplementation.Generate(
                    self.distribution, self.points_sample, True)
                output_design = sobol_model(input_design)
                sobol = ot.SaltelliSensitivityAlgorithm(input_design,
                                                        output_design,
                                                        self.points_sample)

            for i in range(sobol_len):
                try:
                    indices[0].append(np.array(sobol.getSecondOrderIndices(i)))
                except TypeError:
                    indices[0].append(np.zeros((self.p_len, self.p_len)))
            self.logger.debug("Second order: {}".format(indices[0]))

        elif self.method_sobol == 'FAST':
            self.logger.info("\n----- FAST indices -----")
            sobol = ot.FAST(sobol_model, self.distribution, self.points_sample)
            sobol.setBlockSize(self.n_cpus)
            self.logger.warn("No Second order indices with FAST")

        for i in range(sobol_len):
            try:
                indices[1].append(np.array(sobol.getFirstOrderIndices(i)))
            except TypeError:
                    indices[1].append(np.zeros(self.p_len))
            try:
                indices[2].append(np.array(sobol.getTotalOrderIndices(i)))
            except TypeError:
                    indices[2].append(np.zeros(self.p_len))

        self.logger.debug("First order: {}".format(indices[1]))
        self.logger.debug("Total: {}".format(indices[2]))

        # Write Sobol' indices to file: block or map
        if self.output_folder is not None:
            i1 = np.array(indices[1]).flatten('F')
            i2 = np.array(indices[2]).flatten('F')
            data = np.append(i1, i2)
            names = []
            for p in self.p_lst:
                names += ['S_' + str(p)]
            for p in self.p_lst:
                names += ['S_T_' + str(p)]
            if (self.output_len != 1) and (self.type_indices != 'block'):
                full_names = ['x'] + names
                data = np.append(self.f_input, data)
            else:
                full_names = names

            dataset = Dataset(names=full_names, shape=[self.output_len, 1, 1],
                              data=data)
            self.io.write(self.output_folder + '/sensitivity.dat', dataset)
        else:
            self.logger.debug("No output folder to write indices in")

        # Aggregated Indices
        if self.type_indices == 'aggregated':
            self.logger.info("\n----- Aggregated Sensitivity Indices -----")

            try:
                output_var = output_design.computeVariance()
            except NameError:
                output_design = sobol_model(self.sample)
                output_var = output_design.computeVariance()

            sum_var_indices = [np.zeros((self.p_len, self.p_len)),
                               np.zeros((self.p_len)), np.zeros((self.p_len))]
            for i, j in itertools.product(range(self.output_len), range(3)):
                try:
                    indices[:][j][i] = np.nan_to_num(indices[:][j][i])
                    sum_var_indices[
                        j] += float(output_var[i]) * indices[:][j][i]
                except IndexError:
                    sum_var_indices[j] = np.inf
            sum_var = np.sum(output_var)
            for i in range(3):
                indices[i] = sum_var_indices[i] / sum_var
            self.logger.info("Aggregated_indices: {}".format(indices))

            # Write aggregated indices to file
            if self.output_folder is not None:
                i1 = np.array(indices[1]).flatten('F')
                i2 = np.array(indices[2]).flatten('F')
                data = np.append(i1, i2)
                dataset = Dataset(names=names,
                                  shape=[1, 1, 1],
                                  data=data)
                self.io.write(self.output_folder + '/sensitivity_aggregated.dat',
                              dataset)
            else:
                self.logger.debug(
                    "No output folder to write aggregated indices in")

        # Compute error of the POD with a known function
        if (self.type_indices in ['aggregated', 'block']) and (self.test is not None):
            self.error_pod(indices, self.test)

        return indices

    def error_propagation(self):
        """Compute the moments.

        1st and 2nd order moments are computed for every output of the function.
        It also compute the PDF for these outputs as a 2D cartesian plot.

        The file `moment.dat` contains the moments and the file `pdf.dat` contains the PDFs.

        """
        self.logger.info("\n----- Moment evaluation -----")
        output = self.model(self.sample)
        output = output.sort()

        # Compute statistics
        mean = output.computeMean()
        sd = output.computeStandardDeviationPerComponent()
        sd_min = mean - sd
        sd_max = mean + sd
        min = output.getMin()
        max = output.getMax()

        # Write moments to file
        data = np.append([min], [sd_min, mean, sd_max, max])
        names = ["Min", "SD_min", "Mean", "SD_max", "Max"]
        if (self.output_len != 1) and (self.type_indices != 'block'):
            names = ['x'] + names
            data = np.append(self.f_input, data)

        dataset = Dataset(names=names, shape=[
                          self.output_len, 1, 1], data=data)
        self.io.write(self.output_folder + '/moment.dat', dataset)

        # Covariance and correlation matrices
        if (self.output_len != 1) and (self.type_indices != 'block'):
            correlation_matrix = output.computePearsonCorrelation()
            covariance_matrix = output.computeCovariance()

            x_input_2d, y_input_2d = np.meshgrid(self.f_input, self.f_input)
            x_input_2d = np.array([x_input_2d]).flatten()
            y_input_2d = np.array([y_input_2d]).flatten()

            names = ["x", "y", "Correlation", "Covariance"]
            data_coord = np.append(x_input_2d, y_input_2d)
            data_matrices = np.append(correlation_matrix, covariance_matrix)
            data = np.append(data_coord, data_matrices)
            dataset = Dataset(names=names,
                              shape=[self.output_len, self.output_len, 1],
                              data=data)
            self.io.write(self.output_folder +
                          '/correlation_covariance.dat', dataset)

        # Create the PDFs
        kernel = ot.KernelSmoothing()
        pdf_pts = [None] * self.output_len
        d_PDF = 200
        sample = self.distribution.getSample(d_PDF)
        output_extract = self.model(sample)
        for i in range(self.output_len):
            try:
                pdf = kernel.build(output[:, i])
            except:
                pdf = ot.Normal(output[i, i], 0.001)
            pdf_pts[i] = np.array(pdf.computePDF(output_extract[:, i]))
            pdf_pts[i] = np.nan_to_num(pdf_pts[i])

        # Write PDF to file
        output_extract = np.array(output_extract).flatten('C')
        pdf_pts = np.array(pdf_pts).flatten('F')
        names = ["output", "PDF"]
        if (self.output_len != 1) and (self.type_indices != 'block'):
            names = ['x'] + names
            f_input_2d = np.tile(self.f_input, d_PDF)
            data = np.array([f_input_2d, output_extract, pdf_pts])
        else:
            data = np.array([output_extract, pdf_pts])

        dataset = Dataset(names=names, data=data)
        self.io.write(self.output_folder + '/pdf.dat', dataset)
