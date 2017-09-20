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

    >> analyse = UQ(surrogate, settings, output)
    >> analyse.sobol()
    >> analyse.error_propagation()

References
----------

A. Marrel, N. Saint-Geours. M. De Lozzo: Sensitivity Analysis of Spatial and/or Temporal Phenomena. Handbook of Uncertainty Quantification. 2015.   DOI:10.1007/978-3-319-11259-6_39-1

B. Iooss: Revue sur l’analyse de sensibilité globale de modèles numériques. Journal de la Société Française de Statistique. 2010

M. Baudin, A. Dutfoy, B. Iooss, A. Popelin: OpenTURNS: An industrial software for uncertainty quantification in simulation. 2015. ArXiv ID: 1501.05242


"""
import logging
import numpy as np
import openturns as ot
from sklearn.metrics import (r2_score, mean_squared_error)
from openturns.viewer import View
import os
import itertools
from ..functions import multi_eval
from ..input_output import (IOFormatSelector, Dataset)
from .. import functions as func_ref
from .. import visualization
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend('Agg')


class UQ:

    """Uncertainty Quantification class."""

    logger = logging.getLogger(__name__)

    def __init__(self, settings, surrogate, space=None, data=None, output=None):
        """Init the UQ class.

        From the settings file, it gets:

        - Method to use for the Sensitivity Analysis (SA),
        - Type of Sobol' indices to compute,
        - Number of points per sample to use for SA (:math:`N(2p+2)` predictions),
          resulting storage is 6N(out+p)*8 octets => 184Mo if N=1e4
        - Method to use to predict a new snapshot,
        - The list of input variables,
        - The lengh of the output function.

        Also, it creates the `model` and `int_model` as `ot.PythonFunction()`.

        :param dict settings: The settings file
        :param class:`surrogate.surrogate_model.SurrogateModel` surrogate: a surrogate
        :param class:`space.space.Space` space: sample space (can be a list)
        :param np.array data: snapshot's data
        :param str output: output path
        """
        self.logger.info("\n----- UQ module -----")
        try:
            self.test = settings['uq']['test']
        except:
            self.test = None
        self.output_folder = output
        try:
            os.mkdir(output)
        except OSError:
            self.logger.debug("Output folder already exists.")
        except TypeError:
            self.logger.debug("Not using output folder.")
        self.io = IOFormatSelector(settings['snapshot']['io']['format'])
        self.surrogate = surrogate
        self.p_lst = settings['snapshot']['io']['parameter_names']
        self.p_len = len(self.p_lst)
        self.method_sobol = settings['uq']['method']
        self.type_indices = settings['uq']['type']

        # Generate samples
        self.points_sample = settings['uq']['sample']
        input_pdf = ','.join(['ot.' + settings['uq']['pdf'][i]
                              for i in range(self.p_len)])
        self.distribution = eval("ot.ComposedDistribution(["
                                 + input_pdf
                                 + "], ot.IndependentCopula(self.p_len))")
        self.experiment = ot.LHSExperiment(self.distribution,
                                           self.points_sample,
                                           True, True)
        self.sample = self.experiment.generate()
        self.logger.info("Created {} samples with an LHS experiment"
                         .format(self.points_sample))

        self.init_size = settings['space']['sampling']['init_size']
        try:
            self.resamp_size = settings['space']['resampling']['resamp_size']
        except KeyError:
            self.resamp_size = 0

        # Get discretization if functionnal output
        try:
            # With surrogate model
            try:
                f_eval, _ = self.surrogate(self.sample[0])
                self.f_input, _ = np.split(f_eval[0], 2)
                self.output_len = len(self.f_input)
            except ValueError:
                self.f_input = None
                self.output_len = 1

            self.model = self.func
            self.output = ot.Sample(self.model(self.sample))
        except TypeError:
            self.sample = space
            try:
                f_input, output = np.split(np.array(data), 2, axis=1)
                self.f_input = f_input[0]
                self.output_len = len(self.f_input)
            except ValueError:
                self.f_input = None
                self.output_len = 1
                output = data

            self.output = ot.Sample(output)
            self.points_sample = self.init_size

    def __repr__(self):
        """Information about object."""
        return ("UQ object: Method({}), Input({}), Distribution({})"
                .format(self.method_sobol, self.p_lst, self.distribution))

    @multi_eval
    def func(self, coords):
        """Evaluate the surrogate at a given point.

        This function calls the surrogate to compute a prediction.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float

        """
        f_eval, _ = self.surrogate(coords)
        try:
            _, f_eval = np.split(f_eval[0], 2)
        except:
            pass
        return f_eval

    @multi_eval
    def int_func(self, coords):
        """Evaluate the surrogate at a given point and return the integral.

        Same as :func:`func` but compute the integral using the trapezoidale
        rule. It simply returns the prediction in case of a scalar output.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The integral of the function.
        :rtype: float

        """
        f_eval, _ = self.surrogate(coords)
        try:
            f_input, f_eval = np.split(f_eval[0], 2)
            int_f_eval = np.trapz(f_eval, f_input)
        except:
            int_f_eval = f_eval
        return int_f_eval

    def error_model(self, indices, function):
        r"""Compute the error between the POD and the analytic function.

        .. warning:: For test purpose only. Choises are `Ishigami`,
           `Rosenbrock`, `Michalewicz`, `G_Function` and `Channel_Flow` test
           functions.

        From the surrogate of the function, evaluate the error
        using the analytical evaluation of the function on the sample points.

        .. math:: Q^2 = 1 - \frac{err_{l2}}{var_{model}}

        Knowing that :math:`err_{l2} = \sum \frac{(prediction - reference)^2}{n}`,
        :math:`var_{model} = \sum \frac{(prediction - mean)^2}{n}`

        Also, it computes the mean square error on the Sobol first andtotal
        order indices.

        A summary is written within `pod_err.dat`.

        :param lst(array) indices: Sobol first order indices computed using the POD.
        :param str function: name of the analytic function.
        :return: err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total
        :rtype: float
        """
        fun = func_ref.__dict__[function]()

        if fun.d_out > 1:
            wrap_fun = fun
        else:
            def wrap_fun(x):
                return [fun(x)]

        model_ref = ot.PythonFunction(fun.d_in, fun.d_out, wrap_fun)

        s_l2_2nd = np.sqrt(np.sum((fun.s_second - indices[0]) ** 2))
        s_l2_1st = np.sqrt(np.sum((fun.s_first - indices[1]) ** 2))
        s_l2_total = np.sqrt(np.sum((fun.s_total - indices[2]) ** 2))

        # Q2 computation
        y_ref = np.array(model_ref(self.sample))
        y_pred = np.array(self.model(self.sample))
        err_q2 = r2_score(y_ref, y_pred, multioutput='uniform_average')

        # MSE computation
        mse = mean_squared_error(y_ref, y_pred, multioutput='uniform_average')

        self.logger.info("\n----- Surrogate Model Error -----\n"
                         "Q2: {}\n"
                         "MSE: {}\n"
                         "L2(sobol 2nd, 1st and total order indices error): "
                         "{}, {}, {}"
                         .format(err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total))

        # Write error to file pod_err.dat
        if self.output_folder is not None:
            with open(self.output_folder + '/pod_err.dat', 'w') as f:
                f.writelines("{} {} {} {} {} {} {}".format(self.init_size,
                                                           self.resamp_size,
                                                           err_q2,
                                                           self.points_sample,
                                                           s_l2_2nd,
                                                           s_l2_1st,
                                                           s_l2_total))

            # Visual tests
            if fun.d_out == 1:
                # cobweb = ot.VisualTest.DrawCobWeb(self.sample, y_pred, y_min, y_max, 'red', False)
                # View(cobweb).show()
                qq_plot = ot.VisualTest_DrawQQplot(y_ref, y_pred)
                View(qq_plot).save(self.output_folder + '/qq_plot.png')
                # View(qq_plot).show()
            else:
                self.logger.debug(
                    "Cannot draw QQplot with output dimension > 1")
        else:
            self.logger.debug("No output folder to write errors in")

        return err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total

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

        If *aggregated*, *map* indices are computed. In case of a scalar value,
        all types returns the same values. *map* or *block* indices are written
        within `sensitivity.dat` and aggregated indices within `sensitivity_aggregated.dat`.

        Finally, it calls :func:`error_pod` in order to compare the indices
        with their analytical values.

        :return: Sobol' indices
        :rtype: lst(np.array)

        """
        indices = [[], [], []]
        indices_conf = [[], []]

        if self.type_indices == 'block':
            sobol_model = self.int_func
            sobol_len = 1
        else:
            sobol_model = self.model
            sobol_len = self.output_len

        if self.method_sobol == 'sobol':
            self.logger.info("\n----- Sobol' indices -----")

            if float(ot.__version__[:3]) < 1.8:
                sample2 = self.experiment.generate()
                sobol = ot.SensitivityAnalysis(self.sample, sample2,
                                               sobol_model)
                sobol.setBlockSize(self.n_cpus)
            else:
                input_design = ot.SobolIndicesAlgorithmImplementation.Generate(
                    self.distribution, self.points_sample, True)
                output_design = sobol_model(input_design)
                # Martinez, Saltelli, MauntzKucherenko, Jansen
                ot.ResourceMap.SetAsBool('MartinezSensitivityAlgorithm-UseAsmpytoticInterval', True)
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
            if self.output_len > 1:
                wrap_fun = sobol_model
            else:
                def wrap_fun(x):
                    return [sobol_model(x)]

            fast_model = ot.PythonFunction(self.p_len, self.output_len, wrap_fun)
            sobol = ot.FAST(ot.Function(fast_model),
                            self.distribution, self.points_sample)
            self.logger.warn("No Second order indices with FAST")

        # try block used to handle boundary conditions with fixed values
        for i in range(sobol_len):
            try:
                indices[1].append(np.array(sobol.getFirstOrderIndices(i)))
            except TypeError:
                    indices[1].append(np.zeros(self.p_len))
            try:
                indices[2].append(np.array(sobol.getTotalOrderIndices(i)))
            except TypeError:
                    indices[2].append(np.zeros(self.p_len))

        self.logger.debug("First order: {}"
                          "Total: {}"
                          .format(*indices[1:]))

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

            # Plot
            path = self.output_folder + '/sensitivity_map.pdf'
            visualization.sobol(indices[1:], p_lst=self.p_lst,
                                xdata=self.f_input, fname=path)
        else:
            self.logger.debug("No output folder to write indices in")

        # Aggregated Indices
        if self.type_indices == 'aggregated':
            self.logger.info("\n----- Aggregated Sensitivity Indices -----")

            try:
                output_var = output_design.var(axis=0)
            except NameError:
                output_design = sobol_model(self.sample)
                output_var = output_design.var(axis=0)

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

            if (float(ot.__version__[:3]) >= 1.8) and (self.method_sobol != 'FAST'):
                indices[1] = np.array(sobol.getAggregatedFirstOrderIndices())
                indices[2] = np.array(sobol.getAggregatedTotalOrderIndices())
                indices_conf[0] = sobol.getFirstOrderIndicesInterval()
                indices_conf[1] = sobol.getTotalOrderIndicesInterval()

                self.logger.info("First order confidence: {}"
                                 "Total order confidence: {}"
                                 .format(*indices_conf))

            self.logger.info("Aggregated_indices: {}".format(indices))

            # Write aggregated indices to file
            if self.output_folder is not None:
                ind_total_first = np.array(indices[1:]).flatten('F')
                i1 = np.array(indices[1]).flatten('F')
                i2 = np.array(indices[2]).flatten('F')
                if (float(ot.__version__[:3]) >= 1.8) and (self.method_sobol != 'FAST'):
                    i1_min = np.array(indices_conf[0].getLowerBound()).flatten('F')
                    i1_max = np.array(indices_conf[0].getUpperBound()).flatten('F')
                    i2_min = np.array(indices_conf[1].getLowerBound()).flatten('F')
                    i2_max = np.array(indices_conf[1].getUpperBound()).flatten('F')

                    data = np.append([i1_min], [i1, i1_max, i2_min, i2, i2_max])

                    names = [i + str(p) for i, p in
                             itertools.product(['S_min_', 'S_', 'S_max_',
                                                'S_T_min_', 'S_T_', 'S_T_max_'],
                                               self.p_lst)]

                    conf1 = np.vstack((i1_min, i2_min)).flatten('F')
                    conf1 = ind_total_first - conf1
                    conf2 = np.vstack((i1_max, i2_max)).flatten('F')
                    conf2 -= ind_total_first
                    conf = np.vstack((conf1, conf2))
                else:
                    conf = None
                    names = [i + str(p) for i, p in
                             itertools.product(['S_', 'S_T_'],
                                               self.p_lst)]
                    data = np.append(i1, i2)
                dataset = Dataset(names=names, shape=[1, 1, 1], data=data)
                self.io.write(self.output_folder + '/sensitivity_aggregated.dat',
                              dataset)

                # Plot
                path = self.output_folder + '/sensitivity_aggregated.pdf'
                visualization.sobol(indices[1:], conf=conf, p_lst=self.p_lst,
                                    fname=path)
            else:
                self.logger.debug(
                    "No output folder to write aggregated indices in")

        # Compute error of the POD with a known function
        if (self.type_indices in ['aggregated', 'block']) and (self.test is not None):
            self.error_model(indices, self.test)

        return indices

    def error_propagation(self):
        """Compute the moments.

        1st, 2nd order moments are computed for every output of the function.
        Also compute the PDF for these outputs, and compute correlations
        (YY and XY) and correlation (YY). Both exported as 2D cartesian plots.
        Files are respectivelly:

        * :file:`moment.dat`, the moments [discretized on curvilinear abscissa]
        * :file:`pdf.dat` -> the PDFs [discretized on curvilinear abscissa]
        * :file:`correlation_covariance.dat` -> correlation and covariance YY
        * :file:`correlation_XY.dat` -> correlation XY


        """
        self.logger.info("\n----- Moment evaluation -----")
        output = self.output.sort()

        # Compute statistics
        mean = output.computeMean()
        sd = output.computeStandardDeviationPerComponent()
        sd_min = mean - sd
        sd_max = mean + sd
        min_ = output.getMin()
        max_ = output.getMax()

        # Write moments to file
        data = np.append([min_], [sd_min, mean, sd_max, max_])
        names = ["Min", "SD_min", "Mean", "SD_max", "Max"]
        if (self.output_len != 1) and (self.type_indices != 'block'):
            names = ['x'] + names
            data = np.append(self.f_input, data)

        dataset = Dataset(names=names, shape=[self.output_len, 1, 1], data=data)
        self.io.write(self.output_folder + '/moment.dat', dataset)

        # Covariance and correlation matrices
        if (self.output_len != 1) and (self.type_indices != 'block'):
            corr_yy = np.array(self.output.computePearsonCorrelation())
            cov_yy = np.array(self.output.computeCovariance())

            x_input_2d, y_input_2d = np.meshgrid(self.f_input, self.f_input)
            data = np.append(x_input_2d, [y_input_2d, corr_yy, cov_yy])
            dataset = Dataset(names=['x', 'y', 'Correlation-YY', 'Covariance'],
                              shape=[self.output_len, self.output_len, 1],
                              data=data)
            self.io.write(self.output_folder +
                          '/correlation_covariance.dat', dataset)

            cov_matrix_XY = np.dot((np.mean(self.sample) - self.sample).T,
                                   np.array(mean) - self.output) / (self.points_sample - 1)

            x_input_2d, y_input_2d = np.meshgrid(self.f_input,
                                                 np.arange(self.p_len))
            data = np.append(x_input_2d, [y_input_2d, cov_matrix_XY])
            dataset = Dataset(names=['x', 'y', 'Correlation-XY'],
                              shape=[self.p_len, self.output_len, 1],
                              data=data)
            self.io.write(self.output_folder + '/correlation_XY.dat', dataset)

        # Create and plot the PDFs
        visualization.pdf(np.array(self.output), self.f_input,
                          fname=os.path.join(self.output_folder, 'pdf.pdf'))
