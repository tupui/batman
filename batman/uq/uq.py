# coding: utf8
"""
UQ class
========

This class is intented to implement statistical tools provided by the OpenTURNS
framework.

.. seealso:: The documentation of the used class
             :class:`openturns.SensitivityAnalysis`.

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
A. Marrel, N. Saint-Geours. M. De Lozzo: Sensitivity Analysis of Spatial and/or
Temporal Phenomena. Handbook of Uncertainty Quantification. 2015.
DOI:10.1007/978-3-319-11259-6_39-1

B. Iooss: Revue sur l’analyse de sensibilité globale de modèles numériques.
Journal de la Société Française de Statistique. 2010

M. Baudin, A. Dutfoy, B. Iooss, A. Popelin: OpenTURNS: An industrial software
for uncertainty quantification in simulation. 2015. ArXiv ID: 1501.05242

"""
import logging
import os
import itertools
import re
import numpy as np
import openturns as ot
from openturns.viewer import View
from sklearn.metrics import (r2_score, mean_squared_error)
from ..functions.utils import multi_eval
from ..input_output import formater
from .. import functions as func_ref
from .. import visualization


class UQ:
    """Uncertainty Quantification class."""

    logger = logging.getLogger(__name__)

    def __init__(self, surrogate, dists=None, nsample=1000, method='sobol',
                 indices='aggregated', space=None, data=None, plabels=None,
                 xlabel=None, flabel=None, xdata=None, fname=None, test=None,
                 mesh={}):
        """Init the UQ class.

        From the settings file, it gets:

        - Method to use for the Sensitivity Analysis (SA),
        - Type of Sobol' indices to compute,
        - Number of points per sample to use for SA (:math:`N(2p+2)` predictions),
          resulting storage is 6N(out+p)*8 octets => 184Mo if N=1e4
        - Method to use to predict a new snapshot,
        - The list of input variables,
        - The lengh of the output function.

        Also, it creates the `model` and `int_model` as
        :class:`openturns.PythonFunction`.

        :param surrogate: Surrogate model.
        :type surrogate: class:`batman.surrogate.SurrogateModel`.
        :param space: sample space (can be a list).
        :type space: class:`batman.space.Space`.
        :param array_like data: Snapshot's data (n_samples, n_features).
        :param list(str) plabels: parameters' names.
        :param str xlabel: label of the discretization parameter.
        :param str flabel: name of the quantity of interest.
        :param array_like xdata: 1D discretization of the function (n_features,).
        :param str fname: folder output path.
        :param str test: Test function from class:`batman.functions`.

        :param dict mesh: For 2D plots the following keywords are available

            - **fname** (str) -- name of mesh file.
            - **fformat** (str) -- format of the mesh file.
            - **xlabel** (str) -- name of the x-axis.
            - **ylabel** (str) -- name of the y-axis.
            - **vmins** (lst(double)) -- value of the minimal output for data
              filtering.

        """
        self.logger.info("\n----- UQ module -----")
        self.test = test
        self.fname = fname
        self.xlabel = xlabel
        self.flabel = flabel

        self.mesh_kwargs = mesh

        if self.fname is not None:
            try:
                os.mkdir(fname)
            except OSError:
                self.logger.debug("Output folder already exists.")
            finally:
                self.io = formater('json')  # IOFormatSelector('fmt_tp_fortran')
        else:
            self.logger.debug("Not using output folder.")

        self.surrogate = surrogate

        self.p_len = len(dists)
        if plabels is None:
            self.plabels = ["x" + str(i) for i in range(self.p_len)]
        else:
            self.plabels = plabels
        self.method_sobol = method
        self.type_indices = indices
        self.space = space

        # Generate samples
        self.points_sample = nsample
        dists = ','.join(['ot.' + dists[i] for i in range(self.p_len)])
        try:
            self.distribution = eval('ot.ComposedDistribution([' + dists + '])',
                                     {'__builtins__': None},
                                     {'ot': __import__('openturns')})
        except (TypeError, AttributeError):
            self.logger.error('OpenTURNS distribution unknown.')
            raise SystemError
        self.sample = ot.LHSExperiment(self.distribution,
                                       self.points_sample, True, True).generate()
        self.logger.info("Created {} samples with an LHS experiment"
                         .format(self.points_sample))

        try:
            # With surrogate model
            try:
                # Functional output
                f_eval, _ = self.surrogate(self.sample[0])
                self.output_len = len(f_eval[0])
            except ValueError:
                self.output_len = 1

            self.output = self.func(self.sample)
            self.init_size = self.surrogate.space.doe_init
        except TypeError:
            self.sample = space
            self.init_size = len(space)
            self.output_len = data.shape[1]
            self.output = data
            self.points_sample = self.init_size

        if (xdata is None) and (self.output_len > 1):
            self.xdata = np.linspace(0, 1, self.output_len)
        else:
            self.xdata = xdata

    def __repr__(self):
        """Information about object."""
        return ("UQ object: Method({}), Input({}), Distribution({})"
                .format(self.method_sobol, self.plabels, self.distribution))

    @multi_eval
    def func(self, coords):
        """Evaluate the surrogate at a given point.

        This function calls the surrogate to compute a prediction.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The fonction evaluation.
        :rtype: float.
        """
        f_eval, _ = self.surrogate(coords)
        return f_eval[0]

    @multi_eval
    def int_func(self, coords):
        """Evaluate the surrogate at a given point and return the integral.

        Same as :func:`func` but compute the integral using the trapezoidale
        rule. It simply returns the prediction in case of a scalar output.

        :param lst coords: The parameters set to calculate the solution from.
        :return: The integral of the function.
        :rtype: float.
        """
        f_eval, _ = self.surrogate(coords)
        f_eval = np.trapz(f_eval[0])
        return f_eval

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

        A summary is written within `model_err.dat`.

        :param array_like indices: Sobol first order indices computed using the POD.
        :param str function: name of the analytic function.
        :return: err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total.
        :rtype: array_like.
        """
        fun = func_ref.__dict__[function]()

        s_l2_2nd = np.sqrt(np.sum((fun.s_second - indices[0]) ** 2))
        s_l2_1st = np.sqrt(np.sum((fun.s_first - indices[1]) ** 2))
        s_l2_total = np.sqrt(np.sum((fun.s_total - indices[2]) ** 2))

        # Q2 computation
        y_ref = np.array(fun(self.sample))
        y_pred = np.array(self.func(self.sample))
        err_q2 = r2_score(y_ref, y_pred, multioutput='uniform_average')

        # MSE computation
        mse = mean_squared_error(y_ref, y_pred, multioutput='uniform_average')

        self.logger.info("\n----- Surrogate Model Error -----\n"
                         "Q2: {}\n"
                         "MSE: {}\n"
                         "L2(sobol 2nd, 1st and total order indices error): "
                         "{}, {}, {}"
                         .format(err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total))

        # Write error to file model_err.dat
        if self.fname is not None:
            with open(os.path.join(self.fname, 'model_err.dat'), 'w') as f:
                f.writelines("{} {} {} {} {} {}".format(self.init_size, err_q2,
                                                        self.points_sample,
                                                        s_l2_2nd, s_l2_1st,
                                                        s_l2_total))
            # Visual tests
            if fun.d_out == 1:
                # cobweb = ot.VisualTest.DrawCobWeb(self.sample, y_pred, y_min, y_max, 'red', False)
                # View(cobweb).show()
                qq_plot = ot.VisualTest_DrawQQplot(y_ref, y_pred)
                View(qq_plot).save(os.path.join(self.fname, 'qq_plot.png'))
            else:
                self.logger.debug("Cannot draw QQplot with output dimension > 1")
        else:
            self.logger.debug("No output folder to write errors in")

        return err_q2, mse, s_l2_2nd, s_l2_1st, s_l2_total

    def sobol(self):
        """Compute Sobol' indices.

        It returns the second, first and total order indices of Sobol'.
        Two methods are possible for the indices:

        - `sobol`
        - `FAST`

        .. warning:: The second order indices are only available with the sobol
          method. Also, when there is no surrogate (ensemble mode), FAST is not
          available and the DoE must have been generated with `saltelli`.

        And two types of computation are availlable for the global indices:

        - `block`
        - `aggregated`

        If *aggregated*, *map* indices are computed. In case of a scalar value,
        all types returns the same values. *block* indices are written
        within `sensitivity.dat` and aggregated indices within
        `sensitivity_aggregated.dat`.

        Finally, it calls :func:`error_pod` in order to compare the indices
        with their analytical values.

        :return: Sobol' indices.
        :rtype: array_like.
        """
        indices = [[], [], []]
        aggregated = [[], [], []]
        indices_conf = [[], []]

        if self.type_indices == 'block':
            sobol_model = self.int_func
            sobol_len = 1
        else:
            sobol_model = self.func
            sobol_len = self.output_len

        if self.method_sobol == 'sobol':
            self.logger.info("\n----- Sobol' indices -----")

            if self.surrogate is not None:
                size = self.points_sample
                input_design = ot.SobolIndicesExperiment(self.distribution,
                                                         size, True).generate()
                output_design = sobol_model(input_design)
                self.logger.info("Created {} samples for Sobol'"
                                 .format(len(output_design)))
            else:
                input_design = self.space
                output_design = self.output
                size = len(self.space) // (2 * self.p_len + 2)
            # Martinez, Saltelli, MauntzKucherenko, Jansen
            ot.ResourceMap.SetAsBool('MartinezSensitivityAlgorithm-UseAsmpytoticInterval', True)
            sobol = ot.SaltelliSensitivityAlgorithm(input_design,
                                                    output_design, size)

            for i in range(sobol_len):
                try:
                    indices[0].append(np.array(sobol.getSecondOrderIndices(i)))
                except TypeError:
                    indices[0].append(np.zeros((self.p_len, self.p_len)))
            self.logger.debug("-> Second order:\n{}\n".format(indices[0]))

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
            output_design = sobol_model(self.sample)
            self.logger.warning("No Second order indices with FAST")

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

        self.logger.debug("-> First order:\n{}\n"
                          "-> Total:\n{}\n"
                          .format(*indices[1:]))

        # Write Sobol' indices to file: block or map
        if self.fname is not None:
            i1 = np.reshape(indices[1], (sobol_len, self.p_len))
            i2 = np.reshape(indices[2], (sobol_len, self.p_len))
            data = np.append(i1, i2, axis=1)

            names = ['S_{}'.format(p) for p in self.plabels]
            names += ['S_T_{}'.format(p) for p in self.plabels]

            if (self.output_len > 1) and (self.type_indices != 'block'):
                names = ['x'] + names
                data = np.append(np.reshape(self.xdata, (sobol_len, 1)), data, axis=1)
            sizes = [1] * len(names)

            self.io.write(os.path.join(self.fname, 'sensitivity.json'), data, names, sizes)
        else:
            self.logger.debug("No output folder to write indices in")

        # Aggregated Indices
        if self.type_indices == 'aggregated':
            self.logger.info("\n----- Aggregated Sensitivity Indices -----")

            output_var = output_design.var(axis=0)
            sum_var_indices = [np.zeros((self.p_len, self.p_len)),
                               np.zeros((self.p_len)), np.zeros((self.p_len))]

            # Compute manually for FAST and second order, otherwise OT
            if self.method_sobol == 'FAST':
                agg_range = [0, 1, 2]
            else:
                agg_range = [0]
            for i, j in itertools.product(range(self.output_len), agg_range):
                try:
                    indices[:][j][i] = np.nan_to_num(indices[:][j][i])
                    sum_var_indices[j] += float(output_var[i]) * indices[:][j][i]
                except IndexError:
                    sum_var_indices[j] = np.inf
            sum_var = np.sum(output_var)
            for i in range(3):
                aggregated[i] = sum_var_indices[i] / sum_var

            if self.method_sobol != 'FAST':
                aggregated[1] = np.array(sobol.getAggregatedFirstOrderIndices())
                aggregated[2] = np.array(sobol.getAggregatedTotalOrderIndices())
                indices_conf[0] = sobol.getFirstOrderIndicesInterval()
                indices_conf[1] = sobol.getTotalOrderIndicesInterval()

                self.logger.info("-> First order confidence:\n{}\n"
                                 "-> Total order confidence:\n{}\n"
                                 .format(*indices_conf))

            self.logger.info("Aggregated_indices:\n"
                             "-> Second order:\n{}\n"
                             "-> First order:\n{}\n"
                             "-> Total order:\n{}\n"
                             .format(*aggregated))

            # Write aggregated indices to file
            if self.fname is not None:
                i1 = np.array(aggregated[1])
                i2 = np.array(aggregated[2])
                if self.method_sobol != 'FAST':
                    i1_min = np.array(indices_conf[0].getLowerBound())
                    i1_max = np.array(indices_conf[0].getUpperBound())
                    i2_min = np.array(indices_conf[1].getLowerBound())
                    i2_max = np.array(indices_conf[1].getUpperBound())

                    # layout: [S_min_P1, S_min_P2, ..., S_P1, S_p2, ...]
                    data = np.array([i1_min, i1, i1_max, i2_min, i2, i2_max]).flatten()

                    names = [i + str(p) for i, p in
                             itertools.product(['S_min_', 'S_', 'S_max_',
                                                'S_T_min_', 'S_T_', 'S_T_max_'],
                                               self.plabels)]

                    conf = [(i1_max - i1_min) / 2, (i2_max - i2_min) / 2]
                else:
                    conf = None
                    names = [i + str(p) for i, p in
                             itertools.product(['S_', 'S_T_'], self.plabels)]
                    data = np.append(i1, i2)

                self.io.write(os.path.join(self.fname, 'sensitivity_aggregated.json'), data, names)
            else:
                self.logger.debug("No output folder to write aggregated indices in")

            full_indices = [aggregated[1], aggregated[2], indices[1], indices[2]]
        else:
            full_indices = [indices[1][0], indices[2][0]]
            aggregated = [indices[0][0], indices[1][0], indices[2][0]]
            conf = None
            self.xdata = None

        # Plot
        if self.fname:
            path = os.path.join(self.fname, 'sensitivity.pdf')
            plabels = [re.sub(r'(_)(.*)', r'\1{\2}', label)
                       for label in self.plabels]
            visualization.sensitivity_indices(full_indices, plabels=plabels,
                                              conf=conf, xdata=self.xdata,
                                              fname=path)
            path = os.path.join(self.fname, 'sensitivity-polar.pdf')
            visualization.sensitivity_indices(full_indices, plabels=plabels,
                                              conf=conf, polar=True,
                                              xdata=self.xdata, fname=path)
            if self.mesh_kwargs.get('fname'):
                path = os.path.join(self.fname, '1st_order_Sobol_map.pdf')
                visualization.mesh_2D(var=full_indices[2], flabels=plabels,
                                      output_path=path, **self.mesh_kwargs)
                path = os.path.join(self.fname, 'Total_order_Sobol_map.pdf')
                visualization.mesh_2D(var=full_indices[3], flabels=plabels,
                                      output_path=path, **self.mesh_kwargs)

        # Compute error of the POD with a known function
        if (self.type_indices in ['aggregated', 'block'])\
                and (self.test) and (self.surrogate):
            self.error_model(aggregated, self.test)

        return aggregated

    def error_propagation(self):
        """Compute the moments.

        1st, 2nd order moments are computed for every output of the function.
        Also compute the PDF for these outputs, and compute correlations
        (YY and XY) and correlation (YY). Both exported as 2D cartesian plots.
        Files are respectivelly:

        * :file:`pdf-moment.dat`, moments [discretized on curvilinear abscissa]
        * :file:`pdf.dat` -> the PDFs [discretized on curvilinear abscissa]
        * :file:`correlation_covariance.dat` -> correlation and covariance YY
        * :file:`correlation_XY.dat` -> correlation XY
        * :file:`pdf.pdf`, plot of the PDF (with moments if dim > 1)
        """
        self.logger.info("\n----- Uncertainty Propagation -----")

        # Covariance and correlation matrices
        self.logger.info('Creating Covariance/correlation and figures...')
        if (self.output_len > 1) and (self.type_indices != 'block'):
            visualization.corr_cov(self.output, self.sample, self.xdata,
                                   fname=os.path.join(self.fname, 'corr_cov.pdf'))

        # Create and plot the PDFs + moments
        self.logger.info('Creating PDF and figures...')
        visualization.pdf(self.output, self.xdata,
                          xlabel=self.xlabel,
                          flabel=self.flabel,
                          fname=os.path.join(self.fname, 'pdf.pdf'),
                          moments=True)
