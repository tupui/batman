# coding: utf8
"""
Polynomial Chaos class
======================

"""
import numpy as np
from pathos.multiprocessing import cpu_count
from ..misc import NestedPool
from ..functions import multi_eval
import logging
import openturns as ot
import os


class PC(object):

    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, strategy, degree, distributions, n_sample=None):
        """Generate truncature and projection strategies.

        Allong with the strategies the sample is storred as an attribute:
        :attr:`sample`.

        :param str strategy: Least square or Quadrature ['LS', 'Quad'].
        :param int degree: Polynomial degree.
        :param lst(:class:`openturns.Distribution`) distributions:
        Distributions of each input parameter.
        :param int n_sample: Number of samples for least square.
        """
        # distributions
        in_dim = len(distributions)
        self.dist = ot.ComposedDistribution(distributions)

        enumerateFunction = ot.EnumerateFunction(in_dim)
        basis = ot.OrthogonalProductPolynomialFactory(
            [ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(marginal))
             for marginal in distributions], enumerateFunction)

        self.trunc_strategy = ot.FixedStrategy(
            basis,
            enumerateFunction.getStrataCumulatedCardinal(degree))

        # Strategy choice for expansion coefficient determination
        self.strategy = strategy
        if self.strategy == "LS":  # least-squares method
            montecarlo_design = ot.MonteCarloExperiment(self.dist, n_sample)
            self.proj_strategy = ot.LeastSquaresStrategy(montecarlo_design)
            self.sample, self.weights = self.proj_strategy.getExperiment().generateWithWeights()
        else:  # integration method
            # redefinition of sample size
            # n_sample = (degree + 1) ** in_dim
            # marginal degree definition
            # by default: the marginal degree for each input random
            # variable is set to the total polynomial degree 'degree'+1
            measure = basis.getMeasure()
            degrees = [degree + 1] * in_dim

            self.proj_strategy = ot.IntegrationStrategy(
                ot.GaussProductExperiment(measure, degrees))
            self.sample, self.weights = self.proj_strategy.getExperiment().generateWithWeights()

    def fit(self, input, output):
        """Create the predictor.

        The result of the Polynomial Chaos is stored as ``self.pc_result`` and
        the surrogate is stored as ``self.pc``.

        :param array_like input: The input used to generate the output.
        (n_samples, n_parameters)
        :param array_like output: The observed data.
        (n_samples, [n_features])
        """
        try:
            self.model_len = output.shape[1]
            if self.model_len == 1:
                output = output.ravel()
        except TypeError:
            self.model_len = 1
            output = output.ravel()
        except AttributeError:  # output is None
            self.model_len = 1
        # Define the CPU multi-threading/processing strategy
        try:
            n_cpu_system = cpu_count()
        except NotImplementedError:
            n_cpu_system = os.sysconf('SC_NPROCESSORS_ONLN')
        self.n_cpu = self.model_len
        if n_cpu_system // self.model_len < 1:
            self.n_cpu = n_cpu_system

        def model_fitting(column):
            column = column.reshape((-1, 1))

            input_ = np.zeros_like(self.sample)
            input_[:len(input)] = input
            input_arg = np.where(np.linalg.norm(input_ - np.array(self.sample),
                                                axis=1) <= 1e-2)[0]
            weights = np.array(self.weights)[input_arg]

            pc_algo = ot.FunctionalChaosAlgorithm(input, weights, column,
                                                  self.dist, self.trunc_strategy,
                                                  self.proj_strategy)
            ot.Log.Show(ot.Log.ERROR)
            pc_algo.run()
            pc_result = pc_algo.getResult()
            pc = pc_result.getMetaModel()
            return pc, pc_result

        if self.model_len > 1:
            # pool = NestedPool(self.n_cpu)
            # results = pool.imap(model_fitting, output.T)
            # results = list(results)
            # pool.terminate()
            results = [model_fitting(out) for out in output.T]
        else:
            results = [model_fitting(output)]

        self.pc, self.pc_result = zip(*results)

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param tuple(float) point: The point to evaluate.
        :return: The predictions.
        :rtype: lst

        """
        point_array = np.asarray(point).reshape(1, -1)
        prediction = np.empty((self.model_len))

        # Compute a prediction per predictor
        for i, pc in enumerate(self.pc):
            try:
                prediction[i] = np.array(pc(point_array))
            except ValueError:
                prediction = np.array(pc(point_array))

        return prediction
