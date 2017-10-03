# coding: utf8
"""
Polynomial Chaos class
======================

"""
import numpy as np
from pathos.multiprocessing import (cpu_count, ProcessPool)
from ..functions import multi_eval
import logging
import openturns as ot
import os


class PC(object):

    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, strategy, total_deg, dists, n_sample=None):
        """Generate truncature and projection strategies.

        Allong with the strategies the sample is storred as an attribute:
        :attr:`sample`.

        :param str strategy: Least square or Quadrature ['LS', 'Quad'].
        :param int total_deg: Degree of the polynome.
        :param lst(:class:`openturns.Distribution`) dists: distributions of
        each input parameter.
        :param int n_sample: Number of samples for least square.
        """
        in_dim = len(dists)

        # distributions
        correl = ot.CorrelationMatrix(in_dim)
        copula = ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(correl)
        in_copula = ot.NormalCopula(copula)
        self.dist = ot.ComposedDistribution(dists, in_copula)

        # Choice of orthogonal polynomial families
        # Hermite <-> gaussian, Legendre <-> uniform
        poly_coll = ot.PolynomialFamilyCollection(in_dim)

        for i in range(in_dim):
            poly_coll[i] = ot.StandardDistributionPolynomialFactory(self.dist.getMarginal(i))

        # Polynomial index definition
        poly_index = ot.LinearEnumerateFunction(in_dim)

        # Construction of the polynomial basis forming an orthogonal
        # basis with respect to the joint PDF basis construction
        multivar_basis = ot.OrthogonalProductPolynomialFactory(poly_coll,
                                                               poly_index)
        basis = ot.OrthogonalBasis(multivar_basis)
        dim_basis = poly_index.getStrataCumulatedCardinal(total_deg)

        # Strategy choice for truncature of the orthonormal basis
        self.trunc_strategy = ot.FixedStrategy(basis, dim_basis)

        # Strategy choice for expansion coefficient determination
        if strategy == "LS":      # least-squares method
            montecarlo_design = ot.MonteCarloExperiment(self.dist, n_sample)
            self.proj_strategy = ot.LeastSquaresStrategy(montecarlo_design)
            self.sample = np.array(self.proj_strategy.getExperiment().generate())
        elif strategy == "Quad":  # integration method
            # redefinition of sample size
            # n_sample = (total_deg + 1) ** in_dim
            # marginal degree definition
            # by default: the marginal degree for each input random
            # variable is set to the total polynomial degree 'total_deg'+1
            measure = basis.getMeasure()
            quad = ot.Indices(in_dim)
            for i in range(in_dim):
                quad[i] = total_deg + 1

            self.proj_strategy = ot.IntegrationStrategy(ot.GaussProductExperiment(measure, quad))

            # Convert from [-1, 1] -> input distributions
            marg_inv_transf = ot.MarginalTransformationEvaluation(dists, 1)
            sample = (self.proj_strategy.getExperiment().generate() + 1) / 2.
            self.sample = np.array(marg_inv_transf(sample))
        else:
            self.logger.exception("Not implemented strategy: {}"
                                  .format(strategy))
            raise SystemExit

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
            pc_algo = ot.FunctionalChaosAlgorithm(input, column, self.dist,
                                                  self.trunc_strategy)#,
                                                  # self.proj_strategy)
            pc_algo.run()
            pc_result = pc_algo.getResult()
            pc = pc_result.getMetaModel()
            return pc, pc_result

        if self.model_len > 1:
            pool = ProcessPool(self.n_cpu)
            results = pool.map(model_fitting, output)
            results = list(results)
            pool.terminate()
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
