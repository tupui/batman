# coding: utf8
"""
Polynomial Chaos class
======================

"""
import numpy as np
from pathos.multiprocessing import (cpu_count, ProcessPool)
import logging
import openturns as ot


class PC(object):

    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, input=None, output=None,
                 function=None, input_dists=None, out_dim=None,
                 n_sample=None, strategy=None, total_deg=None):
        """Create the predictor.

        The result of the Polynomial Chaos is stored as ``self.pc_result`` and
        the surrogate is stored as ``self.pc``.

        :param ndarray input:
        :param ndarray output:
        :param callable function:
        :param lst(ot.Dist) input_dists:
        :param int out_dim:
        """
        try:
            self.model_len = len(output)
        except TypeError:
            self.model_len = 1
        self.pc = [None] * self.model_len
        self.pc_result = [None] * self.model_len
        # Define the CPU multi-threading/processing strategy
        try:
            n_cpu_system = cpu_count()
        except NotImplementedError:
            n_cpu_system = os.sysconf('SC_NPROCESSORS_ONLN')
        self.n_cpu = self.model_len
        if n_cpu_system // self.model_len < 1:
            self.n_cpu = n_cpu_system

        if function:
            self.logger.info("Polynomial Chaos without prior input/output")
            in_dim = len(input_dists)
            if out_dim > 1:
                def wrap_fun(x):
                    return function(x)
            else:
                def wrap_fun(x):
                    return [function(x)]
            model = ot.PythonFunction(in_dim, out_dim, wrap_fun)

            # distributions
            correl = ot.CorrelationMatrix(in_dim)
            copula = ot.NormalCopula.GetCorrelationFromSpearmanCorrelation(correl)
            in_copula = ot.NormalCopula(copula)
            input_dists = ot.ComposedDistribution(input_dists, in_copula)

            # choice of orthogonal polynomial families
            # Hermite <-> gaussian, Legendre <-> uniform
            poly_coll = ot.PolynomialFamilyCollection(in_dim)

            for i in range(in_dim):
                poly_coll[i] = ot.StandardDistributionPolynomialFactory(input_dists.getMarginal(i))

            # polynomial index definition
            poly_index = ot.LinearEnumerateFunction(in_dim)

            # construction of the polynomial basis forming an orthogonal
            # basis with respect to the joint PDF basis construction
            multivar_basis = ot.OrthogonalProductPolynomialFactory(poly_coll,
                                                                   poly_index)
            basis = ot.OrthogonalBasis(multivar_basis)
            dim_basis = poly_index.getStrataCumulatedCardinal(total_deg)

            # strategy choice for truncature of the orthonormal basis
            trunc_strategy = ot.FixedStrategy(basis, dim_basis)

            # strategy choice for expansion coefficient determination
            if strategy == "LS":   # least-squares method
                montecarlo_design = ot.MonteCarloExperiment(input_dists, n_sample)
                proj_strategy = ot.LeastSquaresStrategy(montecarlo_design)

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
                proj_strategy = ot.IntegrationStrategy(ot.GaussProductExperiment(measure, quad))
            else:
                self.logger.exception("Not implemented strategy: {}"
                                      .format(strategy))
                raise SystemExit

            pc_algo = ot.FunctionalChaosAlgorithm(model, input_dists,
                                                  trunc_strategy, proj_strategy)
            pc_algo.run()
            self.sample = np.array(pc_algo.getInputSample())
            self.pc_result[0] = pc_algo.getResult()
            self.pc[0] = self.pc_result[0].getMetaModel()
        else:
            self.logger.info("Polynomial Chaos with prior input/output")
            def model_fitting(column):
                column = column.reshape((len(input), 1))
                pc_algo = ot.FunctionalChaosAlgorithm(input, column)
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
                output = output[0].reshape((len(input), 1))
                results = [model_fitting(output)]

            for i in range(self.model_len):
                self.pc[i], self.pc_result[i] = results[i]
            self.logger.info("Done")

    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param tuple(float) point: The point to evaluate.
        :return: The predictions.
        :rtype: lst

        """
        prediction = np.ndarray((self.model_len))

        # Compute a prediction per predictor
        for i, pc in enumerate(self.pc):
            try:
                prediction[i] = np.array(pc(point))
            except ValueError:
                prediction = np.array(pc(point))

        return prediction
