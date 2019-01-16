# coding: utf8
"""
Polynomial Chaos class
======================

Interpolation using Polynomial Chaos method.

:Example:

::

    >> from batman.surrogate import PC
    >> from batman.functions import Michalewicz
    >> import numpy as np
    >> f = Michalewicz()
    >> surrogate = PC('Quad', 5, [ot.Uniform(0, 1), ot.Uniform(0, 1)])
    >> sample = np.array(surrogate.sample)
    >> sample.shape
    (36, 2)
    >> data = f(sample)
    >> surrogate.fit(sample, data)
    >> point = [0.4, 0.6]
    >> surrogate.evaluate(point)
    array([ -8.642e-08])

"""
import logging
from itertools import product
import openturns as ot
import numpy as np
from ..functions.utils import multi_eval

ot.ResourceMap.SetAsUnsignedInteger("DesignProxy-DefaultCacheSize", 0)


class PC:
    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, strategy, degree, distributions, N_quad=None, sample=None,
                 stieltjes=True, sparse_param={}):
        """Generate truncature and projection strategies.

        Allong with the strategies the sample is storred as an attribute.
        :attr:`sample` as well as corresponding weights: :attr:`weights`.

        :param str strategy: Least square or Quadrature ['LS', 'Quad', 'SparseLS'].
        :param int degree: Polynomial degree.
        :param  distributions: Distributions of each input parameter.
        :type distributions: lst(:class:`openturns.Distribution`)
        :param array_like sample: Samples for least square
          (n_samples, n_features).
        :param bool stieltjes: Wether to use Stieltjes algorithm for the basis.
        :param dict sparse_param: Parameters for the Sparse Cleaning Truncation
          Strategy and/or hyperbolic truncation of the initial basis.

            - **max_considered_terms** (int) -- Maximum Considered Terms,
            - **most_significant** (int), Most Siginificant number to retain,
            - **significance_factor** (float), Significance Factor,
            - **hyper_factor** (float), factor for hyperbolic truncation
              strategy.
        """
        # distributions
        self.in_dim = len(distributions)
        self.dist = ot.ComposedDistribution(distributions)
        self.sparse_param = sparse_param

        if 'hyper_factor' in self.sparse_param:
            enumerateFunction = ot.EnumerateFunction(self.in_dim, self.sparse_param['hyper_factor'])
        else:
            enumerateFunction = ot.EnumerateFunction(self.in_dim)

        if stieltjes:
            # Tend to result in performance issue
            self.basis = ot.OrthogonalProductPolynomialFactory(
                [ot.StandardDistributionPolynomialFactory(
                    ot.AdaptiveStieltjesAlgorithm(marginal))
                 for marginal in distributions], enumerateFunction)
        else:
            self.basis = ot.OrthogonalProductPolynomialFactory(
                [ot.StandardDistributionPolynomialFactory(margin)
                 for margin in distributions], enumerateFunction)

        self.n_basis = enumerateFunction.getStrataCumulatedCardinal(degree)

        # Strategy choice for expansion coefficient determination
        self.strategy = strategy
        if self.strategy == 'LS' or self.strategy == 'SparseLS':  # least-squares method
            self.sample = sample
        else:  # integration method
            # redefinition of sample size
            # n_samples = (degree + 1) ** self.in_dim
            # marginal degree definition
            # by default: the marginal degree for each input random
            # variable is set to the total polynomial degree 'degree'+1
            measure = self.basis.getMeasure()

            if N_quad is not None:
                degrees = [int(N_quad ** 0.25)] * self.in_dim
            else:
                degrees = [degree + 1] * self.in_dim

            self.proj_strategy = ot.IntegrationStrategy(
                ot.GaussProductExperiment(measure, degrees))
            self.sample, self.weights = self.proj_strategy.getExperiment().generateWithWeights()

            if not stieltjes:
                transformation = ot.Function(ot.MarginalTransformationEvaluation(
                    [measure.getMarginal(i) for i in range(self.in_dim)],
                    distributions, False))
                self.sample = transformation(self.sample)

        self.pc = None
        self.pc_result = None

    def fit(self, sample, data):
        """Create the predictor.

        The result of the Polynomial Chaos is stored as :attr:`pc_result` and
        the surrogate is stored as :attr:`pc`. It exposes :attr:`self.weights`,
        :attr:`self.coefficients` and Sobol' indices :attr:`self.s_first` and
        :attr:`self.s_total`.

        :param array_like sample: The sample used to generate the data
          (n_samples, n_features).
        :param array_like data: The observed data (n_samples, [n_features]).
        """
        trunc_strategy = ot.FixedStrategy(self.basis, self.n_basis)

        if self.strategy == 'LS':  # least-squares method
            proj_strategy = ot.LeastSquaresStrategy(sample, data)
            _, self.weights = proj_strategy.getExperiment().generateWithWeights()

        elif self.strategy == 'SparseLS':
            app = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(), ot.CorrectedLeaveOneOut())
            proj_strategy = ot.LeastSquaresStrategy(sample, data, app)
            _, self.weights = proj_strategy.getExperiment().generateWithWeights()

            max_considered_terms = self.sparse_param.get('max_considered_terms', 120)
            most_significant = self.sparse_param.get('most_significant', 30)
            significance_factor = self.sparse_param.get('significance_factor', 1e-3)

            trunc_strategy = ot.CleaningStrategy(ot.OrthogonalBasis(self.basis),
                                                 max_considered_terms,
                                                 most_significant,
                                                 significance_factor, True)
        else:
            proj_strategy = self.proj_strategy
            sample_ = np.zeros_like(self.sample)
            sample_[:len(sample)] = sample
            sample_arg = np.all(np.isin(sample_, self.sample), axis=1)
            self.weights = np.array(self.weights)[sample_arg]

        # PC fitting
        pc_algo = ot.FunctionalChaosAlgorithm(sample, self.weights, data,
                                              self.dist, trunc_strategy)
        pc_algo.setProjectionStrategy(proj_strategy)
        ot.Log.Show(ot.Log.ERROR)
        pc_algo.run()

        # Accessors
        self.pc_result = pc_algo.getResult()
        self.pc = self.pc_result.getMetaModel()
        self.coefficients = self.pc_result.getCoefficients()

        # sensitivity indices
        sobol = ot.FunctionalChaosSobolIndices(self.pc_result)
        self.s_first, self.s_total = [], []
        for i, j in product(range(self.in_dim), range(np.array(data).shape[1])):
            self.s_first.append(sobol.getSobolIndex(i, j))
            self.s_total.append(sobol.getSobolTotalIndex(i, j))

        self.s_first = np.array(self.s_first).reshape(self.in_dim, -1).T
        self.s_total = np.array(self.s_total).reshape(self.in_dim, -1).T

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param array_like point: The point to evaluate (n_features,).
        :return: The predictions.
        :rtype: array_like (n_features,).
        """
        point_array = np.asarray(point).reshape(1, -1)
        prediction = np.array(self.pc(point_array))

        return prediction
