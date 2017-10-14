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
import openturns as ot
import numpy as np
from ..functions import multi_eval

ot.ResourceMap.SetAsUnsignedInteger("DesignProxy-DefaultCacheSize", 0)


class PC(object):

    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, strategy, degree, distributions, n_sample=None,
                 stieltjes=True):
        """Generate truncature and projection strategies.

        Allong with the strategies the sample is storred as an attribute:
        :attr:`sample` as well as the weights: :attr:`weights`.

        :param str strategy: Least square or Quadrature ['LS', 'Quad'].
        :param int degree: Polynomial degree.
        :param lst(:class:`openturns.Distribution`) distributions:
        Distributions of each input parameter.
        :param int n_sample: Number of samples for least square.
        :param bool stieltjes: Wether to use Stieltjes algorithm for the basis.
        """
        # distributions
        in_dim = len(distributions)
        self.dist = ot.ComposedDistribution(distributions)

        enumerateFunction = ot.EnumerateFunction(in_dim)

        if stieltjes:
            # Tend to result in performance issue
            basis = ot.OrthogonalProductPolynomialFactory(
                [ot.StandardDistributionPolynomialFactory(ot.AdaptiveStieltjesAlgorithm(marginal))
                 for marginal in distributions], enumerateFunction)
        else:
            basis = ot.OrthogonalProductPolynomialFactory(
                [ot.StandardDistributionPolynomialFactory(margin)
                 for margin in distributions], enumerateFunction)

        self.trunc_strategy = ot.FixedStrategy(
            basis,
            enumerateFunction.getStrataCumulatedCardinal(degree))

        # Strategy choice for expansion coefficient determination
        self.strategy = strategy
        if self.strategy == 'LS':  # least-squares method
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

            if not stieltjes:
                transformation = ot.Function(ot.MarginalTransformationEvaluation(
                    [measure.getMarginal(i) for i in range(in_dim)],
                    distributions, False))
                self.sample = transformation(self.sample)

    def fit(self, sample, data):
        """Create the predictor.

        The result of the Polynomial Chaos is stored as :attr:`pc_result` and
        the surrogate is stored as :attr:`pc`.

        :param array_like sample: The sample used to generate the data. (n_samples, n_features)
        :param array_like data: The observed data. (n_samples, [n_features])
        """
        sample_ = np.zeros_like(self.sample)
        sample_[:len(sample)] = sample
        sample_arg = np.all(np.isin(sample_, self.sample), axis=1)
        weights = np.array(self.weights)[sample_arg]

        pc_algo = ot.FunctionalChaosAlgorithm(sample, weights, data,
                                              self.dist, self.trunc_strategy,
                                              self.proj_strategy)
        ot.Log.Show(ot.Log.ERROR)
        pc_algo.run()
        self.pc_result = pc_algo.getResult()
        self.pc = self.pc_result.getMetaModel()

        self.pc, self.pc_result

    @multi_eval
    def evaluate(self, point):
        """Make a prediction.

        From a point, make a new prediction.

        :param tuple(float) point: The point to evaluate.
        :return: The predictions.
        :rtype: lst

        """
        point_array = np.asarray(point).reshape(1, -1)
        prediction = np.array(self.pc(point_array))

        return prediction
