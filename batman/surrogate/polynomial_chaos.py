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
from ..functions.utils import multi_eval

ot.ResourceMap.SetAsUnsignedInteger("DesignProxy-DefaultCacheSize", 0)


class PC(object):
    """Polynomial Chaos class."""

    logger = logging.getLogger(__name__)

    def __init__(self, strategy, degree, distributions,sparse_param=None, sample=None,
                 stieltjes=True ):
        """Generate truncature and projection strategies.

        Allong with the strategies the sample is storred as an attribute:
        :attr:`sample` as well as the weights: :attr:`weights`.

        :param str strategy: Least square or Quadrature ['LS', 'Quad', 'SparseLS'].
        :param int degree: Polynomial degree.
        :param  distributions: Distributions of each input parameter.
        :type distributions: lst(:class:`openturns.Distribution`)
        :param int sample: Samples for least square.
        :param bool stieltjes: Wether to use Stieltjes algorithm for the basis.
        :param array sparse_param:(array) -- ((int) Maximum Considered Terms, 
            (int) Most Siginificant number, (float) Significance Factor):
            parameters for the sparse Cleaning Truncation Strategy. In detail,
            the maximun size for the basis trials, the maximum size of the 
            resulting basis, the tolarance factor to discard useless 
            terms. For more info on the method, here a reference
            Dubreuil, Sylvain and  Berveiller , Marc and  Petitjean, Frank and  Salaün,  Michel
            Determination  of  Bootstrap  confidence  intervals  on  sensitivity
            indices obtained  by  polynomial  chaos  expansion.
            (2012)  In:  JFMS12 -Journées  Fiabilité  des Matériaux et des Structures,
            04-06 Jun 2012, Chambéry, France.



        """
        # distributions
        in_dim = len(distributions)
        self.dist = ot.ComposedDistribution(distributions)

        enumerateFunction = ot.EnumerateFunction(in_dim)

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
        if self.strategy == 'LS' or self.strategy =='SparseLS':  # least-squares method
            self.sample = sample
        else:  # integration method
            # redefinition of sample size
            # n_samples = (degree + 1) ** in_dim
            # marginal degree definition
            # by default: the marginal degree for each input random
            # variable is set to the total polynomial degree 'degree'+1
            measure = self.basis.getMeasure()
            degrees = [degree + 1] * in_dim

            self.proj_strategy = ot.IntegrationStrategy(
                ot.GaussProductExperiment(measure, degrees))
            self.sample, self.weights = self.proj_strategy.getExperiment().generateWithWeights()

            if not stieltjes:
                transformation = ot.Function(ot.MarginalTransformationEvaluation(
                    [measure.getMarginal(i) for i in range(in_dim)],
                    distributions, False))
                self.sample = transformation(self.sample)

        self.pc = None
        self.pc_result = None
        self.sparse_param = sparse_param
    def fit(self, sample, data):
        """Create the predictor.

        The result of the Polynomial Chaos is stored as :attr:`pc_result` and
        the surrogate is stored as :attr:`pc`.

        :param array_like sample: The sample used to generate the data
          (n_samples, n_features).
        :param array_like data: The observed data (n_samples, [n_features]).
        """
        trunc_strategy = ot.FixedStrategy(self.basis, self.n_basis)

        if self.strategy == 'LS':  # least-squares method
            proj_strategy = ot.LeastSquaresStrategy(sample, data)
            _, weights = proj_strategy.getExperiment().generateWithWeights()
            
        elif self.strategy == 'SparseLS': 
            app = ot.LeastSquaresMetaModelSelectionFactory(ot.LARS(),ot.CorrectedLeaveOneOut())
            proj_strategy = ot.LeastSquaresStrategy(sample,data,app) #evalstrategy
            _, weights = proj_strategy.getExperiment().generateWithWeights()
            
            
            if self.sparse_param is not None:
                maximumConsideredTerms = int(self.sparse_param[0])
                mostSignificant = int(self.sparse_param[1])
                significanceFactor = self.sparse_param[2]
            else:
                maximumConsideredTerms = 120
                mostSignificant = 30
                significanceFactor = 10e-4

            trunc_strategy = ot.CleaningStrategy(ot.OrthogonalBasis(self.basis), maximumConsideredTerms, mostSignificant, significanceFactor, True)
                    
        else:
            proj_strategy = self.proj_strategy
            sample_ = np.zeros_like(self.sample)
            sample_[:len(sample)] = sample
            sample_arg = np.all(np.isin(sample_, self.sample), axis=1)
            weights = np.array(self.weights)[sample_arg]


        pc_algo = ot.FunctionalChaosAlgorithm(sample, weights, data,
                                              self.dist, trunc_strategy)
        pc_algo.setProjectionStrategy(proj_strategy)

        ot.Log.Show(ot.Log.ERROR)
        pc_algo.run()
        self.pc_result = pc_algo.getResult()
        self.pc = self.pc_result.getMetaModel()

        self.pc, self.pc_result

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
