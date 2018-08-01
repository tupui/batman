# coding: utf8


import numpy as np
import pytest
#from sklearn.metrics import r2_score
#from batman.space import Space
#from batman.functions import Branin
from batman.surrogate import Mixture
import numpy.testing as npt


samples = np.array([[1., 5.], [2., 5.], [3., 5.],
                    [4., 5.], [5., 5.], [6., 5.], [7., 5.]])
data = np.array([[1], [1], [1], [1], [100], [100], [100]]).T
sample_new = np.array([[0., 5.], [1.5, 5.], [8., 5.], [2.5, 5.], [7.5, 5.]])


class TestMixture:

    @pytest.fixture(scope="session")
    def obj(self):

        algo = Mixture('GaussianMixture',data,
                       n_components=2,random_state=np.random.seed(123456))
        algo.clustering()
        algo.localmodels(samples, data)
        algo.classification('SVC', samples, sample_new, kernel="linear")
        algo.prediction(sample_new)

        return algo

    def test_init(self, obj):

        assert obj.clusterer.get_params()['n_components'] == 2

        algo = Mixture('KMeans', data, n_clusters=2)

        assert algo.clusterer.get_params()['n_clusters'] == 2

        with pytest.raises(ValueError):
            algo = Mixture('nimp', data, n_components=2)

    def test_clustering(self, obj):

        target = np.array([0, 0, 0, 0, 1, 1, 1])
        indice = {0: [0, 1, 2, 3], 1: [4, 5, 6]}

        algo = Mixture('KMeans',data,n_clusters=2,
                       random_state=np.random.seed(123456))
        algo.clustering()
        npt.assert_almost_equal(algo.label, target, decimal=2)

        npt.assert_almost_equal(obj.label, target, decimal=2)
        assert obj.indice_clt == indice

    def test_localmodels(self, obj):

        predict, sigma = obj.model[0].evaluate(sample_new[0])
        assert predict == 1
        predict, sigma = obj.model[1].evaluate(sample_new[-1])
        assert predict == 100

    def test_classification(self, obj):

        target = np.array([0, 0, 1, 0, 1])

        assert obj.classifier.get_params()['kernel'] == 'linear'
        npt.assert_almost_equal(obj.classif, target, decimal=2)

    def test_prediction(self, obj):

        target = np.array([[1], [1], [100], [1], [100]])
        indice = {0: [0, 1, 3], 1: [2, 4]}

        assert obj.indice_clf == indice
        predict, sigma = obj.prediction(sample_new)
        npt.assert_almost_equal(predict, target, decimal=2)

#    def test_Branin(self,obj):
#
#        f = Branin()
#        bounds = [[-5,0],[10,15]]
#        space = Space(corners=bounds)
#        sample = space.sampling(n_samples=100)
#        data=f(sample).T
#        sample_new = space.sampling(n_samples=1000)
#        ref=f(sample_new)
#        algo = Mixture('GaussianMixture', data, n_components=2)
#        algo.clustering()
#        algo.localmodels(sample,data)
#        algo.classification('SVC', sample, sample_new, kernel="linear")
#        algo.prediction(sample_new)
#        q2 = r2_score(ref, algo.result, multioutput='uniform_average')
#        assert q2 == pytest.approx(0.9, 0.2)
