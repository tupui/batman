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
data = np.array([[1], [1], [1], [1], [100], [100], [100]])
sample_new = np.array([[0., 5.], [1.5, 5.], [8., 5.], [2.5, 5.], [7.5, 5.]])


class TestMixture:

    @pytest.fixture(scope="session")
    def obj(self):

        algo = Mixture('GaussianMixture', 'SVC', samples, data,
                       {'n_components':2,'random_state':np.random.seed(123456)},{'kernel':'linear'})
        algo.prediction(sample_new)

        return algo

    def test_init(self, obj):
        
        target_clt = np.array([0, 0, 0, 0, 1, 1, 1])
        indice_clt = {0: [0, 1, 2, 3], 1: [4, 5, 6]}
        
        #Error Test
        with pytest.raises(ValueError):
            algo = Mixture('nimp', 'SVC', samples, data,
                       {'n_components':2,'random_state':np.random.seed(123456)},{'kernel':'linear'})
        # Test with Gaussian Mixture
        assert obj.clusterer.get_params()['n_components'] == 2
        npt.assert_almost_equal(obj.label, target_clt, decimal=2)
        assert obj.indice_clt == indice_clt

        predict, sigma = obj.model[0].evaluate(sample_new[0])
        assert predict == 1
        predict, sigma = obj.model[1].evaluate(sample_new[-1])
        assert predict == 100
        
        assert obj.classifier.get_params()['kernel'] == 'linear'        
        #Test with KMeans
        algo = Mixture('KMeans', 'SVC', samples, data,
                       {'n_clusters':2,'random_state':np.random.seed(123456)},{'kernel':'linear'})
        assert algo.clusterer.get_params()['n_clusters'] == 2
        npt.assert_almost_equal(algo.label, target_clt, decimal=2)
        assert algo.indice_clt == indice_clt

        predict, sigma = algo.model[0].evaluate(sample_new[0])
        assert predict == 1
        predict, sigma = algo.model[1].evaluate(sample_new[-1])
        assert predict == 100
        
        assert algo.classifier.get_params()['kernel'] == 'linear'

    def test_prediction(self,obj):
        
        target_clf = np.array([0, 0, 1, 0, 1])
        target_predict = np.array([[1], [1], [100], [1], [100]])
        target_sigma = np.array([[2.068e-05],[7.115e-06],[2.094e-05],[6.828e-06],[1.405e-05]])
        indice_clf = {0: [0, 1, 3], 1: [2, 4]}
        
        predict, sigma, classif = obj.prediction(sample_new)
        
        assert obj.indice_clf == indice_clf
        npt.assert_almost_equal(classif, target_clf, decimal=2)
        npt.assert_almost_equal(predict, target_predict, decimal=2)
        npt.assert_almost_equal(sigma, target_sigma, decimal=2)
        
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