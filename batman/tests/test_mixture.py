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
corners = [[1., 5.], [7., 5.]]

data = np.array([[1.], [1.], [1.], [1.], [100.], [100.], [100.]])
sample_new = np.array([[0., 5.], [1.5, 5.], [8., 5.], [2.5, 5.], [7.5, 5.]])
np.random.seed(123456)
plabels = ["x1", "x2"]

algo1 = Mixture(samples, data, plabels, corners, clusterer='mixture.GaussianMixture(n_components=2, n_init=10)',
                classifier='svm.SVC(kernel="linear")')
algo2 = Mixture(samples, data, plabels, corners, clusterer='cluster.KMeans(n_clusters=2, n_init=10)',
                classifier='gaussian_process.GaussianProcessClassifier()')

class TestMixture:

    def test_init(self):
        indice_clt = {0: [0, 1, 2, 3], 1: [4, 5, 6]}
        #Error Test
        with pytest.raises(AttributeError):
            algo = Mixture(samples, data, plabels, corners, clusterer='nimp')
            algo = Mixture(samples, data, plabels, corners, classifier='nimp')
        # Test with Gaussian Mixture
        assert algo1.indice_clt == indice_clt
        assert algo2.indice_clt == indice_clt

        predict, sigma = algo1.model[0](sample_new[0])
        assert predict == 1
        predict, sigma = algo2.model[0](sample_new[0])
        assert predict == 1
        predict, sigma = algo1.model[1](sample_new[-1])
        assert predict == 100
        predict, sigma = algo2.model[1](sample_new[-1])
        assert predict == 100
        
        assert algo1.classifier.get_params()['kernel'] == 'linear'        

    def test_evaluate(self):
        target_clf = np.array([0, 0, 1, 0, 1])
        target_predict = np.array([[1], [1], [100], [1], [100]])
        target_sigma = np.array([[2.068e-05], [7.115e-06], [2.094e-05], [6.828e-06], [1.405e-05]])
        
        predict1, sigma1, classif1 = algo1.evaluate(sample_new)
        predict2, sigma2, classif2 = algo2.evaluate(sample_new)
        
        npt.assert_almost_equal(classif1, target_clf, decimal=2)
        npt.assert_almost_equal(predict1, target_predict, decimal=2)
        npt.assert_almost_equal(sigma1, target_sigma, decimal=2)
        
        npt.assert_almost_equal(classif2, target_clf, decimal=2)
        npt.assert_almost_equal(predict2, target_predict, decimal=2)
        npt.assert_almost_equal(sigma2, target_sigma, decimal=2)
    
    def test_quality(self):
        target_q2 = 1.0
        target_point = np.array([0., 0.])
        
        
        q2_1, point_1 = algo1.estimate_quality()
        q2_2, point_2 = algo2.estimate_quality()
        
        npt.assert_almost_equal(q2_1, target_q2, decimal=2)
        npt.assert_almost_equal(q2_2, target_q2, decimal=2)
        npt.assert_almost_equal(point_1, target_point, decimal=2)
        npt.assert_almost_equal(point_2, target_point, decimal=2)
        
        
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