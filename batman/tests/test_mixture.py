# coding: utf8
import copy
import os
import numpy as np
import pytest
import numpy.testing as npt
from batman.visualization import reshow
from batman.surrogate import Mixture

sample = np.array([[1., 5.], [2., 5.], [3., 5.],
                   [4., 5.], [5., 5.], [6., 5.], [7., 5.]])
corners = np.array([[1., 5.], [7., 5.]])
data = np.array([[1.], [1.], [1.], [1.], [100.], [100.], [100.]])
sample_new = np.array([[0., 5.], [1.5, 5.], [8., 5.], [2.5, 5.], [10, 5.]])
plabels = ["x1", "x2"]
fsizes = 1


class TestMixture:

    @pytest.fixture(scope="session")
    def algo(self):
        np.random.seed(123456)
        algo = Mixture(sample, data, corners, fsizes)

        return algo

    def test_init(self, algo):
        algo1 = Mixture(sample, data, corners, fsizes,
                        classifier='svm.SVC(kernel="linear")')
        assert algo1.classifier.get_params()['kernel'] == 'linear'

        Mixture(sample, data, corners, fsizes,
                local_method=[{'kriging': {'noise': True}},
                              {'kriging': {}}])

        # Error Test
        with pytest.raises(AttributeError):
            Mixture(sample, data, corners, fsizes, clusterer='foo')

        with pytest.raises(AttributeError):
            Mixture(sample, data, corners, fsizes, classifier='foo')

        # Test with Gaussian Mixture
        indice_clt = {0: [0, 1, 2, 3], 1: [4, 5, 6]}
        assert algo.indice_clt == indice_clt
        predict, sigma = algo.local_models[0](sample_new[0])
        assert predict == 1
        predict, sigma = algo.local_models[1](sample_new[-1])
        assert predict == 100

    def test_sensor(self, seed):
        data_shuffled = copy.deepcopy(data)
        np.random.shuffle(data_shuffled)
        data_ = np.concatenate((data_shuffled, data), axis=1)
        algo = Mixture(sample, data_, corners, fsizes)
        indice_clt = {0: [0, 1, 2, 3], 1: [4, 5, 6]}
        assert algo.indice_clt == indice_clt

        algo2 = Mixture(sample, data_shuffled, corners, fsizes)
        assert algo2.indice_clt != indice_clt

    def test_evaluate(self, algo):
        target_clf = np.array([0, 0, 1, 0, 1])
        target_predict = np.array([[1], [1], [100], [1], [100]])
        target_sigma = np.array([[2.068e-05], [7.115e-06], [2.094e-05],
                                 [6.828e-06], [1.405e-05]])

        predict, sigma, classif = algo.evaluate(sample_new, classification=True)

        npt.assert_almost_equal(classif, target_clf, decimal=2)
        npt.assert_almost_equal(predict, target_predict, decimal=2)
        npt.assert_almost_equal(sigma, target_sigma, decimal=2)

    def test_vect(self, mascaret_data, seed):
        sample = mascaret_data.space
        data = mascaret_data.target_space
        corners = sample.corners
        algo = Mixture(sample, data, corners, fsizes=3)
        results = algo.evaluate([[20, 4000], [50, 1000]])
        npt.assert_almost_equal(results[0], [[27.42, 26.43, 25.96],
                                             [22.33, 21.22, 20.89]], decimal=2)

        assert 1 in algo.indice_clt[0]
        assert 0 in algo.indice_clt[1]

    def test_quality(self, algo):
        target_q2 = 1.0
        target_point = np.array([0, 0.])

        q2, point = algo.estimate_quality()
        npt.assert_almost_equal(q2, target_q2, decimal=2)
        npt.assert_almost_equal(point, target_point, decimal=2)

    def test_boundaries(self, g_function_data, tmp):
        sample = g_function_data.space
        data = g_function_data.target_space
        data[:5] *= 10
        corners = np.array(sample.corners)

        # 4D
        algo = Mixture(sample, data, corners, fsizes=1)
        algo.boundaries(sample, fname=os.path.join(tmp, 'boundaries_4d.pdf'))

        # 2D
        algo = Mixture(sample[:, :2], data, corners[:, :2], fsizes=1)
        algo.boundaries(sample[:, :2], fname=os.path.join(tmp, 'boundaries_2d.pdf'))

        # 1D
        algo = Mixture(sample[:, 0].reshape(-1, 1), data, corners[:, 0].reshape(-1, 1), fsizes=1)
        algo.boundaries(sample[:, 0].reshape(-1, 1), fname=os.path.join(tmp, 'boundaries_1d.pdf'))
