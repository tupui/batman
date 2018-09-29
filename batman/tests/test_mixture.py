# coding: utf8
import numpy as np
import pytest
import numpy.testing as npt
import os
from batman.visualization import reshow
from batman.surrogate import Mixture

samples = np.array([[1., 5.], [2., 5.], [3., 5.],
                    [4., 5.], [5., 5.], [6., 5.], [7., 5.]], dtype=float)
corners = np.array([[1., 5.], [7., 5.]])
data = np.array([[1.], [1.], [1.], [1.], [100.], [100.], [100.]], dtype=float)
sample_new = np.array([[0., 5.], [1.5, 5.], [8., 5.], [2.5, 5.], [10, 5.]], dtype=float)
plabels = ["x1", "x2"]
fsizes = [1]


class TestMixture:

    @pytest.fixture(scope="session")
    def algo(self):
        algo1 = Mixture(samples, data, corners, fsizes, classifier='svm.SVC(kernel="linear")')
        assert algo1.classifier.get_params()['kernel'] == 'linear'
        algo = Mixture(samples, data, corners, fsizes)

        return algo

    def test_init(self, algo, seed):
        indice_clt = {0: [0, 1, 2, 3], 1: [4, 5, 6]}

        # Mixture(samples, data, corners, fsizes, local_method={})

        # Error Test
        with pytest.raises(AttributeError):
            Mixture(samples, data, corners, fsizes, clusterer='foo')

        with pytest.raises(AttributeError):
            Mixture(samples, data, corners, fsizes, classifier='foo')

        # Test with Gaussian Mixture
        assert algo.indice_clt == indice_clt
        predict, sigma = algo.model[0](sample_new[0])
        assert predict == 1
        predict, sigma = algo.model[1](sample_new[-1])
        assert predict == 100
        assert algo.clust.shape[1] == fsizes[0]

    def test_evaluate(self, algo, seed):
        target_clf = np.array([0, 0, 1, 0, 1])
        target_predict = np.array([[1], [1], [100], [1], [100]])
        target_sigma = np.array([[2.068e-05], [7.115e-06], [2.094e-05],
                                 [6.828e-06], [1.405e-05]])

        predict, sigma = algo.evaluate(sample_new)

        npt.assert_almost_equal(algo.classif, target_clf, decimal=2)
        npt.assert_almost_equal(predict, target_predict, decimal=2)
        npt.assert_almost_equal(sigma, target_sigma, decimal=2)

    def test_quality(self, algo):
        target_q2 = 1.0
        target_point = np.array([0, 0.])

        q2, point = algo.estimate_quality()
        npt.assert_almost_equal(q2, target_q2, decimal=2)
        npt.assert_almost_equal(point, target_point, decimal=2)

    def test_boundaries(self, algo, tmp):
        fig, ax = algo.boundaries(samples, fname=os.path.join(tmp, 'boundaries.pdf'))

        fig = reshow(fig)
        ax.plot([3., 3.], [4., 4.])
        fig.savefig(os.path.join(tmp, 'boundaries_change.pdf'))
