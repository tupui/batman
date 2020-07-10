# coding: utf8
import copy
import pytest
import numpy.testing as npt
from batman.uq import (UQ, cosi)
from batman.surrogate import SurrogateModel


def test_indices(tmp, ishigami_data, settings_ishigami):
    surrogate = SurrogateModel('kriging', ishigami_data.space.corners,
                               ishigami_data.space.plabels)
    surrogate.fit(ishigami_data.space, ishigami_data.target_space)

    analyse = UQ(surrogate, nsample=settings_ishigami['uq']['sample'],
                 dists=settings_ishigami['uq']['pdf'],
                 plabels=settings_ishigami['snapshot']['plabels'],
                 method=settings_ishigami['uq']['method'],
                 indices=settings_ishigami['uq']['type'],
                 test=settings_ishigami['uq']['test'])

    indices = analyse.sobol()
    errors = analyse.error_model(indices, 'Ishigami')
    assert errors[0] == pytest.approx(1, 0.2)
    # 2nd order
    assert errors[2] == pytest.approx(0.1, abs=0.2)
    # 1st order
    assert errors[3] == pytest.approx(0.05, abs=0.05)
    # total order
    assert errors[4] == pytest.approx(0.05, abs=0.05)


def test_block(mascaret_data, settings_ishigami):
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['uq']['type'] = 'block'
    test_settings['uq'].pop('test')
    test_settings['snapshot']['plabels'] = ['Ks', 'Q']

    surrogate = SurrogateModel('rbf', mascaret_data.space.corners,
                               mascaret_data.space.plabels)
    surrogate.fit(mascaret_data.space, mascaret_data.target_space)

    analyse = UQ(surrogate, nsample=test_settings['uq']['sample'],
                 dists=['Uniform(15., 60.)', 'Normal(4035., 400.)'],
                 plabels=test_settings['snapshot']['plabels'],
                 method=test_settings['uq']['method'],
                 indices=test_settings['uq']['type'])

    indices = analyse.sobol()
    print(indices)


def test_cosi(ishigami_data):
    ishigami_data_ = copy.deepcopy(ishigami_data)
    ishigami_data_.space.max_points_nb = 5000
    X = ishigami_data_.space.sampling(5000, 'olhs')
    Y = ishigami_data_.func(X).flatten()

    cosi_ = cosi(X, Y)
    npt.assert_almost_equal(cosi_, [0.326, 0.444, 0.014], decimal=2)
