# coding: utf8

import pytest
import numpy as np
import numpy.testing as npt
import copy
from batman.space import (Point, Space, Doe,
                          UnicityError, AlienPointError, FullSpaceError)
from batman.functions import (Ishigami, Branin)
from batman.tasks import Snapshot
from batman.surrogate import SurrogateModel
from batman.space.refiner import Refiner

settings = {
    "space": {
        "corners": [[1.0, 1.0, 1.0], [3.1415, 3.1415, 3.1415]],
        "sampling": {
            "init_size": 10,
            "method": "halton"
        },
        "resampling": {
            "delta_space": 0.08,
            "resamp_size": 6,
            "method": "sigma",
            "hybrid": [["sigma", 4], ["loo_sobol", 2]],
            "q2_criteria": 0.8
        }
    }
}


def test_point():
    point_a = Point([2, 3, 9])
    point_b = Point([1, 2, 8])
    point_c = Point([2, 3, 9])
    assert point_a != point_b
    assert point_a == point_c

    Point.set_threshold(2)
    point_a = Point([2, 3, 9])
    point_b = Point([1, 2, 8])
    point_c = Point([2, 3, 9])
    assert point_a == point_b
    assert point_a == point_c

    with pytest.raises(ValueError):
        Point.set_threshold(-0.1)

    with pytest.raises(ValueError):
        Point([2, 's', 9])

    Point.set_threshold(0)


def test_point_evaluation():
    f_3d = Ishigami()
    point = Point([2.20, 1.57, 3])
    target_point = f_3d(point)
    assert target_point == pytest.approx(14.357312835804658, 0.05)


def test_space():
    space = Space(settings)

    space += (1, 2, 3)
    assert space[:] == [(1, 2, 3)]

    space.empty()
    assert space[:] == []

    space += [(1, 2, 3), (1, 1, 3)]
    assert space[:] == [(1, 2, 3), (1, 1, 3)]

    s1 = space.sampling()
    space2 = Space(settings)
    s2 = space2.sampling(10, kind='lhsc')
    assert s1[:] != s2[:]

    space.empty()
    with pytest.raises(UnicityError):
        space += (1, 2, 3)
        space += (1, 2, 3)

    with pytest.raises(AlienPointError):
        space += (1, 7, 3)

    with pytest.raises(FullSpaceError):
        space.sampling(17)


def test_space_evaluation():
    f_3d = Ishigami()
    space = Space(settings)
    space.sampling(2)
    targets_space = f_3d(space)

    f_data_base = np.array([8.10060038,  5.18818004]).reshape(2, 1)
    npt.assert_almost_equal(targets_space, f_data_base)


def test_doe():
    bounds = np.array([[0, 2], [10, 5]])
    discrete_var = 0
    n = 5

    kind = 'uniform'
    doe = Doe(n, bounds, kind, discrete_var)
    sample = doe.generate()
    out = np.array([[0., 2.], [10., 2.], [0., 5.], [10., 5.]])
    npt.assert_almost_equal(sample, out, decimal=1)

    kind = 'discrete'
    doe = Doe(n, bounds, kind, discrete_var)
    sample = doe.generate()
    out = np.array([[5., 3.], [2., 4.], [8., 2.3], [1., 3.3], [6., 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)

    kind = 'halton'
    doe = Doe(n, bounds, kind, discrete_var)
    sample = doe.generate()
    out = np.array([[5., 3.], [2.5, 4.], [7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)

    kind = 'sobolscramble'
    doe = Doe(n, bounds, kind, discrete_var)
    sample = doe.generate()


def test_resampling(tmp, branin_data, settings_ishigami):
    f_2d, dists, model, point, target_point, space, target_space = branin_data
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['sampling']['init_size'] = len(space)
    test_settings['space']['resampling']['method'] = 'sigma'
    test_settings['space']['resampling']['resamp_size'] = 1
    test_settings['space']['corners'] = space.corners
    test_settings['space']['corners'] = space.corners
    test_settings['uq']['test'] = 'Branin'
    test_settings['snapshot']['io']['parameter_names'] = ['x1', 'x2']
    f_obj = Branin()
    test_settings['snapshot']['provider'] = f_obj

    Snapshot.initialize(test_settings['snapshot']['io'])
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, target_space)

    out = space.refine(surrogate)
    assert len(space) == 11

    refiner = Refiner(surrogate, test_settings)
    new_point = refiner.sigma()
    point_loo = refiner.points[0]
    new_point = refiner.leave_one_out_sigma(point_loo)
    new_point = refiner.leave_one_out_sobol(point_loo)


def test_discrepancy(settings_ishigami):
    test_settings = copy.deepcopy(settings)
    test_settings['space']['corners'] = [[0.5, 0.5], [6.5, 6.5]]
    space_1 = Space(test_settings)
    space_2 = Space(test_settings)

    space_1 += [[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]]
    space_2 += [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]]

    assert space_1.discrepancy() == pytest.approx(0.0081, abs=0.0001)
    assert space_2.discrepancy() == pytest.approx(0.0105, abs=0.0001)
