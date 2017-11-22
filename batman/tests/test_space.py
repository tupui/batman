# coding: utf8
import copy
import pytest
import numpy as np
import numpy.testing as npt
from batman.space import (Point, Space, Doe)
from batman.functions import Ishigami
from batman.surrogate import SurrogateModel
from batman.space.refiner import Refiner
import openturns as ot


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


def test_space(settings_ishigami):
    corners = settings_ishigami['space']['corners']
    space = Space(corners)
    assert space.max_points_nb == np.inf

    space = Space(corners, sample=10)
    assert space.max_points_nb == 10

    space = Space(corners, sample=10, nrefine=6,
                  plabels=['x', 'y', 'z'])

    assert space.max_points_nb == 16

    space += (1, 2, 3)
    assert space[:] == [(1, 2, 3)]

    space.empty()
    assert space[:] == []

    space += [(1, 2, 3), (1, 1, 3)]
    assert space[:] == [(1, 2, 3), (1, 1, 3)]

    s1 = space.sampling()
    assert len(s1) == 10
    space2 = Space(corners,
                   sample=settings_ishigami['space']['sampling']['init_size'],
                   nrefine=settings_ishigami['space']['resampling']['resamp_size'])
    ot.RandomGenerator.SetSeed(123456)
    s2 = space2.sampling(10, kind='lhsc')
    assert len(s2) == 10
    assert s1[:] != s2[:]

    space.empty()
    space += (1, 2, 3)
    space += (1, 2, 3)
    assert len(space) == 1

    space = Space(corners, sample=16, duplicate=True)
    space += (1, 2, 3)
    space += (1, 2, 3)
    assert len(space) == 2

    space += (1, 2)
    assert len(space) == 2

    space += (1, 7, 3)
    assert len(space) == 2

    space.sampling(17)
    assert len(space) == 16

    space.empty()
    dists = ['Uniform(0., 1.)', 'Uniform(-1., 2.)', 'Uniform(-2., 3.)']
    space.sampling(5, kind='halton', dists=dists)
    out = [(0.5, 0.0, -1.0), (0.25, 1.0, 0.0), (0.75, -0.67, 1.0),
           (0.125, 0.33, 2.0), (0.625, 1.33, -1.8)]
    npt.assert_almost_equal(space, out, decimal=1)

    space = Space(corners, sample=np.array([(1, 2, 3), (1, 1, 3)]))
    assert space.doe_init == 2
    assert space.max_points_nb == 2

    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['corners'][1] = [np.pi, -np.pi, np.pi]
    with pytest.raises(ValueError):
        Space(test_settings['space']['corners'])


def test_space_evaluation(settings_ishigami):
    f_3d = Ishigami()
    space = Space(settings_ishigami['space']['corners'])
    space.sampling(2, 'halton')
    targets_space = f_3d(space)
    f_data_base = np.array([5.25, 4.2344145]).reshape(2, 1)
    npt.assert_almost_equal(targets_space, f_data_base)


def test_doe():
    bounds = np.array([[0, 2], [10, 5]])
    n = 5

    doe = Doe(n, bounds, 'uniform', discrete=0)
    sample = doe.generate()
    out = [[0., 2.], [10., 2.], [0., 5.], [10., 5.]]
    npt.assert_almost_equal(sample, out, decimal=1)

    doe = Doe(n, bounds, 'halton', discrete=0)
    sample = doe.generate()
    out = [[5., 3.], [2., 4.], [8., 2.3], [1., 3.3], [6., 4.3]]
    npt.assert_almost_equal(sample, out, decimal=1)

    doe = Doe(n, bounds, 'halton', discrete=1)
    sample = doe.generate()
    out = [[5, 3], [2.5, 4], [7.5, 2], [1.25, 3], [6.25, 5]]
    npt.assert_almost_equal(sample, out, decimal=1)

    doe = Doe(n, np.array([[0, 2, -2], [10, 5, -1]]), 'halton', discrete=1)
    sample = doe.generate()
    out = [[5, 3, -1.8], [2.5, 4, -1.6], [7.5, 2, -1.4],
           [1.25, 3, -1.2], [6.25, 5, -1.96]]
    npt.assert_almost_equal(sample, out, decimal=1)

    doe = Doe(n, bounds, 'halton')
    sample = doe.generate()
    out = [[5., 3.], [2.5, 4.], [7.5, 2.3], [1.25, 3.3], [6.25, 4.3]]
    npt.assert_almost_equal(sample, out, decimal=1)

    doe = Doe(n, bounds, 'sobolscramble', discrete=0)
    sample = doe.generate()

    doe = Doe(n, bounds, 'lhsopt')
    ot.RandomGenerator.SetSeed(123456)
    sample = doe.generate()
    out = [[6.149, 2.343], [9.519, 3.497], [1.991, 4.058],
           [5.865, 4.995], [2.551, 2.737]]
    npt.assert_almost_equal(sample, out, decimal=1)

    bounds = [[15.0, 2500.0], [60.0, 6000.0]]

    with pytest.raises(SystemError):
        dists = ['Um(15., 60.)', 'Normal(4035., 400.)']
        doe = Doe(n, bounds, 'halton', dists)

    dists = ['Uniform(15., 60.)', 'Normal(4035., 400.)']
    doe = Doe(n, bounds, 'halton', dists)
    sample = doe.generate()
    out = np.array([[37.5, 3862.709], [26.25, 4207.291], [48.75, 3546.744],
                    [20.625, 3979.116], [43.125, 4340.884]])
    npt.assert_almost_equal(sample, out, decimal=1)

    dists = ['Uniform(15., 60.)', 'Normal(4035., 400.)']
    doe = Doe(13, bounds, 'saltelli', dists)
    sample = doe.generate()
    assert len(sample) == 12

    doe = Doe(10, bounds, 'saltelli', dists)
    sample = doe.generate()
    assert len(sample) == 6


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_resampling(tmp, branin_data, settings_ishigami):
    f_2d = branin_data.func
    space = branin_data.space
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['snapshot']['parameters'] = ['x1', 'x2']
    space.empty()
    space.sampling(5, 'halton')
    space.max_points_nb = 100

    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, f_2d(space))

    # LOO tests on small set
    refiner = Refiner(surrogate, space.corners, delta_space=0.1)
    point_loo = refiner.points[1]
    refiner.leave_one_out_sigma(point_loo)
    refiner.leave_one_out_sobol(point_loo, ['Uniform(-5, 0)',
                                            'Uniform(10, 15)'])

    # Larger dataset to ensure stable results
    space.empty()
    space.sampling(11, 'halton')
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, f_2d(space))
    for _ in range(2):
        space.refine(surrogate, 'sigma')
        surrogate.fit(space, f_2d(space))
    assert len(space) == 13

    refiner = Refiner(surrogate, space.corners, delta_space=0.1)

    sigma = refiner.sigma()
    npt.assert_almost_equal(sigma, [8.47,  13.65], decimal=1)

    optim_EI = refiner.optimization(method='EI')
    npt.assert_almost_equal(optim_EI, [-1.387, 8.586], decimal=1)

    optim_PI = refiner.optimization(method='PI')
    npt.assert_almost_equal(optim_PI, [-1.985, 8.87], decimal=1)

    disc = refiner.discrepancy()
    npt.assert_almost_equal(disc, [8.47, 12.415], decimal=1)

    extrema = np.array(refiner.extrema([])[0])
    npt.assert_almost_equal(extrema, [[-2.694, 2.331], [2.219, 1.979]], decimal=1)

    base_sigma_disc = refiner.sigma_discrepancy()
    npt.assert_almost_equal(base_sigma_disc,
                            refiner.sigma_discrepancy([0.5, 0.5]), decimal=1)
    assert (base_sigma_disc != refiner.sigma_discrepancy([-0.1, 1.])).any()

    # Refiner without surrogate
    refiner = Refiner(surrogate, space.corners, delta_space=0.1)
    disc2 = refiner.discrepancy()
    npt.assert_almost_equal(disc2, [8.47, 12.416], decimal=1)

    # # Plotting
    # import os
    # import itertools
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # num = 25
    # x = np.linspace(-7, 10, num=num)
    # y = np.linspace(0, 15, num=num)
    # points = np.array([(float(i), float(j)) for i, j in itertools.product(x, y)])
    # x = points[:, 0].flatten()
    # y = points[:, 1].flatten()
    # pred, _ = surrogate(points)
    # pred = np.array(pred).flatten()
    # space = np.array(space[:])

    # color = True
    # c_map = cm.viridis if color else cm.gray
    # fig = plt.figure('Composite sigma/discrepancy')
    # plt.plot(space[:11, 0], space[:11, 1], 'ko', label='initial sample')
    # plt.plot(space[11:, 0], space[11:, 1], 'm^', label='firsts sigma')
    # plt.plot(-3.68928528, 13.62998774, 'rx')
    # plt.tricontourf(x, y, pred, antialiased=True, cmap=c_map)
    # plt.plot(sigma[0], sigma[1], '+', label='sigma')
    # plt.plot(optim_EI[0], optim_EI[1], 's', label='optimization EI')
    # plt.plot(optim_PI[0], optim_PI[1], 'p', label='optimization PI')
    # plt.plot(disc[0], disc[1], 'v', label='discrepancy')
    # plt.plot(disc[0], disc2[1], '-', label='discrepancy without surrogate')
    # plt.plot(extrema[:, 0], extrema[:, 1], '>', label='extrema')
    # plt.plot(base_sigma_disc[0], base_sigma_disc[1], '<', label='sigma+discrepancy')
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$')
    # plt.xlabel(r'$x_1$', fontsize=24)
    # plt.ylabel(r'$x_2$', fontsize=24)
    # plt.tick_params(axis='y')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')

    # plt.legend(fontsize=21, bbox_to_anchor=(1.3, 1), borderaxespad=0)
    # fig.tight_layout()
    # path = os.path.join(tmp, 'refinements.pdf')
    # fig.savefig(path, transparent=True, bbox_inches='tight')


def test_discrepancy():
    corners = [[0.5, 0.5], [6.5, 6.5]]
    space_1 = Space(corners)
    space_2 = Space(corners)

    space_1 += [[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]]
    space_2 += [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]]

    assert space_1.discrepancy() == pytest.approx(0.0081, abs=0.0001)
    assert space_2.discrepancy() == pytest.approx(0.0105, abs=0.0001)
    assert space_2.discrepancy(space_1) == pytest.approx(0.0081, abs=0.0001)
