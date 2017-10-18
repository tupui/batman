# coding: utf8
import copy
import pytest
import numpy as np
import numpy.testing as npt
from batman.space import (Point, Space, Doe,
                          UnicityError, AlienPointError, FullSpaceError)
from batman.functions import Ishigami
from batman.tasks import Snapshot
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
                  p_lst=['x', 'y', 'z'])

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
    s2 = space2.sampling(10, kind='lhsc')
    assert len(s2) == 10
    assert s1[:] != s2[:]

    space.empty()
    with pytest.raises(UnicityError):
        space += (1, 2, 3)
        space += (1, 2, 3)

    with pytest.raises(SystemExit):
        space += (1, 2)

    with pytest.raises(AlienPointError):
        space += (1, 7, 3)

    with pytest.raises(FullSpaceError):
        space.sampling(17)

    space = Space(corners, sample=np.array([(1, 2, 3), (1, 1, 3)]))
    assert space.doe_init == 2
    assert space.max_points_nb == 2

    corners[1] = [np.pi, -np.pi, np.pi]
    with pytest.raises(ValueError):
        space = Space(corners)


def test_space_evaluation(settings_ishigami):
    f_3d = Ishigami()
    space = Space(settings_ishigami['space']['corners'])
    space.sampling(2, 'halton')
    targets_space = f_3d(space)
    f_data_base = np.array([5.25, 4.2344145]).reshape(2, 1)
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
    doe = Doe(n, bounds, kind)
    sample = doe.generate()
    out = np.array([[5., 3.], [2.5, 4.], [7.5, 2.3], [1.25, 3.3], [6.25, 4.3]])
    npt.assert_almost_equal(sample, out, decimal=1)

    kind = 'sobolscramble'
    doe = Doe(n, bounds, kind, discrete_var)
    sample = doe.generate()

    kind = 'lhsopt'
    doe = Doe(n, bounds, kind)
    ot.RandomGenerator.SetSeed(123)
    sample = doe.generate()
    out = np.array([[8.097, 3.646], [6.592, 4.806], [0.622, 2.909],
                    [5.361, 2.162], [2.921, 4.041]])
    npt.assert_almost_equal(sample, out, decimal=1)

    bounds = np.array([[15.0, 2500.0], [60.0, 6000.0]])
    kind = ['Uniform(15., 60.)', 'Normal(4035., 400.)']
    doe = Doe(n, bounds, kind)
    sample = doe.generate()
    out = np.array([[37.5, 3862.709], [26.25, 4207.291], [48.75, 3546.744],
                    [20.625, 3979.116], [43.125, 4340.884]])
    npt.assert_almost_equal(sample, out, decimal=1)


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_resampling(tmp, branin_data, settings_ishigami):
    f_2d = branin_data.func
    space = branin_data.space
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['sampling']['init_size'] = 5
    test_settings['space']['sampling']['method'] = 'halton'
    test_settings['space']['resampling']['method'] = 'sigma'
    test_settings['space']['resampling']['resamp_size'] = 2
    test_settings['space']['resampling']['delta_space'] = 0.1
    test_settings['space']['corners'] = space.corners
    test_settings['uq']['test'] = 'Branin'
    test_settings['snapshot']['io']['parameter_names'] = ['x1', 'x2']
    test_settings['snapshot']['provider'] = f_2d
    space.empty()
    space.sampling(5, 'halton')

    Snapshot.initialize(test_settings['snapshot']['io'])
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, f_2d(space))

    # LOO tests on small set
    refiner = Refiner(surrogate, test_settings)
    point_loo = refiner.points[1]
    refiner.leave_one_out_sigma(point_loo)
    refiner.leave_one_out_sobol(point_loo)

    # Larger dataset to ensure stable results
    space.empty()
    space.sampling(11)
    surrogate = SurrogateModel('kriging', space.corners)
    surrogate.fit(space, f_2d(space))
    for _ in range(2):
        space.refine(surrogate)
        surrogate.fit(space, f_2d(space))
    assert len(space) == 13

    refiner = Refiner(surrogate, test_settings)

    sigma = refiner.sigma()
    optim_EI = refiner.optimization(method='EI')
    optim_PI = refiner.optimization(method='PI')
    disc = refiner.discrepancy()
    extrema = np.array(refiner.extrema([])[0])

    base_sigma_disc = refiner.sigma_discrepancy()
    npt.assert_almost_equal(base_sigma_disc,
                            refiner.sigma_discrepancy([0.5, 0.5]), decimal=2)
    assert (base_sigma_disc != refiner.sigma_discrepancy([-0.1, 1.])).any()

    # Refiner without surrogate
    refiner = Refiner(space, test_settings)
    disc2 = refiner.discrepancy()

    print(sigma, optim_EI, optim_PI, disc, extrema, disc2)
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


def test_discrepancy(settings_ishigami):
    corners = [[0.5, 0.5], [6.5, 6.5]]
    space_1 = Space(corners)
    space_2 = Space(corners)

    space_1 += [[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]]
    space_2 += [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]]

    assert space_1.discrepancy() == pytest.approx(0.0081, abs=0.0001)
    assert space_2.discrepancy() == pytest.approx(0.0105, abs=0.0001)
    assert space_2.discrepancy(space_1) == pytest.approx(0.0081, abs=0.0001)
