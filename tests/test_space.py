# coding: utf8
import copy
import os
import pytest
import numpy as np
import numpy.testing as npt
import openturns as ot
import matplotlib.pyplot as plt
from batman.space import (Space, Doe, dists_to_ot)
from batman.functions import Ishigami
from batman.surrogate import SurrogateModel
from batman.space.refiner import Refiner


def test_dists_to_ot():
    dists = dists_to_ot(['Uniform(12, 15)', 'Normal(400, 10)'])
    out = [ot.Uniform(12, 15), ot.Normal(400, 10)]
    assert dists == out

    with pytest.raises(AttributeError):
        dists_to_ot(['Uniorm(12, 15)'])


def test_space(settings_ishigami, seed):
    corners = settings_ishigami['space']['corners']
    space = Space(corners)
    assert space.max_points_nb == np.inf

    space = Space(corners, sample=10)
    assert space.max_points_nb == 10

    space = Space(corners, sample=10, nrefine=6,
                  plabels=['x', 'y', 'z'])

    assert space.max_points_nb == 16

    space += (1, 2, 3)
    npt.assert_array_equal(space.values, [(1, 2, 3)])

    space.empty()
    npt.assert_array_equal(space.values, np.empty((0, 3)))

    space += [(1, 2, 3), (1, 1, 3)]
    npt.assert_array_equal(space.values, [(1, 2, 3), (1, 1, 3)])

    space2 = Space(corners, space.values)
    npt.assert_array_equal(space2.values, [(1, 2, 3), (1, 1, 3)])

    s1 = space.sampling()
    assert len(s1) == 10
    space2 = Space(corners,
                   sample=settings_ishigami['space']['sampling']['init_size'],
                   nrefine=settings_ishigami['space']['resampling']['resamp_size'])

    s2 = space2.sampling(10, kind='lhsc')
    assert len(s2) == 10
    assert np.any(s1 != s2)

    space.empty()
    space += (1, 2, 3)
    space += (1, 2, 3)
    assert len(space) == 1

    space = Space(corners, sample=16, duplicate=True)
    space += (1, 2, 3)
    space += (1, 2, 3)
    assert len(space) == 2

    with pytest.raises(ValueError):
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


def test_doe(seed):
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

    doe = Doe(n, bounds, 'olhs')
    sample = doe.generate()
    out = [[6.149, 2.343], [9.519, 3.497], [1.991, 4.058],
           [5.865, 4.995], [2.551, 2.737]]
    npt.assert_almost_equal(sample, out, decimal=1)

    bounds = [[15.0, 2500.0], [60.0, 6000.0]]

    with pytest.raises(AttributeError):
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
    assert (len(sample) == 12) or (len(sample) == 8)

    doe = Doe(10, bounds, 'saltelli', dists)
    sample = doe.generate()
    assert (len(sample) == 6) or (len(sample) == 4)


def plot_hypercube(hypercube):
    """Plot an hypercube.

    :param array_like hypercube ([min, n_features], [max, n_features]).
    """
    hypercube = hypercube.T
    plt.plot([hypercube[0, 0], hypercube[0, 0],
              hypercube[0, 0], hypercube[1, 0],
              hypercube[1, 0], hypercube[1, 0],
              hypercube[0, 0], hypercube[1, 0]],
             [hypercube[0, 1], hypercube[1, 1],
              hypercube[1, 1], hypercube[1, 1],
              hypercube[1, 1], hypercube[0, 1],
              hypercube[0, 1], hypercube[0, 1]])


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_refiner_basics(tmp, branin_data, settings_ishigami, seed):
    f_2d = branin_data.func
    space = branin_data.space
    space.sampling(11, 'halton')
    surrogate = SurrogateModel('kriging', space.corners, space.plabels)
    surrogate.fit(space, f_2d(space))

    refiner = Refiner(surrogate, space.corners, delta_space=0.08)

    distance_min = refiner.distance_min(refiner.space.values[0])
    assert distance_min == pytest.approx(0.163461, abs=0.001)

    hypercube = refiner.hypercube_distance(refiner.space.values[0], distance_min)
    npt.assert_almost_equal(hypercube, [[-0.62, 3.62], [3.04, 6.96]], decimal=2)

    hypercube_optim = refiner.hypercube_optim(refiner.space.values[0])
    npt.assert_almost_equal(hypercube_optim,
                            [[-0.61, 5.74], [1.0, 11.66]], decimal=2)

    # Plotting
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

    # plt.figure()
    # plt.tricontourf(x, y, pred, antialiased=True, cmap=cm.viridis)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$')
    # plt.scatter(space[:11, 0], space[:11, 1], label='initial sample')

    # # plt.scatter(space[4, 0], space[4, 1], label='Anchor point')
    # plot_hypercube(refiner.corners)
    # plot_hypercube(hypercube)
    # plot_hypercube(hypercube_optim)

    # plt.xlabel(r'$x_1$', fontsize=24)
    # plt.ylabel(r'$x_2$', fontsize=24)
    # plt.tick_params(axis='y')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, xycoords='offset points')

    # plt.legend(fontsize=21, bbox_to_anchor=(1.3, 1), borderaxespad=0)
    # plt.show()


@pytest.mark.xfail(raises=AssertionError, reason='Global optimization')
def test_resampling(tmp, branin_data, settings_ishigami, seed):
    f_2d = branin_data.func
    space = branin_data.space
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['snapshot']['plabels'] = ['x1', 'x2']
    space.empty()
    max_points_nb = 5
    space.sampling(max_points_nb, 'halton')
    space.max_points_nb = 100

    surrogate = SurrogateModel('kriging', space.corners, space.plabels)
    surrogate.fit(space, f_2d(space))

    # Larger dataset to ensure stable results
    space.empty()
    max_points_nb = 11
    space.sampling(max_points_nb, 'halton')
    surrogate = SurrogateModel('kriging', space.corners, space.plabels)
    surrogate.fit(space, f_2d(space))
    for _ in range(2):
        space.refine(surrogate, 'sigma')
        surrogate.fit(space, f_2d(space))
    assert len(space) == 13

    refiner = Refiner(surrogate, space.corners, delta_space=0.15)

    point_loo = refiner.space.values[5]
    loo_si = refiner.leave_one_out_sigma(point_loo)
    npt.assert_almost_equal(loo_si, [-2.76,  2.], decimal=2)

    loo_so = refiner.leave_one_out_sobol(point_loo, ['Uniform(-5, 0)',
                                                     'Uniform(10, 15)'])
    npt.assert_almost_equal(loo_so, [-2.86,  2.28], decimal=2)

    sigma = refiner.sigma()
    npt.assert_almost_equal(sigma, [4.85,  6.561], decimal=1)

    optim_EI_min = refiner.optimization(method='EI')
    npt.assert_almost_equal(optim_EI_min, [-2.176, 9.208], decimal=1)

    optim_EI_max = refiner.optimization(extremum='max')
    npt.assert_almost_equal(optim_EI_max, [6.59,  12.999], decimal=1)

    optim_PI = refiner.optimization(method='PI')
    npt.assert_almost_equal(optim_PI, [-2.328, 9.441], decimal=1)

    disc = refiner.discrepancy()
    npt.assert_almost_equal(disc, [7, 13.], decimal=1)

    extrema = np.array(refiner.extrema([])[0])
    # npt.assert_almost_equal(extrema, [[-2.694, 2.331], [2.576, 2.242]], decimal=1)

    base_sigma_disc = refiner.sigma_discrepancy()
    npt.assert_almost_equal(base_sigma_disc,
                            refiner.sigma_discrepancy([0.5, 0.5]), decimal=1)
    assert (base_sigma_disc != refiner.sigma_discrepancy([-0.1, 1.])).any()

    # Refiner without surrogate
    refiner = Refiner(surrogate, space.corners, delta_space=0.1)
    disc2 = refiner.discrepancy()
    npt.assert_almost_equal(disc2, [8., 13.], decimal=1)

    # Plotting
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
    # pred, si = surrogate(points)
    # si = np.array(si).flatten()
    # pred = np.array(pred).flatten()
    # space = np.array(space[:])

    # plt.figure()
    # plt.tricontourf(x, y, si, antialiased=True, cmap=cm.viridis)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$')
    # plt.show()

    # plt.figure()
    # plt.tricontourf(x, y, pred, antialiased=True, cmap=cm.viridis)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$')
    # plt.scatter(space[:11, 0], space[:11, 1], label='initial sample')
    # plt.scatter(space[11:, 0], space[11:, 1], label='firsts sigma')
    # plt.scatter(-3.68928528, 13.62998774, label='global extrema')

    # hypercube_optim = refiner.hypercube_optim(refiner.space.values[5])
    # plot_hypercube(hypercube_optim)
    # plot_hypercube(refiner.corners)
    # plt.scatter(loo_so[0], loo_so[1], label='LOO-sigma')
    # plt.scatter(loo_si[0], loo_si[1], label='LOO-sobol')

    # plt.scatter(sigma[0], sigma[1], label='sigma')
    # plt.scatter(optim_EI_min[0], optim_EI_min[1], label='optimization EI min')
    # plt.scatter(optim_EI_max[0], optim_EI_max[1], label='optimization EI max')
    # plt.scatter(optim_PI[0], optim_PI[1], label='optimization PI')
    # plt.scatter(disc[0], disc[1], label='discrepancy')
    # plt.scatter(disc2[0], disc2[1], label='discrepancy without surrogate')
    # # plt.scatter(extrema[:, 0], extrema[:, 1], label='extrema')
    # plt.scatter(base_sigma_disc[0], base_sigma_disc[1], label='sigma+discrepancy')

    # plt.xlabel(r'$x_1$', fontsize=24)
    # plt.ylabel(r'$x_2$', fontsize=24)
    # plt.tick_params(axis='y')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, xycoords='offset points')

    # plt.legend(fontsize=21, bbox_to_anchor=(1.3, 1), borderaxespad=0)
    # plt.show()


def test_discrepancy():
    corners = [[0.5, 0.5], [6.5, 6.5]]
    space_1 = Space(corners)
    space_2 = Space(corners)

    space_1 += [[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]]
    space_2 += [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [6, 6]]

    assert Space.discrepancy(space_1, space_1.corners) == pytest.approx(0.0081, abs=1e-4)
    assert Space.discrepancy(space_2, space_2.corners) == pytest.approx(0.0105, abs=1e-4)

    space_1 = (2.0 * space_1.values - 1.0) / (2.0 * 6.0)
    assert Space.discrepancy(space_1) == pytest.approx(0.0081, abs=1e-4)

    space = np.array([[2, 1, 1, 2, 2, 2],
                      [1, 2, 2, 2, 2, 2],
                      [2, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 2, 2],
                      [1, 2, 2, 2, 1, 1],
                      [2, 2, 2, 2, 1, 1],
                      [2, 2, 2, 1, 2, 2]])
    space = (2.0 * space - 1.0) / (2.0 * 2.0)

    assert Space.discrepancy(space, method='MD') == pytest.approx(2.5000, abs=1e-4)
    assert Space.discrepancy(space, method='WD') == pytest.approx(1.3680, abs=1e-4)
    assert Space.discrepancy(space, method='CD') == pytest.approx(0.3172, abs=1e-4)


def test_mst(tmp):
    sample = np.array([[0.25, 0.5], [0.6, 0.4], [0.7, 0.2]])
    mean, std, edges = Space.mst(sample, fname=os.path.join(tmp, 'mst.pdf'))

    assert mean == pytest.approx(0.2938, abs=1e-4)
    assert std == pytest.approx(0.0702, abs=1e-4)
    npt.assert_equal(edges, [[0, 1], [1, 2]])
