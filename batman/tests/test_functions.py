# coding: utf8
import pytest
from batman.functions import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                              Rastrigin, Ishigami, G_Function,
                              Forrester,  Manning, Mascaret)
from scipy.optimize import differential_evolution
import numpy as np
import numpy.testing as npt
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend('Agg')


def test_SixHumpCamel():
    f_2d = SixHumpCamel()
    assert f_2d([0.0898, -0.7126]) == pytest.approx(-1.0316, 0.01)
    assert f_2d([-0.0898, 0.7126]) == pytest.approx(-1.0316, 0.01)

    bounds = [[-3, 3], [-2, 2]]
    results = differential_evolution(f_2d, bounds, tol=0.001, popsize=20)
    f_obj = results.fun
    assert f_obj == pytest.approx(-1.0316, 0.05)


def test_Branin():
    f = Branin()
    assert f([2.20, 1.57]) == pytest.approx(17.76357802, 0.0001)

    bounds = [[-5, 10], [0, 15]]
    results = differential_evolution(f, bounds, tol=0.001, popsize=20)
    assert results.fun == pytest.approx(-16.64402157, 0.05)
    x_target = [-3.68928528, 13.62998774]
    npt.assert_almost_equal(results.x, x_target, decimal=2)


def test_Michalewicz():
    f_2d = Michalewicz()
    assert f_2d([2.20, 1.57]) == pytest.approx(-1.8013, 0.01)

    f_5d = Michalewicz(d=5)
    bounds = [[0., np.pi]] * 5
    results = differential_evolution(f_5d, bounds, tol=0.001, popsize=20)
    f_obj_5d = results.fun
    assert f_obj_5d == pytest.approx(-4.687, 0.05)


def test_Rosenbrock():
    f_2d = Rosenbrock()
    assert f_2d([1., 2.]) == 100.
    assert f_2d([1., 1]) == 0.

    f_3d = Rosenbrock(d=3)
    assert f_3d([1., 1., 1]) == 0.


def test_Rastrigin():
    f_2d = Rastrigin()
    assert f_2d([0., 0.]) == 0.


def test_Ishigami():
    f_3d = Ishigami()
    assert f_3d([2, -3, 1]) == pytest.approx(1.1396, 0.01)
    assert f_3d([0, 0, 0]) == 0.


def test_G_Function():
    f_6d = G_Function(d=6, a=np.array([78., 12., 0.5, 2., 97., 33.]))
    assert f_6d([0., 2./3., 1., 0., 0., 1./3.]) == pytest.approx(2.193, 0.01)

    f_5d = G_Function(d=5)
    npt.assert_almost_equal(f_5d.s_first,
                            [0.48, 0.21, 0.12, 0.08, 0.05],
                            decimal=2)


def test_Forrester():
    f_e = Forrester('e')
    f_c = Forrester('c')
    assert f_e([0.4]) == pytest.approx(0.11477697, 0.0001)
    assert f_c([0.4]) == pytest.approx(-5.94261151, 0.0001)
    assert f_e([0.6]) == pytest.approx(-0.14943781, 0.0001)
    assert f_c([0.6]) == pytest.approx(-4.0747189, 0.0001)


def test_Mascaret():
    f = Mascaret()
    f_out = f([31.54645246710516560, 4237.025232805773157])
    f_data_base = [2.747e1, 2.635e1, 2.5815e1, 2.5794e1, 2.4539e1, 2.2319e1,
                   2.132e1, 2.1313e1, 2.1336e1, 2.0952e1, 1.962e1, 1.8312e1,
                   1.7149e1, 1.446e1]
    npt.assert_almost_equal(f_out, f_data_base, decimal=2)


def test_Plot():
    f = Branin()

    num = 25
    x = np.linspace(-5, 10, num=num)
    y = np.linspace(0, 15, num=num)
    points = []
    for i, j in itertools.product(x, y):
        points += [(float(i), float(j))]
    pred = f(points)
    points = np.array(points)
    x = points[:, 0].flatten()
    y = points[:, 1].flatten()
    pred = np.array(pred).flatten()

    # Plotting
    color = True
    c_map = cm.viridis if color else cm.gray
    plt.figure("Expected Improvement")
    bounds = np.linspace(-17, 300., 30, endpoint=True)
    plt.tricontourf(x, y, pred, bounds,
                    antialiased=True, cmap=c_map)
    cbar = plt.colorbar()
    cbar.set_label('f', fontsize=28)
    plt.xlabel(r'$x_1$', fontsize=28)
    plt.ylabel(r'$x_2$', fontsize=28)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.legend(fontsize=26, loc='upper left')
    # plt.show()
