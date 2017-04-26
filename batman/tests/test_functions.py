# coding: utf8
import pytest
from batman.functions import (Michalewicz, Rosenbrock, Ishigami, G_Function,
                              Forrester, Manning, Mascaret)
from scipy.optimize import differential_evolution
import numpy as np
import numpy.testing as npt


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


def test_Ishigami():
    f_3d = Ishigami()
    assert f_3d([2, -3, 1]) == pytest.approx(1.1396, 0.01)
    assert f_3d([0, 0, 0]) == 0.


def test_G_Function():
    f_6d = G_Function(d=6, a=np.array([78., 12., 0.5, 2., 97., 33.]))
    assert f_6d([0., 2./3., 1., 0., 0., 1./3.]) == pytest.approx(2.193, 0.01)

    f_5d = G_Function(d=5)
    test_indices = npt.assert_almost_equal(f_5d.s_first,
                                           [0.48, 0.21, 0.12, 0.08, 0.05],
                                           decimal=2)
    assert True if test_indices is None else False


def test_Forrester():
    f_e = Forrester('e')
    f_c = Forrester('c')
    assert f_e([0.4]) == pytest.approx(0.11477697, 0.0001)
    assert f_c([0.4]) == pytest.approx(-4.85223025, 0.0001)
    assert f_e([0.6]) == pytest.approx(-0.14943781, 0.0001)
    assert f_c([0.6]) == pytest.approx(-5.49437807, 0.0001)


def test_Mascaret():
    f = Mascaret()
    f_out = f([31.54645246710516560, 4237.025232805773157])
    f_data_base = [2.747e1, 2.635e1, 2.5815e1, 2.5794e1, 2.4539e1, 2.2319e1,
                   2.132e1, 2.1313e1, 2.1336e1, 2.0952e1, 1.962e1, 1.8312e1,
                   1.7149e1, 1.446e1]
    test_output = npt.assert_almost_equal(f_out, f_data_base, decimal=2)
    assert True if test_output is None else False


def test_Manning():
    f = Manning()
    assert f([20]) == pytest.approx(6.45195012, 0.01)
    f2 = Manning(flag='2D')
    assert f2([20, 3000]) == pytest.approx(12.47279413, 0.01)
