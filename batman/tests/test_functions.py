# coding: utf8
import os
import pytest
from batman.functions import (SixHumpCamel, Branin, Michalewicz, Rosenbrock,
                              Rastrigin, Ishigami, G_Function, Forrester,
                              ChemicalSpill, Manning, db_Mascaret, mascaret,
                              Channel_Flow, DbGeneric)
from scipy.optimize import differential_evolution
import numpy as np
import numpy.testing as npt


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


def test_Michalewicz(seed):
    f_2d = Michalewicz()
    assert f_2d([2.20, 1.57]) == pytest.approx(-1.8013, 0.01)

    f_5d = Michalewicz(d=5)
    bounds = [[0., np.pi]] * 5
    results = differential_evolution(f_5d, bounds, tol=0.001, popsize=20)
    f_obj_5d = results.fun
    assert f_obj_5d == pytest.approx(-4.687, 0.1)


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


def test_Manning():
    f = Manning()
    assert f(25) == pytest.approx(5.64345405, 0.0001)
    f_2d = Manning(d=2)
    assert f_2d([25, 1200]) == pytest.approx(6.29584085, 0.0001)


def test_Channel_Flow():
    f = Channel_Flow(dx=8000., length=40000., width=170., slope=2.8e-4, hinit=6.917)
    sample = [[11, 2000], [40, 4000]]

    results = f(sample, h_nc=True)
    npt.assert_almost_equal(results,
                            [[12.65, 11.82, 11.21, 10.78, -4.28, 2.42, 12.12],
                             [6.23, 4.01, 2.02, 0.55, -4.28, 3.84, 8.46]],
                            decimal=2)


def test_Mascaret():
    f = db_Mascaret()
    f_out = f([31.54645246710516560, 4237.025232805773157])
    f_data_base = [[2.747e1, 2.635e1, 2.5815e1, 2.5794e1, 2.4539e1, 2.2319e1,
                    2.132e1, 2.1313e1, 2.1336e1, 2.0952e1, 1.962e1, 1.8312e1,
                    1.7149e1, 1.446e1]]
    npt.assert_almost_equal(f_out, f_data_base, decimal=2)

    f = db_Mascaret(multizone=True)
    f_out = f([51.5625, 46.66, 27.6, 4135.007205626885])[0, :6]
    f_data_base = [26.39, 26.36, 26.35, 26.34, 26.33, 26.32]
    npt.assert_almost_equal(f_out, f_data_base, decimal=2)


def test_ChemicalSpill():
    f = ChemicalSpill()
    y = f([10, 0.07, 1.505, 30.1525])
    assert y.shape == (1, 1000)


def test_DbGeneric():
    # From samples
    f_ = mascaret()
    f = DbGeneric(space=f_.space, data=f_.data)

    s_out, f_out = f([31.54645246710516560, 4237.025232805773157], full=True)
    f_data_base = [[2.747e1, 2.635e1, 2.5815e1, 2.5794e1, 2.4539e1, 2.2319e1,
                    2.132e1, 2.1313e1, 2.1336e1, 2.0952e1, 1.962e1, 1.8312e1,
                    1.7149e1, 1.446e1]]
    npt.assert_almost_equal(f_out, f_data_base, decimal=2)
    s_data_base = [[31.54645246710516560, 4237.025232805773157]]
    npt.assert_almost_equal(s_out, s_data_base, decimal=2)

    # From paths
    PATH = os.path.dirname(os.path.realpath(__file__))
    fnames = ['input_mascaret.npy', 'output_mascaret.npy']
    fnames = [os.path.join(PATH, '../functions/data/', p) for p in fnames]

    f = DbGeneric(fnames=fnames)

    f_out = f([31.54645246710516560, 4237.025232805773157])
    f_data_base = [[2.747e1, 2.635e1, 2.5815e1, 2.5794e1, 2.4539e1, 2.2319e1,
                    2.132e1, 2.1313e1, 2.1336e1, 2.0952e1, 1.962e1, 1.8312e1,
                    1.7149e1, 1.446e1]]
    npt.assert_almost_equal(f_out, f_data_base, decimal=2)
