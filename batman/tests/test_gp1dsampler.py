# coding: utf8
from mock import patch
import pytest
from distutils.version import LooseVersion
import numpy as np
import numpy.testing as npt
import openturns as ot
from batman.space.gp_1d_sampler import Gp1dSampler

ot_comp = LooseVersion(ot.__version__) < LooseVersion('1.10')


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
@patch("matplotlib.pyplot.show")
def test_Gp1dSampler(mock_show, tmp):
    sampler = Gp1dSampler(x=[[0.104], [1.]])

    print(sampler)

    # Plot of the modes of the Karhunen Loeve Decomposition
    sampler.plot_modes(tmp)

    # Sample of the GP1D and plot the instances
    size = 5
    Y = sampler.sample(size)
    sampler.plot_sample(Y, tmp)

    # Build a GP1D instance and plot the instances
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
    Y = sampler.build(coeff)
    sampler.plot_sample(Y, tmp)


@pytest.fixture(scope="session")
def sampler():
    return Gp1dSampler(t_ini=0, t_end=1, Nt=3, sigma=1.0, theta=0.5,
                       threshold=0.01, cov="AbsoluteExponential")


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
def test_Gp1dSampler_modes(sampler):
    sol = np.array([[5.98847892e-01, 6.57519854e-01, 4.57218596e-01],
                    [9.17674523e-01, -1.81049751e-16, -3.97332946e-01],
                    [5.98847892e-01, -6.57519854e-01, 4.57218596e-01]])
    npt.assert_almost_equal(sampler.modes, sol, decimal=2)


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
def test_Gp1dSampler_sample_values(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    Y = sampler.sample(size)
    sol = np.array([[-0.92131796, 0.19442651],
                    [1.42486738, -1.30101805],
                    [-0.34498127, -1.39082519]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
def test_Gp1dSampler_sample_coeff(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    Y = sampler.sample(size)
    sol = np.array([[0.60820165, -0.43826562, -2.18138523],
                    [-1.2661731, 1.2054782, 0.35004209]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
def test_Gp1dSampler_build_values(sampler):
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
    Y = sampler.build(coeff)
    sol = np.array([0.39714604, 0.34246808, -0.52338176])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


@pytest.mark.skipif(not ot_comp, reason='openturns version > 1.9')
def test_Gp1dSampler_build_coeff(sampler):
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0.4]
    Y = sampler.build(coeff)
    sol = [0.2, 0.7, -0.4]
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)
