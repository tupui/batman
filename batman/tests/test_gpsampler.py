# coding: utf8
from mock import patch
import pytest
import numpy as np
import numpy.testing as npt
import openturns as ot
from batman.space.gp_sampler import GpSampler


@patch("matplotlib.pyplot.show")
def test_GpSampler(mock_show, tmp):
    n_nodes = 100
    reference = {'indices': [[x / float(n_nodes)] for x in range(n_nodes)],
                 'values': [0 for _ in range(n_nodes)]}
    sampler = GpSampler(reference)

    print(sampler)

    # Plot of the modes of the Karhunen Loeve Decomposition
    sampler.plot_modes(tmp)

    # Sample of the Gp and plot the instances
    size = 5
    Y = sampler(size)
    sampler.plot_sample(Y, tmp)

    # Build a Gp instance and plot the instances
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8]
    Y = sampler(coeff=coeff)
    sampler.plot_sample(Y, tmp)


@pytest.fixture(scope="session")
def sampler():
    n_nodes = 3
    reference = {'indices': [[x / float(n_nodes)] for x in range(n_nodes)],
                 'values': [0 for _ in range(n_nodes)]}
    return GpSampler(reference)


def test_GpSampler_modes(sampler):
    sol = np.array([[6.938e-01, 6.068e-01, 3.878e-01],
                    [9.363e-01, 1.440e-16, -3.512e-01],
                    [6.938e-01, -6.068e-01, 3.878e-01]])
    npt.assert_almost_equal(sampler.modes, sol, decimal=2)


def test_GpSampler_sample_values(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    Y = sampler(size)
    sol = np.array([[-1.999, -2.494, -0.407],
                    [-0.999, -1.152, -0.889]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_sample_coeff(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    Y = sampler(size)
    sol = np.array([[-2.29, -1.312, 0.996],
                    [-1.283, -0.091, -0.139]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)


def test_GpSampler_build_values(sampler):
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8]
    Y = sampler(coeff=coeff)
    sol = np.array([0.408,  0.328, -0.441])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_build_coeff(sampler):
    coeff = [0.2, 0.7, -0.4, 1.6, 0.2, 0.8]
    Y = sampler(coeff=coeff)
    sol = np.array([0.2,  0.7, -0.4])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)
