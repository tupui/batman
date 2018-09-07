# coding: utf8
import os
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
    sampler.plot_modes(os.path.join(tmp, 'gp_modes.pdf'))

    # Sample of the Gp and plot the instances
    size = 5
    Y = sampler(size)
    sampler.plot_sample(Y, os.path.join(tmp, 'gp_samples.pdf'))

    # Build a Gp instance and plot the instances
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    sampler.plot_sample(Y)


@pytest.fixture(scope="session")
def sampler():
    n_nodes = 3
    reference = {'indices': [[x / float(n_nodes)] for x in range(n_nodes)],
                 'values': [0 for _ in range(n_nodes)]}
    return GpSampler(reference)


def test_GpSampler_modes(sampler):
    sol = np.array([[0.531, 0.635, 0.562],
                    [-0.729, 0.004, 0.684],
                    [0.432, -0.773, 0.465]])
    npt.assert_almost_equal(sampler.modes, sol, decimal=2)


def test_GpSampler_sample_values(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    Y = sampler(size)
    sol = np.array([[0.742, 0.608, 1.076],
                    [0.11, 1.39, 1.455]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_sample_coeff(sampler):
    size = 2
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    Y = sampler(size)
    sol = np.array([[1.764, 0.4, 0.979],
                    [2.241, 1.868, -0.977]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)


def test_GpSampler_build_values(sampler):
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    sol = np.array([[-0.23, 0.212, 0.258]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_build_coeff(sampler):
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    sol = np.array([[0.2,  0.7, -0.4]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)
