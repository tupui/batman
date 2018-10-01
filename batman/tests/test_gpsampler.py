# coding: utf8
import os
from mock import patch
import pytest
import numpy as np
import numpy.testing as npt
from batman.space.gp_sampler import GpSampler
import sys


# a simple class with a write method
class WritableObject:
    def __init__(self):
        self.content = []

    def write(self, string):
        self.content.append(string)


@patch("matplotlib.pyplot.show")
def test_GpSampler1D(mock_show, tmp):
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
    sampler.plot_sample(Y, os.path.join(tmp, 'gp_instances.pdf'))

    # Build a Gp instance and plot the instances
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    sampler.plot_sample(Y)


@patch("matplotlib.pyplot.show")
def test_GpSampler2D(mock_show, tmp):
    n_nodes_by_dim = 10
    n_nodes = n_nodes_by_dim**2
    reference = {'indices': [[x / float(n_nodes_by_dim),
                              y / float(n_nodes_by_dim)]
                             for x in range(n_nodes_by_dim)
                             for y in range(n_nodes_by_dim)],
                 'values': [0. for x in range(n_nodes)]}
    sampler = GpSampler(reference, "Matern([0.5, 0.5], nu=0.5)")

    print(sampler)

    # Plot of the modes of the Karhunen Loeve Decomposition
    sampler.plot_modes(os.path.join(tmp, 'gp_modes.pdf'))

    # Sample of the Gp and plot the instances
    size = 5
    Y = sampler(size)
    sampler.plot_sample(Y, os.path.join(tmp, 'gp_instance.pdf'))

    # Build a Gp instance and plot the instances
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    sampler.plot_sample(Y, os.path.join(tmp, 'gp_instance.pdf'))


@patch("matplotlib.pyplot.show")
def test_GpSampler3D(mock_show, tmp):
    n_nodes_by_dim = 10
    n_nodes = n_nodes_by_dim**3
    reference = {'indices': [[x / float(n_nodes_by_dim),
                              y / float(n_nodes_by_dim),
                              z / float(n_nodes_by_dim)]
                             for x in range(n_nodes_by_dim)
                             for y in range(n_nodes_by_dim)
                             for z in range(n_nodes_by_dim)],
                 'values': [0. for x in range(n_nodes)]}
    sampler = GpSampler(reference, "Matern([0.5, 0.5, 0.5], nu=0.5)")

    print(sampler)

    # Sample of the Gp and plot the instances
    size = 5
    Y = sampler(size)

    # Build a Gp instance and plot the instances
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)


def sampler1D_from_file(tmp):
    n_nodes = 3
    reference = {'indices': [[x / float(n_nodes)] for x in range(n_nodes)],
                 'values': [0. for _ in range(n_nodes)]}
    reference_filename = os.path.join(tmp, 'reference_file.npy')
    np.save(reference_filename, reference)
    sampler = GpSampler(reference_filename)
    printOutput = WritableObject()
    old = sys.stdout
    sys.stdout = printOutput
    print(sampler)
    sys.stdout = old
    sol = ['Gp sampler summary:\n'
           '- Dimension = 1\n'
           '- Kernel = Matern(0.5, nu=0.5)\n'
           '- Standard deviation = 1.0\n'
           '- Mesh size = 3\n'
           '- Threshold for the KLd = 0.01\n'
           '- Number of modes = 3', '\n']
    npt.assert_array_equal(printOutput.content, sol)


@pytest.fixture(scope="function", params=[1, 2, 3])
def sampler(request, seed):
    if request.param == 1:
        n_nodes = 3
        reference = {'indices': [[x / float(n_nodes)] for x in range(n_nodes)],
                     'values': [0. for _ in range(n_nodes)]}
        gpsampler = GpSampler(reference)
    elif request.param == 2:
        n_nodes_by_dim = 2
        n_nodes = n_nodes_by_dim**2
        reference = {'indices': [[x / float(n_nodes_by_dim),
                                  y / float(n_nodes_by_dim)]
                                 for x in range(n_nodes_by_dim)
                                 for y in range(n_nodes_by_dim)],
                     'values': [0. for x in range(n_nodes)]}
        gpsampler = GpSampler(reference, "Matern([0.5, 0.5], nu=0.5)")
    elif request.param == 3:
        n_nodes_by_dim = 2
        n_nodes = n_nodes_by_dim**3
        reference = {'indices': [[x / float(n_nodes_by_dim),
                                  y / float(n_nodes_by_dim),
                                  z / float(n_nodes_by_dim)]
                                 for x in range(n_nodes_by_dim)
                                 for y in range(n_nodes_by_dim)
                                 for z in range(n_nodes_by_dim)],
                     'values': [0. for x in range(n_nodes)]}
        gpsampler = GpSampler(reference, "Matern([0.5, 0.5, 0.5], nu=0.5)")
    return gpsampler


def test_GpSampler_print(sampler):
    printOutput = WritableObject()
    old = sys.stdout
    sys.stdout = printOutput
    print(sampler)
    sys.stdout = old
    if sampler.n_dim == 1:
        sol = ['Gp sampler summary:\n'
               '- Dimension = 1\n'
               '- Kernel = Matern(0.5, nu=0.5)\n'
               '- Standard deviation = 1.0\n'
               '- Mesh size = 3\n'
               '- Threshold for the KLd = 0.01\n'
               '- Number of modes = 3', '\n']
    elif sampler.n_dim == 2:
        sol = ['Gp sampler summary:\n'
               '- Dimension = 2\n'
               '- Kernel = Matern([0.5, 0.5], nu=0.5)\n'
               '- Standard deviation = 1.0\n'
               '- Mesh size = 4\n'
               '- Threshold for the KLd = 0.01\n'
               '- Number of modes = 4', '\n']
    elif sampler.n_dim == 3:
        sol = ['Gp sampler summary:\n'
               '- Dimension = 3\n'
               '- Kernel = Matern([0.5, 0.5, 0.5], nu=0.5)\n'
               '- Standard deviation = 1.0\n'
               '- Mesh size = 8\n'
               '- Threshold for the KLd = 0.01\n'
               '- Number of modes = 8', '\n']
    npt.assert_array_equal(printOutput.content, sol)


def test_GpSampler_std(sampler):
    if sampler.n_dim == 1:
        sol = np.array([0.788, 0.497, 0.363])
    elif sampler.n_dim == 2:
        sol = np.array([0.699, 0.439, 0.433, 0.356])
    elif sampler.n_dim == 3:
        sol = np.array([0.609, 0.35, 0.344, 0.339, 0.266, 0.266, 0.264, 0.237])
    npt.assert_almost_equal(sampler.standard_deviation, sol, decimal=2)


def test_GpSampler_modes(sampler):
    if sampler.n_dim == 1:
        sol = np.array([[-0.548, -0.639, -0.54],
                        [0.708, -0.011, -0.706],
                        [-0.445, 0.769, -0.459]])
    elif sampler.n_dim == 2:
        sol = np.array([[-0.505, -0.496, -0.496, -0.503],
                        [-0.037, 0.705, -0.707, 0.038],
                        [-0.723, -0.018, 0.057, 0.688],
                        [0.47, -0.506, -0.501, 0.522]])
    elif sampler.n_dim == 3:
        sol = np.array([[-0.35, -0.365, -0.359, -0.361, -0.354, -0.353, -0.336, -0.349],
                        [0.253, 0.516, 0.111, 0.392, -0.386, -0.169, -0.508, -0.261],
                        [0.544, -0.097, 0.306, -0.342, 0.317, -0.295, 0.107, -0.534],
                        [-0.134, -0.334, 0.505, 0.351, -0.32, -0.519, 0.313, 0.148],
                        [-0.527, 0.458, -0.047, 0.104, 0.117, -0.069, 0.469, -0.511],
                        [-0.294, -0.208, 0.613, -0.073, -0.091, 0.577, -0.255, -0.288],
                        [-0.173, -0.294, -0.096, 0.539, 0.644, -0.148, -0.362, -0.132],
                        [0.325, -0.374, -0.35, 0.412, -0.294, 0.362, 0.321, -0.376]])
    npt.assert_almost_equal(sampler.modes, sol, decimal=2)


def test_GpSampler_sample_values_sobol(sampler):
    size = 2
    Y = sampler(size, kind="sobol")
    if sampler.n_dim == 1:
        sol = np.array([[0., 0., 0.],
                        [-0.638, -0.148, -0.163]])
    elif sampler.n_dim == 2:
        sol = np.array([[0., 0., 0., 0.],
                        [-0.551, -0.327, 0.112, -0.173]])
    elif sampler.n_dim == 3:
        sol = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                        [-0.171, -0.09, -0.298, -0.339, 0.292, -0.255, -0.05, -0.243]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_sample_coeff_sobol(sampler):
    size = 2
    Y = sampler(size, kind="sobol")
    if sampler.n_dim == 1:
        sol = np.array([[0., 0., 0.],
                        [0.674, -0.674, 0.674]])
    elif sampler.n_dim == 2:
        sol = np.array([[0., 0., 0., 0.],
                        [0.674, -0.674, 0.674, -0.674]])
    elif sampler.n_dim == 3:
        sol = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                        [0.674, -0.674, 0.674, -0.674, 0.674, -0.674, 0.674, -0.674]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)


def test_GpSampler_sample_values(sampler, seed):
    size = 2
    Y = sampler(size)
    if sampler.n_dim == 1:
        sol = np.array([[-0.48, 0.16, 0.59],
                        [0.29, -0.31, -0.43]])
    elif sampler.n_dim == 2:
        sol = np.array([[-0.51, -0.09, -0.05, -0.21],
                        [0.13, 0.65, 0.89, 0.25]])
    elif sampler.n_dim == 3:
        sol = np.array([[-0.37, -0.62, -0.36, -0.54, -0.19, 0.2, 0.17, 0.22],
                        [-0.2, 0.34, 0.76, 0.42, 0.21, 0.01, 0.15, 0.54]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_sample_coeff(sampler, seed):
    size = 2
    Y = sampler(size)
    if sampler.n_dim == 1:
        sol = np.array([[-0.2, -1.53, 0.18],
                        [0.34, 1.03, -0.47]])
    elif sampler.n_dim == 2:
        sol = np.array([[0.62, -0.05, 0.51, -0.77],
                        [-1.37, -0.38, 0.28, -1.65]])
    elif sampler.n_dim == 3:
        sol = np.array([[0.89, -2.21, -0.83, -0.2, -0.72, 0.33, -0.97, 0.48],
                        [-1.3, 0.2, -0.76, 1.48, 0.32, 0.8, 0.35, -2.09]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)


def test_GpSampler_build_values(sampler):
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    if sampler.n_dim == 1:
        sol = np.array([[0.225, -0.216, -0.264]])
    elif sampler.n_dim == 2:
        sol = np.array([[0.311, -0.138, -0.582, 0.119]])
    elif sampler.n_dim == 3:
        sol = np.array([[-0.22, -0.11, 0.34, 0.28, -0.37, -0.21, -0.04, -0.04]])
    npt.assert_almost_equal(Y['Values'], sol, decimal=2)


def test_GpSampler_build_coeff(sampler):
    coeff = [[0.2, 0.7, -0.4, 1.6, 0.2, 0.8]]
    Y = sampler(coeff=coeff)
    if sampler.n_dim == 1:
        sol = np.array([[0.2, 0.7, -0.4]])
    elif sampler.n_dim == 2:
        sol = np.array([[0.2, 0.7, -0.4, 1.6]])
    elif sampler.n_dim == 3:
        sol = np.array([[0.2, 0.7, -0.4, 1.6, 0.2, 0.8, 0., 0.]])
    npt.assert_almost_equal(Y['Coefficients'], sol, decimal=2)
