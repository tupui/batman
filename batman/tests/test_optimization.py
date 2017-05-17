# coding: utf8
import pytest
import copy
import numpy as np
from scipy.stats import norm
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from batman import Driver
from batman.functions import Branin


def test_optimization(tmp, branin_data, settings_ishigami):
    f_2d, dists, model, point, target_point, space, target_space = branin_data
    test_settings = copy.deepcopy(settings_ishigami)
    init_size = len(space)
    res_size = 0
    test_settings['space']['sampling']['init_size'] = init_size
    test_settings['space']['resampling']['method'] = 'optimization'
    test_settings['space']['resampling']['resamp_size'] = res_size
    test_settings["space"]["corners"] = space.corners
    test_settings["snapshot"]["io"]["parameter_names"] = ["x1", "x2"]
    f_obj = Branin()
    test_settings["snapshot"]["provider"] = f_obj

    driver = Driver(test_settings, tmp)
    driver.sampling()
    driver.resampling()

    num = 25
    x = np.linspace(-5, 10, num=num)
    x = np.linspace(-7, 10, num=num)
    y = np.linspace(0, 15, num=num)
    points = []
    for i, j in itertools.product(x, y):
        points += [(float(i), float(j))]
    pred, sigma = driver.prediction(points=points)
    points = np.array(points)

    x = points[:,0].flatten()
    y = points[:,1].flatten()
    pred = np.array(pred).flatten()
    sigma = np.array(sigma).flatten()
    space = np.array(driver.space[:])

    arg_min = np.argmin(target_space)
    min_value = target_space[arg_min]

    ei = np.empty_like(pred)
    for i, p, s in zip(range(num * num), pred, sigma):
        ei[i] = (min_value - p) * norm.cdf((min_value - p) / s)\
            + s * norm.pdf((min_value - p) / s)

    # Plotting
    color = True
    c_map = cm.viridis if color else cm.gray
    # fig = plt.figure("predictions")
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # bounds = np.linspace(-17, 300., 30, endpoint=True)
    # plt.tricontourf(x, y, pred, bounds,
    #                 antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$', fontsize=28)
    # plt.xlabel(r'$x_1$', fontsize=28)
    # plt.ylabel(r'$x_2$', fontsize=28)
    # plt.tick_params(axis='x', labelsize=26)
    # plt.tick_params(axis='y', labelsize=26)
    # plt.legend(fontsize=26, loc='upper left')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')
    # fig.tight_layout()
    # fig.savefig('pred.pdf', transparent=True, bbox_inches='tight')
    # plt.show()

    # fig = plt.figure("sigma")
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # plt.tricontourf(x, y, sigma,
    #                 antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\sigma(x_1, x_2)$', fontsize=28)
    # plt.xlabel(r'$x_1$', fontsize=28)
    # plt.ylabel(r'$x_2$', fontsize=28)
    # plt.tick_params(axis='x', labelsize=26)
    # plt.tick_params(axis='y', labelsize=26)
    # plt.legend(fontsize=26, loc='upper left')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')
    # fig.tight_layout()
    # fig.savefig('sigma.pdf', transparent=True, bbox_inches='tight')
    # plt.show()

    # fig = plt.figure("Expected Improvement")
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # plt.tricontourf(x, y, ei,
    #                 antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\sigma(x_1, x_2)$', fontsize=28)
    # plt.xlabel(r'$x_1$', fontsize=28)
    # plt.ylabel(r'$x_2$', fontsize=28)
    # plt.tick_params(axis='x', labelsize=26)
    # plt.tick_params(axis='y', labelsize=26)
    # plt.legend(fontsize=26, loc='upper left')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')
    # fig.tight_layout()
    # fig.savefig('expected_improvement.pdf',
    #             transparent=True, bbox_inches='tight')
    # plt.show()

    fig = plt.figure('Efficient Global Optimization', figsize=(20,5))
    plt.subplot(131)
    plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    plt.plot(-3.68928528, 13.62998774, 'r<')
    bounds = np.linspace(-17, 300., 30, endpoint=True)
    plt.tricontourf(x, y, pred, bounds, antialiased=True, cmap=c_map)
    cbar = plt.colorbar()
    cbar.set_label(r'$f(x_1, x_2)$')
    plt.ylabel(r'$x_2$', fontsize=24)
    plt.tick_params(axis='y')
    for txt, point in enumerate(space):
        plt.annotate(txt, point, textcoords='offset points')

    plt.subplot(132)
    plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    plt.plot(-3.68928528, 13.62998774, 'r<')
    plt.tricontourf(x, y, sigma, antialiased=True, cmap=c_map)
    cbar = plt.colorbar()
    cbar.set_label(r'$\sigma(x_1, x_2)$')
    plt.xlabel(r'$x_1$', fontsize=24)
    for txt, point in enumerate(space):
        plt.annotate(txt, point, textcoords='offset points')

    plt.subplot(133)
    plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    plt.plot(-3.68928528, 13.62998774, 'r<')
    plt.tricontourf(x, y, ei, antialiased=True, cmap=c_map)
    cbar = plt.colorbar()
    cbar.set_label(r'$\mathbb{E}[I(x_1, x_2)]$')
    for txt, point in enumerate(space):
        plt.annotate(txt, point, textcoords='offset points')

    fig.tight_layout()
    path = 'expected_improvement_' + str(res_size) + '.pdf'
    fig.savefig(path, transparent=True, bbox_inches='tight')
    # plt.show()
