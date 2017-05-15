# coding: utf8
import pytest
import copy
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
from batman import Driver
from batman.functions import Branin


def test_optimization(tmp, branin_data, settings_ishigami):
    f_2d, dists, model, point, target_point, space, target_space = branin_data
    test_settings = copy.deepcopy(settings_ishigami)
    test_settings['space']['sampling']['init_size'] = len(space)
    test_settings['space']['resampling']['method'] = 'optimization'
    test_settings['space']['resampling']['resamp_size'] = 5
    test_settings["space"]["corners"] = space.corners
    test_settings["snapshot"]["io"]["parameter_names"] = ["x1", "x2"]
    f_obj = Branin()
    test_settings["snapshot"]["provider"] = f_obj

    driver = Driver(test_settings, tmp)
    driver.sampling()
    driver.resampling()

    # [ -3.68928528  13.62998774]

    num = 25
    x = np.linspace(-5, 10, num=num)
    y = np.linspace(0, 15, num=num)
    points = []
    for i, j in itertools.product(x, y):
        points += [(float(i), float(j))]
    pred, _ = driver.prediction(points=points)
    points = np.array(points)

    x = points[:,0].flatten()
    y = points[:,1].flatten()
    pred = np.array(pred).flatten()
    space = np.array(driver.space[:])

    # Plotting
    color = True
    c_map = cm.viridis if color else cm.gray
    plt.figure("Expected Improvement")
    plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    plt.plot(space[init_size:, 0], space[init_size:, 1], 'r^')
    bounds = np.linspace(-17, 300., 30, endpoint=True)
    plt.tricontourf(x, y, pred, bounds,
                    antialiased=True, cmap=c_map)
    cbar = plt.colorbar()
    cbar.set_label('f', fontsize=28)
    plt.xlabel('x1', fontsize=28)
    plt.ylabel('x2', fontsize=28)
    plt.tick_params(axis='x', labelsize=26)
    plt.tick_params(axis='y', labelsize=26)
    plt.legend(fontsize=26, loc='upper left')
    # plt.show()
