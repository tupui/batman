# coding: utf8
import copy
from batman.driver import Driver
from mock import patch


@patch("matplotlib.pyplot.show")
def test_optimization(mock_show, tmp, branin_data, settings_ishigami):
    test_settings = copy.deepcopy(settings_ishigami)
    init_size = len(branin_data.space)
    res_size = 2
    test_settings['space']['sampling']['init_size'] = init_size
    test_settings['space']['sampling']['discrete'] = 1
    test_settings['space']['resampling']['method'] = 'optimization'
    test_settings['space']['resampling']['resamp_size'] = res_size
    test_settings['space']['resampling']['delta_space'] = 0.1
    test_settings['space']['corners'] = branin_data.space.corners
    test_settings['snapshot']['plabels'] = ['x1', 'x2']
    test_settings['snapshot']['provider'] = {
        "type": "function",
        "module": "batman.tests.plugins",
        "function": "f_branin"
    }

    driver = Driver(test_settings, tmp)
    driver.sampling()
    driver.resampling()

    # import itertools
    # from scipy.stats import norm
    # from sklearn import preprocessing
    # import numpy as np
    # num = 25
    # x = np.linspace(-7, 10, num=num)
    # y = np.linspace(0, 15, num=num)
    # points = np.array([(float(i), float(j)) for i, j in itertools.product(x, y)])
    # x = points[:, 0].flatten()
    # y = points[:, 1].flatten()
    # pred, sigma = driver.prediction(points=points)
    # pred = np.array(pred).flatten()
    # sigma = np.array(sigma).flatten()
    # space = np.array(driver.space[:])

    # arg_min = np.argmin(branin_data.target_space)
    # min_value = branin_data.target_space[arg_min]

    # # Expected improvement
    # ei = np.empty_like(pred)
    # for i, p, s in zip(range(num * num), pred, sigma):
    #     ei[i] = (min_value - p) * norm.cdf((min_value - p) / s)\
    #         + s * norm.pdf((min_value - p) / s)

    # # Discrepancy
    # disc = np.empty_like(pred)
    # for i, p in enumerate(points):
    #     disc[i] = 1 / driver.space.discrepancy(np.vstack([space, p]))

    # scale_sigma = preprocessing.StandardScaler().fit(sigma.reshape(-1, 1))
    # scale_disc = preprocessing.StandardScaler().fit(disc.reshape(-1, 1))
    # min_value = scale_disc.transform(1 / driver.space.discrepancy(space).reshape(1, -1))

    # # EGO discrepancy...
    # ei_disc = np.empty_like(pred)
    # for i_s, p in zip(enumerate(sigma), disc):
    #     i, s = i_s
    #     s = scale_sigma.transform(s.reshape(1, -1))
    #     p = scale_disc.transform(p.reshape(1, -1))
    #     diff = min_value - p

    #     ei_disc[i] = diff * norm.cdf(diff / s)\
    #         + s * norm.pdf(diff / s)

    # # Sigma + Discrepancy
    # scale_sigma = preprocessing.scale(sigma.reshape(-1, 1))
    # scale_disc = preprocessing.scale(disc.reshape(-1, 1))
    # sigma_disc = scale_sigma + scale_disc

    # print(sigma_disc)

    # # Plotting
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # color = True
    # c_map = cm.viridis if color else cm.gray
    # fig = plt.figure('Efficient Global Optimization', figsize=(20, 5))
    # plt.subplot(131)
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # bounds = np.linspace(-17, 300., 30, endpoint=True)
    # plt.tricontourf(x, y, pred, bounds, antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$f(x_1, x_2)$')
    # plt.ylabel(r'$x_2$', fontsize=24)
    # plt.tick_params(axis='y')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')

    # plt.subplot(132)
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # plt.tricontourf(x, y, sigma, antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\sigma(x_1, x_2)$')
    # plt.xlabel(r'$x_1$', fontsize=24)
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')

    # plt.subplot(133)
    # plt.plot(space[:init_size, 0], space[:init_size, 1], 'ko')
    # plt.plot(space[init_size:, 0], space[init_size:, 1], 'm^')
    # plt.plot(-3.68928528, 13.62998774, 'r<')
    # plt.tricontourf(x, y, ei, antialiased=True, cmap=c_map)
    # cbar = plt.colorbar()
    # cbar.set_label(r'$\mathbb{E}[I(x_1, x_2)]$')
    # for txt, point in enumerate(space):
    #     plt.annotate(txt, point, textcoords='offset points')

    # fig.tight_layout()
    # path = 'expected_improvement_' + str(res_size) + '.pdf'
    # fig.savefig(path, transparent=True, bbox_inches='tight')
    # plt.show()
