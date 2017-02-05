#!/usr/bin/env python
# coding:utf-8
"""Plot aggregated indices.

Reads ``sensitivity_aggregated.dat`` and plot first and total order
indices with their confidence intervales.
"""
import numpy as np
import matplotlib.pyplot as plt


def tecplot_reader(file, nb_var):
    """Tecplot reader.

    :param str file: file path
    :param int nb_var: number of variables to extract
    :return: Extracted variables
    :rtype: np.array(np.arrays)
    """
    arrays = []
    append = arrays.append
    with open(file, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < 3:
                continue
            else:
                append([float(s) for s in line.split()])

    arrays = np.concatenate(arrays)
    output = np.split(arrays, nb_var)

    return output

path = './output/uq/'
sensitivity_aggr_file = path + 'sensitivity_aggregated.dat'
param = {'p1': {'name': 'x_1', 's_min': None, 's': None,'s_max': None,
                's_t_min': None, 's_t': None, 's_t_max': None},
         'p2': {'name': 'x_2', 's_min': None, 's': None,'s_max': None,
                's_t_min': None, 's_t': None, 's_t_max': None},}
n = len(param)

output = tecplot_reader(sensitivity_aggr_file, n * 6)

s_min, s, s_max, s_t_min, s_t, s_t_max = np.split(np.array(output).flatten(), 6)

objects = []
conf = [[], []]
indices = []
for i, p in enumerate(param):
    param[p]['s_min'] = s_min[i]
    param[p]['s'] = s[i]
    param[p]['s_max'] = s_max[i]
    param[p]['s_t_min'] = s_t_min[i]
    param[p]['s_t'] = s_t[i]
    param[p]['s_t_max'] = s_t_max[i]
    objects.append([r"$S_{" + param[p]['name'] + r"}$", r"$S_{T_{" + param[p]['name'] + r"}}$"])

    ind = np.array([param[p]['s'], param[p]['s_t']])
    indices.append(ind)

    i_min = ind - np.array([param[p]['s_min'], param[p]['s_t_min']])
    i_max = np.array([param[p]['s_max'], param[p]['s_t_max']]) - ind
    conf[0].append(i_min)
    conf[1].append(i_max)

y_pos = np.arange(2* n)
indices = np.array(indices).flatten()
conf = np.array(conf).reshape((n, 4))

objects = [item for sublist in objects for item in sublist]

fig = plt.figure('Aggregated Indices')

plt.bar(y_pos, indices, yerr=conf, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Sobol' aggregated indices")
plt.show()
