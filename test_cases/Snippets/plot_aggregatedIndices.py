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
param = [r"x_1", r"x_2", r"x_3"]
n = len(param)

output = tecplot_reader(sensitivity_aggr_file, n * 6)

s_min, s, s_max, s_t_min, s_t, s_t_max = np.split(np.array(output).flatten(), 6)

objects = []
conf = [[], []]
indices = []
for i, p in enumerate(param):
    objects.append([r"$S_{" + p + r"}$", r"$S_{T_{" + p + r"}}$"])

    ind = np.array([s[i], s_t[i]])
    indices.append(ind)

    i_min = ind - np.array([s_min[i], s_t_min[i]])
    i_max = np.array([s_max[i], s_t_max[i]]) - ind
    conf[0].append(i_min)
    conf[1].append(i_max)

y_pos = np.arange(2 * n)
indices = np.array(indices).flatten()
conf = np.array(conf).reshape((2, 2 * n))

objects = [item for sublist in objects for item in sublist]

fig = plt.figure('Aggregated Indices')

plt.bar(y_pos, indices, yerr=conf, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.ylabel("Sobol' aggregated indices", fontsize=20)
plt.xlabel("Input parameters", fontsize=20)
fig.tight_layout()
plt.show()
