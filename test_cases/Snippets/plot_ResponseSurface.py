#!/usr/bin/env python
# coding:utf-8
"""Post processing QoI.

Allows response surface visualization in 2D and 3D.
It works on 0D and 1D output.
Addapt this script to your case.

"""

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
import json
import itertools

print("2D input function post processing")

# ---- Configuration variables ---- #
p1 = {'name': "x1", 'label': r"$x_1$",
      'data': [], 'data_doe': []}
p2 = {'name': "x2", 'label': r"$x_2$",
      'data': [], 'data_doe': []}
z = {'name': "f", 'label': r"$f\;$", 'data': None, 'data_doe': None}
x = {'name': "x", 'label': "Curvilinear abscissa (m)", 'data': None}
int_z = {'data': [], 'data_doe': []}
analytical = False
prediction = True
post_processing = False
len_sampling = 20
len_resample = 10
len_doe = len_sampling + len_resample
len_prediction = 625
nb_samples = len_prediction if prediction else len_doe
output_shape = '0D'  # 1D
snap_path = './output/snapshots/'
pred_path = './output/predictions/Newsnap'
reshape = False
idx_init = 8
idx_end = 9
color = True
c_map = cm.viridis if color else cm.gray


def header_reader(p_names, header_file):
    """Header reader.

    :param list(str) p_names: first param name
    :param str header_file: file path
    :return: parameters values
    :rtype: lst(floats)
    """
    with open(header_file, 'r') as fd:
        params = json.load(fd)
        p = np.array([params.get(n, 0.) for n in p_names])
    return p


def tecplot_reader(file, nb_var):
    """Tecplot reader.

    :param str file: file path
    :param int nb_var: number of variables to extract
    :return: Extracted variables
    :rtype: np.array(np.arrays)
    """
    arrays = []
    with open(file, 'r') as a:
        rest_of_file = itertools.islice(a, 3, None)
        for line in rest_of_file:
            arrays.append([float(s) for s in line.split()])

    arrays = np.concatenate(arrays)
    output = np.split(arrays, nb_var)

    return output


def integral_processing(file, header_file, output_shape):
    """Computes integral of output.

    Computes the integral on the output if 2D, returns value for 1D.

    :param str file: output file path
    :param str header_file: reader path
    :param str output_shape: type of output
    :return: x, y, z
    :rtype: np.array, np.array, float
    """
    if output_shape == '0D':
        x = [0, 1]
        z = np.array([tecplot_reader(file, 1),
                                  tecplot_reader(file, 1)]).flatten()
    else:
        x, z = tecplot_reader(file, 2)

    if reshape:
        x = x[idx_init:idx_end]
        z = z[idx_init:idx_end]

    int_f = np.trapz(z, x) /\
            (np.max(x) - np.min(x))

    return x, z, int_f


# Get the integral and header for sampling or predictions
for i in range(nb_samples):
    if prediction:

        if i < 10:
            index = '000' + str(i)
        elif 9 < i < 100:
            index = '00' + str(i)
        elif 99 < i < 1000:
            index = '0' + str(i)
        else:
            index = str(i)

        file = pred_path + index + '/function.dat'
        header_file = pred_path + index + '/param.json'
        x['data'], z['data'], int_f = integral_processing(file,
                                                                  header_file,
                                                                  output_shape)
        int_z['data'].append(int_f)

    else:
        file = snap_path + str(i) + '/batman-data/function.dat'
        header_file = snap_path + str(i) + '/batman-data/param.json'
        x['data'], z['data'], int_f = integral_processing(file,
                                                                  header_file,
                                                                  output_shape)
        int_z['data'].append(int_f)

        if post_processing:
            # Split every 1000 for Tecplot to read the file
            x_splitted = np.split(x['data'], nb_value // 1000 + 1)
            z_splitted = np.split(z['data'], nb_value // 1000 + 1)
            # Filter only the extrados
            file = snap_path + str(i) + '/batman-data/reshaped_function.dat'
            with open(file, 'w') as f:
                f.write('TITLE = " Reshaped output " \n')
                f.write('VARIABLES = "x" "z"  \n')
                f.write('ZONE T = "zone1 " , I={}, F=BLOCK \n'.format(nb_value))
                for x_split in x_splitted:
                    string_list = ["{:.7E}".format(val) for val in x_split]
                    f.write('\t'.join(string_list) + '\t')
                f.writelines('\n')
                for z_split in z_splitted:
                    string_list = ["{:.7E}".format(val) for val in z_split]
                    f.write('\t'.join(string_list) + '\t')

    print("File: {}".format(file))
    int_f = int_z['data'][-1]
    print("Integral Z: {}".format(int_f))

    a, b = header_reader([p1['name'], p2['name']], header_file)

    p1['data'].append(a)
    p2['data'].append(b)

    print("With Header -> {}: {}, {}: {}\n".format(p1['name'], a,
                                                   p2['name'], b))

# Get DOE from param.json
for i in range(len_doe):
    header_file = snap_path + str(i) + '/batman-data/param.json'

    a, b = header_reader([p1['name'], p2['name']], header_file)

    p1['data_doe'].append(a)
    p2['data_doe'].append(b)

    file = snap_path + str(i) + '/batman-data/function.dat'

    x['data'], z['data_doe'], int_f = integral_processing(file,
                                                                  header_file,
                                                                  output_shape)
    int_z['data_doe'].append(int_f)

p1['data'] = np.array(p1['data'])
p2['data'] = np.array(p2['data'])

# If analytical function change the equation
if analytical:
    int_z['data'] = -1.0 - \
        np.sin(p1['data']) * (np.power(np.sin(p1['data'] * p1['data'] / np.pi), 20.)) - \
        np.sin(p2['data']) * \
        (np.power(np.sin(2 * p2['data'] * p2['data'] / np.pi), 20.))

# Plot figures
#plt.rc('text', usetex=True)
#plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['SF-UI-Text-Light']})
bound_z = np.linspace(-2.76, -0.96, 50, endpoint=True)

fig = plt.figure('Response Surface_2D')
plt.tricontourf(p1['data'], p2['data'], int_z['data'],
                antialiased=True, cmap=c_map)
if not analytical:
    plt.plot(p1['data_doe'][0:len_sampling], p2[
             'data_doe'][0:len_sampling], 'ko')
    plt.plot(p1['data_doe'][len_sampling:], p2[
             'data_doe'][len_sampling:], 'r^')

cbar = plt.colorbar()
cbar.set_label(z['label'], fontsize=28)
plt.xlabel(p1['label'], fontsize=28)
plt.ylabel(p2['label'], fontsize=28)
plt.tick_params(axis='x', labelsize=28)
plt.tick_params(axis='y', labelsize=28)
cbar.ax.tick_params(labelsize=28)
plt.show()

fig = plt.figure('Response Surface_3D')
axis = fig.gca(projection='3d')
if not analytical:
    axis.scatter(p1['data_doe'][0:len_sampling],
                 p2['data_doe'][0:len_sampling],
                 int_z['data_doe'][0:len_sampling],
                 c='k', marker='o')
    axis.scatter(p1['data_doe'][len_sampling:],
                 p2['data_doe'][len_sampling:],
                 int_z['data_doe'][len_sampling:],
                 c='r', marker='^')
surface = axis.plot_trisurf(p1['data'], p2['data'], int_z['data'],
                            cmap=c_map, antialiased=False,
                            linewidth=0, alpha=0.5)
cbar = plt.colorbar(surface)
axis.set_zlabel(z['label'], fontsize=24, labelpad=20)
plt.xlabel(p1['label'], fontsize=26, labelpad=22)
plt.ylabel(p2['label'], fontsize=26, labelpad=22)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.tick_params(axis='z', labelsize=28)
plt.show()

