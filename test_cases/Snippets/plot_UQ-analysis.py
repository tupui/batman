#!/usr/bin/env python
#-*-coding:utf-8-*
"""Post processing UQ.

This scrit uses matplotlib to plot Probability Density Function (PDF),
moments and sensitivity map.
Addapt this script to your case.

"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def tecplot_reader(file, nb_var):
    """Tecplot reader.

    :param str file: file path
    :param int nb_var: number of variables to extract
    :return: Extracted variables
    :rtype: np.array(np.arrays)
    """
    arrays = []
    with open(file, 'r') as a:
        for idx, line in enumerate(a.readlines()):
            if idx < 3:
                continue
            else:
                arrays.append([float(s) for s in line.split()])

    arrays = np.concatenate(arrays)
    output = np.split(arrays, nb_var)

    return output

print("Post processing UQ results...")

# Color
color = False
c_map = cm.viridis if color else cm.gray

# Files path
path = './output/uq/'
pdf_file = path + 'pdf.dat'
moment_file = path + 'moment.dat'
sensitivity_file = path + 'sensitivity.dat'
p1 = {'name': "Q", 's_1': None, 's_t': None}
p2 = {'name': "Ks", 's_1': None, 's_t': None}
z = {'name': "Z", 'label': r"$Z$ (m)", 'data': None, 'shape': 400}
x = {'name': "x", 'label': "Curvilinear abscissa (km)", 'data': None}
pdf_discretization = 100
get_pdf = 8  # n-th element of x
bound_pdf = np.linspace(0., 1., 50, endpoint=True)

x_pdf, z['data'], pdf = tecplot_reader(pdf_file, 3)
x['data'], mini, sd_min, mean, sd_max, maxi = tecplot_reader(moment_file, 6)
_, p1['s_1'], p1['s_t'], p2['s_1'], p2['s_t'] = tecplot_reader(sensitivity_file, 5)

# Reshape data
x_pdf_matrix = x_pdf.reshape((pdf_discretization, z['shape']))
z_matrix = z['data'].reshape((pdf_discretization, z['shape']))
pdf_matrix = pdf.reshape((pdf_discretization, z['shape']))

# Get a specific PDF
pdf_array = []
z_array = []
pdf_array = pdf_matrix[:, 9]
z_array = z_matrix[:, 9]

pdf_array = np.array(pdf_array)
z_array = np.array(z_array)
idx = np.argsort(z_array)
z_array = z_array[idx]
pdf_array = pdf_array[idx]

# Plot figures
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['SF-UI-Text-Light']})

fig = plt.figure('PDF')
plt.contourf(x_pdf_matrix/1000, z_matrix, pdf_matrix, bound_pdf, cmap=c_map)
cbar = plt.colorbar()
cbar.set_label(r"PDF")
plt.xlabel(x['label'], fontsize=26)
plt.ylabel(z['label'], fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.show()

fig = plt.figure('Extracted PDF: ' + str(get_pdf+1))
plt.plot(z_array, pdf_array, color='k', ls='-', linewidth=3, label="Extracted PDF: " + str(get_pdf+1))
plt.xlabel(z['label'], fontsize=26)
plt.ylabel("PDF", fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.legend(fontsize=26, loc='upper right')
plt.show()

fig = plt.figure('Moments')
plt.plot(x['data']/1000, mini, color='k', ls='--', linewidth=3, label="Min")
plt.plot(x['data']/1000, sd_min, color='k', ls='-.', linewidth=3, label="Standard Deviation")
plt.plot(x['data']/1000, mean, color='k', ls='-', linewidth=3, label="Mean")
plt.plot(x['data']/1000, sd_max, color='k', ls='-.', linewidth=3, label="Standard Deviation")
plt.plot(x['data']/1000, maxi, color='k', ls='--', linewidth=3, label="Max")
plt.xlabel(x['label'], fontsize=26)
plt.ylabel(z['label'], fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.legend(fontsize=26, loc='upper right')
plt.show()

fig = plt.figure('PDF-Moments')
plt.plot(x['data']/1000, mini, color='k', ls='--', linewidth=3, label="Min")
plt.plot(x['data']/1000, sd_min, color='k', ls='-.', linewidth=3, label="Standard Deviation")
plt.plot(x['data']/1000, mean, color='k', ls='-', linewidth=3, label="Mean")
plt.plot(x['data']/1000, sd_max, color='k', ls='-.', linewidth=3, label="Standard Deviation")
plt.plot(x['data']/1000, maxi, color='k', ls='--', linewidth=3, label="Max")
plt.legend(fontsize=26, loc='upper right')
plt.contourf(x_pdf_matrix/1000, z_matrix, pdf_matrix, bound_pdf, cmap=c_map, alpha=0.5)
cbar = plt.colorbar()
cbar.set_label(r"PDF")
plt.xlabel(x['label'], fontsize=26)
plt.ylabel(z['label'], fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.show()

fig = plt.figure('Sensitivity Map')
plt.plot(x['data']/1000, p1['s_1'], color='k', ls='--', linewidth=3, label=r"$S_{" + p1['name'] + r"}$")
plt.plot(x['data']/1000, p1['s_t'], color='k', ls='-.', linewidth=3, label=r"$S_{T_{" + p1['name'] + r"}}$")
plt.plot(x['data']/1000, p2['s_1'], color='k', ls='-', linewidth=3, label=r"$S_{" + p2['name'] + r"}$")
plt.plot(x['data']/1000, p2['s_t'], color='k', ls=':', linewidth=3, label=r"$S_{T_{" + p2['name'] + r"}}$")
plt.xlabel(x['label'], fontsize=26)
plt.ylabel(r"Indices", fontsize=26)
plt.ylim(-0.1, 1.1)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.legend(fontsize=26, loc='center right')
plt.show()
