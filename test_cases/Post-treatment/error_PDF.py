#!/usr/bin/env python
#-*-coding:utf-8-*
"""Post processing UQ.

This scrit uses matplotlib to plot Probability Density Function (PDF).
Addapt this script to your case.

"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import openturns as ot
from scipy.stats import ks_2samp


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
color = True
c_map = cm.viridis if color else cm.gray

# Files path
path = './results/output_100/uq/'
pdf_file = path + 'pdf.dat'
p1 = {'name': "Q", 's_1': None, 's_t': None}
p2 = {'name': "Ks", 's_1': None, 's_t': None}
z = {'name': "Z", 'label': r"$Z$ (m)", 'data': None, 'shape': 14}
x = {'name': "x", 'label': "Curvilinear abscissa (km)", 'data': None}
pdf_discretization = 100
get_pdf = 8  # n-th element of x
bound_pdf = np.linspace(0., 1., 50, endpoint=True)

x_pdf, z['data'], pdf = tecplot_reader(pdf_file, 3)

# Reshape data
x_pdf_matrix = x_pdf.reshape((pdf_discretization, z['shape']))
z_matrix = z['data'].reshape((pdf_discretization, z['shape']))
pdf_matrix = pdf.reshape((pdf_discretization, z['shape']))

# Get a specific PDF
pdf_array = pdf_matrix[:, get_pdf + 1]
z_array = z_matrix[:, get_pdf + 1]

pdf_array = np.array(pdf_array)
z_array = np.array(z_array)
idx = np.argsort(z_array)
z_array = z_array[idx]
pdf_array = pdf_array[idx]


# Get database
print("Reading data...")
output_file = "./results/model_appr.dat"
data_output = np.loadtxt(output_file, unpack=False)
print("Data read.")

sample = ot.NumericalSample(data_output[:, 9].reshape((100000, -1)))
kernel = ot.KernelSmoothing()
pdf = kernel.build(sample, True)
data_points = np.array(pdf.getSample(1000))
data_pdf = np.array(pdf.computePDF(z_array.reshape((pdf_discretization, 1))))

err = np.sum((data_pdf.flatten() - pdf_array) ** 2)
var = np.sum((data_pdf.flatten() - np.average(data_pdf)) ** 2)

Q2 = 1 - err / var
print("Error Q2: {}".format(Q2))

stats, pvalue = ks_2samp(pdf_array, np.concatenate(data_pdf))
print("Kolmogorov test -> stats: {}, pvalue: {}".format(stats, pvalue))


# Plot figures
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['SF-UI-Text-Light']})

fig = plt.figure('Extracted PDF: ' + str(get_pdf+1))
plt.plot(z_array, pdf_array, color='k', ls='-', linewidth=3, label="Model PDF: " + str(get_pdf+1))
plt.plot(z_array, data_pdf, color='k', ls='-.', linewidth=3, label="Monte Carlo PDF: " + str(get_pdf+1))
plt.xlabel(z['label'], fontsize=26)
plt.ylabel("PDF", fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
plt.legend(fontsize=26, loc='upper right')
plt.show()
