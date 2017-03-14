#!/usr/bin/env python
# coding:utf-8
"""Post processing output.

This scrit uses matplotlib to plot the QoI (1D output)
Addapt this script to your case.

"""

"""Plot output file."""
import matplotlib.pyplot as plt
import numpy as np

# Configuration
x = {'label': "Curvilinear abscissa (m)", 'data': None}
z = {'label': "Water elevation (m)", 'data': None}
format = 'fmt_tp'
file = "./output/snapshots/5/batman-data/function.dat"


def tecplot_reader(file, nb_var):
    """Tecplot reader."""
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

if format == 'fmt_tp':
    x['data'], z['data'] = tecplot_reader(file, 2)
else:
    x['data'], z['data'] = np.loadtxt(file, unpack=True)

print("Check data: ({}[2]: {}, {}[2]: {})".format(x['label'], x['data'][0], z['label'], z['data'][0]))

# Plot figure
plt.rc('text', usetex=True)
# plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['SF-UI-Text-Light']})

fig = plt.figure('Output')
plt.plot(x['data'], z['data'], linestyle='None', marker='^', color='k')
plt.ylabel(z['label'], fontsize=26)
plt.xlabel(x['label'], fontsize=26)
plt.tick_params(axis='x', labelsize=26)
plt.tick_params(axis='y', labelsize=26)
# plt.ylim(0, 1200)
# plt.text(-50, 600, 'Pressure side', fontsize=20)
# plt.text(30, 600, 'Suction side', fontsize=20)
# plt.xlim(-62, 86)
plt.show()
